use std::collections::VecDeque;
use std::time::Instant;

use crate::image::atlas;

use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Sender, Receiver};
use crate::image::atlas::Atlas;
use crate::core::Size;

// Background worker pool for parallel image processing
#[derive(Debug)]
struct ImageWorkerPool {
    sender: Sender<ImageJob>,
    workers: Vec<thread::JoinHandle<()>>,
    processed_queue: Arc<Mutex<VecDeque<ProcessedImage>>>,
}

struct ImageJob {
    data: Vec<u8>,
    id: u64,
    processed_queue: Arc<Mutex<VecDeque<ProcessedImage>>>,
}

#[derive(Debug)]
struct ProcessedImage {
    data: Vec<u8>,
    width: u32,
    height: u32,
    id: u64,
    processing_time: std::time::Duration,
}

impl ImageWorkerPool {
    fn new(num_workers: usize) -> Self {
        let (sender, receiver): (Sender<ImageJob>, Receiver<ImageJob>) = channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let processed_queue = Arc::new(Mutex::new(VecDeque::new()));
        
        let workers = (0..num_workers).map(|i| {
            let receiver = receiver.clone();
            let processed_queue = processed_queue.clone();
            
            thread::spawn(move || {
                log::debug!("Image worker {} started", i);
                
                while let Ok(job) = {
                    let receiver = receiver.lock().unwrap();
                    receiver.recv()
                } {
                    let start = Instant::now();
                    
                    // Actually process the image (decode, resize, etc.)
                    let processed = Self::process_image_data(job.data);
                    let (width, height, data) = processed;
                    
                    let processing_time = start.elapsed();
                    log::trace!("Worker {} processed image {} in {:?}", 
                              i, job.id, processing_time);
                    
                    // Add to processed queue
                    if let Ok(mut queue) = job.processed_queue.lock() {
                        queue.push_back(ProcessedImage {
                            data,
                            width,
                            height,
                            id: job.id,
                            processing_time,
                        });
                    }
                }
                
                log::debug!("Image worker {} stopped", i);
            })
        }).collect();
        
        Self {
            sender,
            workers,
            processed_queue,
        }
    }
    
    fn process_image_data(data: Vec<u8>) -> (u32, u32, Vec<u8>) {
        // Decode image using image crate
        #[cfg(feature = "image")]
        {
            use image::GenericImageView;
            
            if let Ok(img) = image::load_from_memory(&data) {
                let (width, height) = img.dimensions();
                let rgba = img.into_rgba8();
                return (width, height, rgba.into_raw());
            }
        }
        
        // Fallback empty image
        (1, 1, vec![255, 0, 255, 255])
    }
    
    fn submit(&self, data: Vec<u8>, id: u64) {
        let job = ImageJob {
            data,
            id,
            processed_queue: self.processed_queue.clone(),
        };
        
        let _ = self.sender.send(job);
    }
    
    fn get_processed(&self) -> Vec<ProcessedImage> {
        let mut result = Vec::new();
        
        if let Ok(mut queue) = self.processed_queue.lock() {
            while let Some(img) = queue.pop_front() {
                result.push(img);
            }
        }
        
        result
    }
}

/// A single queued texture upload
#[derive(Debug)]
struct PendingUpload {
    data: Vec<u8>,
    width: u32,
    height: u32,
    padding: u32,
    offset: usize,
    allocation: atlas::Allocation,
    queued_at: Instant,
}

/// Manages asynchronous texture uploads using staging buffers
#[derive(Debug)]
pub struct StagingBuffer {
    pending_uploads: VecDeque<PendingUpload>,
    max_batch_size: usize,
    total_queued: usize,   // Add stats tracking fields
    total_processed: usize,
    worker_pool: ImageWorkerPool,
}

impl StagingBuffer {
    pub fn new() -> Self {
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .min(4); // Use up to 4 worker threads
            
        log::info!("Creating image worker pool with {} threads", num_workers);

        Self {
            pending_uploads: VecDeque::with_capacity(10),
            max_batch_size: 4, // Process up to 4 uploads per frame
            total_queued: 0,
            total_processed: 0,
            worker_pool: ImageWorkerPool::new(num_workers),
        }
    }

    // Submit raw image data for parallel processing
    pub fn submit_for_processing(&mut self, data: Vec<u8>, id: u64) {
        self.worker_pool.submit(data, id);
    }

    // Check for processed images and queue them for upload
    pub fn collect_processed_images(&mut self, atlas: &mut crate::image::atlas::Atlas) -> usize {
        let processed = self.worker_pool.get_processed();
        let count = processed.len();
        
        if count > 0 {
            log::debug!("Collected {} processed images from worker threads", count);
        }
        
        for img in processed {
            // Try to allocate in the atlas
            if let Some(entry) = atlas.allocate(img.width, img.height) {
                // Extract allocation from Entry based on its actual structure
                // From the Entry enum definition you shared:
                let allocation = match entry {
                    // If Contiguous, we can directly use the allocation
                    atlas::entry::Entry::Contiguous(allocation) => allocation,
                    
                    // If Fragmented, we need to check if there are fragments and use the first one
                    // This might not be the right approach for all cases
                    atlas::entry::Entry::Fragmented { fragments, .. } => {
                        if let Some(fragment) = fragments.first() {
                            fragment.allocation.clone()
                        } else {
                            // No fragments available, we can't create a proper upload
                            log::error!("Cannot upload fragmented image with no fragments");
                            continue;
                        }
                    }
                };
                
                let upload = PendingUpload {
                    data: img.data,
                    width: img.width,
                    height: img.height,
                    padding: 0, 
                    offset: 0,
                    allocation,
                    queued_at: Instant::now(),
                };
                
                self.pending_uploads.push_back(upload);
            }
        }
        
        count
    }

    /// Queue a texture upload to be processed during the next frame submission
    pub fn queue_upload(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        padding: u32,
        offset: usize,
        allocation: atlas::Allocation,
    ) {
        // Add detailed logging
        log::debug!("Queueing texture upload: {}x{} pixels, {}/{} pending", 
                  width, height, self.pending_uploads.len(), self.pending_count());
        
        self.pending_uploads.push_back(PendingUpload {
            data: data.to_vec(), // Clone the data (could use a shared buffer in the future)
            width,
            height,
            padding,
            offset,
            allocation,
            queued_at: Instant::now(),
        });
        
        self.total_queued += 1;
    }

    /// Process queued uploads up to the batch size limit
    pub fn process_uploads(
        &mut self,
        atlas: &mut atlas::Atlas,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> usize {
        let start = Instant::now();
        let mut processed = 0;
        
        // Determine the number of uploads to process in this batch
        let pending_count = self.pending_uploads.len();
        let count = std::cmp::min(pending_count, self.max_batch_size);
        
        // Process the oldest uploads first
        for _ in 0..count {
            if let Some(upload) = self.pending_uploads.pop_front() {
                // Check if the allocation is valid before processing
                let allocation_layer = upload.allocation.layer();
                let layer_count = atlas.layer_count();
                
                if allocation_layer >= layer_count {
                    log::error!("SKIPPING INVALID UPLOAD: Layer {} exceeds atlas size {}", 
                               allocation_layer, layer_count);
                    continue;
                }
                
                // Upload to the atlas texture
                atlas.upload_allocation(
                    &upload.data,
                    upload.width,
                    upload.height,
                    upload.padding,
                    upload.offset,
                    &upload.allocation,
                    device,
                    encoder,
                );
                
                processed += 1;
            }
        }
        
        let processing_time = start.elapsed();
        if processed > 0 {
            // Always log for debugging
            log::info!("Processed {}/{} texture uploads in {:.2}ms (total: {}/{})", 
                     processed, self.pending_uploads.len() + processed,
                     processing_time.as_secs_f64() * 1000.0,
                     self.total_processed, self.total_queued);
        }
        
        processed
    }
    
    /// Get number of pending uploads
    pub fn pending_count(&self) -> usize {
        self.pending_uploads.len()
    }

    pub fn process_uploads_parallel(
        &mut self,
        atlas: &mut crate::image::atlas::Atlas,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        max_parallel: usize,
    ) -> usize {
        let start = Instant::now();
        let mut processed = 0;
        
        // Process uploads in parallel batches
        let batch_size = std::cmp::min(self.pending_uploads.len(), max_parallel);
        
        for _ in 0..batch_size {
            if let Some(upload) = self.pending_uploads.pop_front() {
                // Check if the allocation is valid before processing
                let allocation_layer = upload.allocation.layer();
                let layer_count = atlas.layer_count();
                
                if allocation_layer >= layer_count {
                    log::error!("SKIPPING INVALID UPLOAD: Layer {} exceeds atlas size {}", 
                               allocation_layer, layer_count);
                    continue;
                }
                
                // Prepare buffer for upload
                use wgpu::util::DeviceExt;
                
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("parallel image upload buffer"),
                    contents: &upload.data,
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
                
                // Get upload parameters
                let (x, y) = upload.allocation.position();
                let size = upload.allocation.size();
                
                // Execute copy operation
                encoder.copy_buffer_to_texture(
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: upload.offset as u64,
                            bytes_per_row: Some(4 * upload.width + upload.padding),
                            rows_per_image: Some(upload.height),
                        },
                    },
                    wgpu::ImageCopyTexture {
                        texture: atlas.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x,
                            y,
                            z: upload.allocation.layer() as u32,
                        },
                        aspect: wgpu::TextureAspect::default(),
                    },
                    wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth_or_array_layers: 1,
                    },
                );
                
                processed += 1;
                self.total_processed += 1;
            }
        }
        
        if processed > 0 {
            log::debug!("Processed {} uploads in parallel in {:?}", 
                      processed, start.elapsed());
        }
        
        processed
    }
}