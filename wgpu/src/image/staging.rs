use std::collections::VecDeque;
use std::time::Instant;

use crate::image::atlas;

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
}

impl StagingBuffer {
    pub fn new() -> Self {
        Self {
            pending_uploads: VecDeque::with_capacity(10),
            max_batch_size: 4, // Process up to 4 uploads per frame
        }
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
        self.pending_uploads.push_back(PendingUpload {
            data: data.to_vec(), // Clone the data (could use a shared buffer in the future)
            width,
            height,
            padding,
            offset,
            allocation,
            queued_at: Instant::now(),
        });
    }

    /// Process queued uploads up to the batch size limit
    pub fn process_uploads(
        &mut self,
        atlas: &mut atlas::Atlas,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> usize {
        let start = Instant::now();
        let count = self.pending_uploads.len().min(self.max_batch_size);
        
        if count == 0 {
            return 0;
        }

        let mut processed = 0;
        
        // Process the oldest uploads first
        for _ in 0..count {
            if let Some(upload) = self.pending_uploads.pop_front() {
                let queue_time = upload.queued_at.elapsed();
                
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
                
                // Log if uploads were queued for a long time
                if queue_time.as_millis() > 100 {
                    log::warn!("Texture upload queued for {}ms before processing", 
                              queue_time.as_millis());
                }
            }
        }
        
        let processing_time = start.elapsed();
        if processed > 0 && processing_time.as_millis() > 5 {
            log::debug!("Processed {} texture uploads in {:.2}ms", 
                      processed, processing_time.as_secs_f64() * 1000.0);
        }
        
        processed
    }
    
    /// Get number of pending uploads
    pub fn pending_count(&self) -> usize {
        self.pending_uploads.len()
    }
}