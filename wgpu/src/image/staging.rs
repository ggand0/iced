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
    total_queued: usize,   // Add stats tracking fields
    total_processed: usize,
}

impl StagingBuffer {
    pub fn new() -> Self {
        Self {
            pending_uploads: VecDeque::with_capacity(10),
            max_batch_size: 4, // Process up to 4 uploads per frame
            total_queued: 0,
            total_processed: 0,
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
}