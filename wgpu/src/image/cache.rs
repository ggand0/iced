use crate::core::{self, Size};
use crate::image::atlas::{self, Atlas};
use crate::image::staging::StagingBuffer;

use std::sync::Arc;
use std::time::Instant;
use std::hash::Hash;

#[derive(Debug)]
pub struct Cache {
    atlas: Atlas,
    #[cfg(feature = "image")]
    raster: crate::image::raster::Cache,
    #[cfg(feature = "svg")]
    vector: crate::image::vector::Cache,
    
    // Add staging buffer for async uploads
    staging: StagingBuffer,
    
    // Track pending atlas growth to apply on next submission
    pending_growth: usize,
}

impl Cache {
    pub fn new(
        device: &wgpu::Device,
        backend: wgpu::Backend,
        layout: Arc<wgpu::BindGroupLayout>,
    ) -> Self {
        Self {
            atlas: Atlas::new(device, backend, layout),
            #[cfg(feature = "image")]
            raster: crate::image::raster::Cache::default(),
            #[cfg(feature = "svg")]
            vector: crate::image::vector::Cache::default(),
            staging: StagingBuffer::new(),
            pending_growth: 0,
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        self.atlas.bind_group()
    }

    pub fn layer_count(&self) -> usize {
        self.atlas.layer_count()
    }

    #[cfg(feature = "image")]
    pub fn measure_image(&mut self, handle: &core::image::Handle) -> Size<u32> {
        self.raster.load(handle).dimensions()
    }

    #[cfg(feature = "svg")]
    pub fn measure_svg(&mut self, handle: &core::svg::Handle) -> Size<u32> {
        self.vector.load(handle).viewport_dimensions()
    }

    // Process any pending uploads - call this during the render pass
    pub fn process_pending_uploads(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> usize {
        // First check if growth is needed
        if self.pending_growth > 0 {
            log::debug!("Growing atlas by {} layers before uploads", self.pending_growth);
            self.atlas.grow_if_needed(self.pending_growth, device, encoder);
            self.pending_growth = 0;
        }
        
        // Process pending uploads in parallel
        let max_parallel = 8; // Process up to 8 uploads in parallel
        let processed = self.staging.process_uploads_parallel(
            &mut self.atlas, device, encoder, max_parallel);
        
        if processed > 0 {
            log::debug!("Processed {} uploads in parallel", processed);
        }
        
        processed
    }

    #[cfg(feature = "image")]
    pub fn upload_raster(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &core::image::Handle,
    ) -> Option<&atlas::Entry> {
        // Generate a unique ID for this handle for logging
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        handle.id().hash(&mut hasher);
        let handle_id = hasher.finish();
        
        // Check if entry exists *without retaining a reference*
        let entry_exists = self.raster.get_cached_device_entry(handle).is_some();
        
        if entry_exists {
            log::trace!("Cache hit for image {:#x}", handle_id);
            // We don't return the reference directly, we'll get it again at the end
        } else {
            // Entry doesn't exist, determine if parallel processing is appropriate
            let should_process_in_parallel = match handle {
                core::image::Handle::Bytes(_, bytes) if bytes.len() > 50_000 => true,
                core::image::Handle::Path(_, _) => true, 
                _ => false,
            };
            
            if should_process_in_parallel {
                // Extract the bytes for processing
                let data: Vec<u8> = match handle {
                    core::image::Handle::Bytes(_, bytes) => bytes.to_vec(),
                    core::image::Handle::Path(_, path) => {
                        std::fs::read(path).unwrap_or_else(|e| {
                            log::error!("Failed to read image from path {:?}: {}", path, e);
                            vec![]
                        })
                    },
                    core::image::Handle::Rgba { pixels, .. } => pixels.to_vec(),
                };
                
                if !data.is_empty() {
                    log::info!("Submitting image {:#x} for parallel processing ({} bytes)", 
                             handle_id, data.len());
                    self.staging.submit_for_processing(data, handle_id);
                }
            }
            
            // Handle pending growth
            if self.pending_growth > 0 {
                self.atlas.grow_if_needed(self.pending_growth, device, encoder);
                self.pending_growth = 0;
            }
            
            // Process parallel uploads
            if self.staging.pending_count() > 0 {
                let processed = self.staging.process_uploads_parallel(
                    &mut self.atlas, device, encoder, 8);
                
                if processed > 0 {
                    log::debug!("Processed {} uploads in parallel", processed);
                }
            }
            
            // Check if our image was processed in parallel without retaining a reference
            let processed_in_parallel = self.raster.get_cached_device_entry(handle).is_some();
            
            if !processed_in_parallel {
                // Fall back to synchronous upload
                log::debug!("Using synchronous upload for image {:#x}", handle_id);
                let _ = self.raster.upload(device, encoder, handle, &mut self.atlas);
            }
        }
        
        // Now get the entry at the end, after all modifications
        self.raster.get_cached_device_entry(handle)
    }

    #[cfg(feature = "svg")]
    pub fn upload_vector(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &core::svg::Handle,
        color: Option<core::Color>,
        size: [f32; 2],
        scale: f32,
    ) -> Option<&atlas::Entry> {
        let _upload_start = Instant::now(); // Add underscore to suppress warning
        
        // TODO: Implement async vector upload similar to raster
        // For now, use the existing synchronous implementation
        let result = self.vector.upload(
            device,
            encoder,
            handle,
            color,
            size,
            scale,
            &mut self.atlas,
        );
        
        if let Ok(mut tracker) = crate::image::IMAGE_DISPLAY_TRACKER.lock() {
            tracker.record_upload_complete();
        }
        
        result
    }

    pub fn trim(&mut self) {
        #[cfg(feature = "image")]
        self.raster.trim(&mut self.atlas);

        #[cfg(feature = "svg")]
        self.vector.trim(&mut self.atlas);
    }

    // Add debug information
    pub fn log_state(&self) {
        log::debug!("Cache state: Atlas has {} layers, {} pending uploads", 
                  self.atlas.layer_count(), self.staging.pending_count());
    }
}
