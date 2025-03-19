use crate::core::{self, Size};
use crate::image::atlas::{self, Atlas};
use crate::image::staging::StagingBuffer;

use std::sync::Arc;
use std::time::Instant;

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
        
        // Now process pending uploads
        let processed = self.staging.process_uploads(&mut self.atlas, device, encoder);
        
        if processed > 0 {
            log::debug!("Processed {} uploads, {} still pending", 
                      processed, self.staging.pending_count());
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
        // Check if pending growth is needed before uploading
        if self.pending_growth > 0 {
            log::debug!("Growing atlas by {} layers before raster upload", self.pending_growth);
            self.atlas.grow_if_needed(self.pending_growth, device, encoder);
            self.pending_growth = 0;
        }
        
        // Process any pending uploads first
        if self.staging.pending_count() > 0 {
            let processed = self.process_pending_uploads(device, encoder);
            if processed > 0 {
                log::debug!("Processed {} pending uploads before raster upload", processed);
            }
        }
        
        // Now do the main upload - this happens last so we can return the entry
        let entry = self.raster.upload(device, encoder, handle, &mut self.atlas);
        
        entry
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
