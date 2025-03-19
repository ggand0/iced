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

    // Process any pending texture uploads
    pub fn process_uploads(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // First, grow the atlas if needed from previous allocations
        if self.pending_growth > 0 {
            self.atlas.grow_if_needed(self.pending_growth, device, encoder);
            self.pending_growth = 0;
        }
        
        // Then process any queued uploads
        let start = Instant::now();
        let count = self.staging.process_uploads(&mut self.atlas, device, encoder);
        
        if count > 0 {
            let process_time = start.elapsed();
            if process_time.as_millis() > 10 {
                log::debug!("ASYNC UPLOADS: Processed {} texture uploads in {:.2}ms", 
                         count, process_time.as_secs_f64() * 1000.0);
            }
            
            if let Ok(mut tracker) = crate::image::IMAGE_DISPLAY_TRACKER.lock() {
                tracker.record_batch_upload_complete(count);
            }
        }
    }

    #[cfg(feature = "image")]
    pub fn upload_raster(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &core::image::Handle,
    ) -> Option<&atlas::Entry> {
        let upload_start = Instant::now();
        
        // If already in atlas, just return it
        if self.raster.has_device_entry(handle) {
            return self.raster.get_cached_device_entry(handle);
        }
        
        // Load if needed
        if !self.raster.has_cache_entry(handle) || !self.raster.has_host_memory(handle) {
            let _ = self.raster.load(handle);
        }
        
        // Try to process the host memory
        let mut did_queue_upload = false;
        
        // Check if we now have host memory after loading
        if self.raster.has_host_memory(handle) {
            // Process the host memory to device memory
            if let Some(dims) = self.raster.get_image_dimensions(handle) {
                let width = dims.width;
                let height = dims.height;
                
                // Allocate entry in atlas
                if let Some(entry) = self.atlas.allocate_entry(device, wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                }) {
                    // Process the upload based on whether entry is contiguous or fragmented
                    let should_add_entry = match &entry {
                        atlas::Entry::Contiguous(allocation) => {
                            // Queue contiguous upload using staging buffer
                            if let Some(bytes) = self.raster.get_image_bytes(handle) {
                                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
                                let padded_width = ((4 * width as usize) + (align - 1)) & !(align - 1);
                                let padded_size = padded_width * height as usize;
                                let mut padded_data = vec![0; padded_size];
                                
                                // Copy rows with padding
                                for y in 0..height as usize {
                                    let src_offset = y * 4 * width as usize;
                                    let dst_offset = y * padded_width;
                                    
                                    padded_data[dst_offset..dst_offset + 4 * width as usize]
                                        .copy_from_slice(&bytes[src_offset..src_offset + 4 * width as usize]);
                                }
                                
                                let padding = padded_width - 4 * width as usize;
                                
                                // Queue the upload
                                self.staging.queue_upload(
                                    &padded_data,
                                    width,
                                    height,
                                    padding as u32,
                                    0,
                                    allocation.clone(),
                                );
                                
                                // Note the needed atlas growth
                                let layers_before = self.atlas.layer_count();
                                let allocation_layer = allocation.layer();
                                
                                if allocation_layer >= layers_before {
                                    self.pending_growth = 
                                        self.pending_growth.max(allocation_layer + 1 - layers_before);
                                }
                                
                                true
                            } else {
                                false
                            }
                        },
                        atlas::Entry::Fragmented { fragments, .. } => {
                            // Handle each fragment separately
                            if let Some(bytes) = self.raster.get_image_bytes(handle) {
                                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
                                
                                for fragment in fragments {
                                    let fragment_allocation = &fragment.allocation;
                                    let fragment_width = fragment_allocation.size().width;
                                    let fragment_height = fragment_allocation.size().height;
                                    let fragment_x = fragment.position.0;
                                    let fragment_y = fragment.position.1;
                                    
                                    // Create padded buffer for this fragment
                                    let fragment_padded_width = ((4 * fragment_width as usize) + (align - 1)) & !(align - 1);
                                    let fragment_size = fragment_padded_width * fragment_height as usize;
                                    let mut fragment_data = Vec::with_capacity(fragment_size);
                                    
                                    // Copy fragment data row by row
                                    for y in 0..fragment_height as usize {
                                        let src_y = fragment_y as usize + y;
                                        
                                        if src_y < height as usize {
                                            let src_offset = src_y * 4 * width as usize + fragment_x as usize * 4;
                                            let src_end = (src_offset + 4 * fragment_width as usize).min(src_y * 4 * width as usize + 4 * width as usize);
                                            
                                            if src_offset < src_end {
                                                let src_row = &bytes[src_offset..src_end];
                                                fragment_data.extend_from_slice(src_row);
                                                
                                                // Pad if fragment is smaller than allocation
                                                if src_row.len() < 4 * fragment_width as usize {
                                                    fragment_data.resize(fragment_data.len() + 4 * fragment_width as usize - src_row.len(), 0);
                                                }
                                            } else {
                                                // Full row padding
                                                fragment_data.resize(fragment_data.len() + 4 * fragment_width as usize, 0);
                                            }
                                        } else {
                                            // Add padding for rows beyond the source image
                                            fragment_data.resize(fragment_data.len() + 4 * fragment_width as usize, 0);
                                        }
                                        
                                        // Add padding at end of each row
                                        let padding = (align - (4 * fragment_width as usize) % align) % align;
                                        fragment_data.resize(fragment_data.len() + padding, 0);
                                    }
                                    
                                    // Queue the fragment upload
                                    self.staging.queue_upload(
                                        &fragment_data,
                                        fragment_width,
                                        fragment_height,
                                        ((4 * fragment_width as usize + (align - 1)) & !(align - 1)) as u32 - 4 * fragment_width,
                                        0,
                                        fragment_allocation.clone(),
                                    );
                                }
                                
                                true
                            } else {
                                false
                            }
                        }
                    };
                    
                    // Register this entry in the cache
                    if should_add_entry {
                        self.raster.insert_device_entry(handle, entry);
                        did_queue_upload = true;
                        
                        if let Ok(mut tracker) = crate::image::IMAGE_DISPLAY_TRACKER.lock() {
                            // Record that we've queued an upload
                            let handle_hash = format!("{:?}", handle.id());
                            tracker.record_image_upload(handle_hash, width, height);
                        }
                        
                        let queue_time = upload_start.elapsed();
                        if queue_time.as_millis() > 5 {
                            log::debug!("Queued texture upload in {:.2}ms", 
                                      queue_time.as_secs_f64() * 1000.0);
                        }
                    }
                }
            }
        }
        
        // If we have a device entry now, return it
        if self.raster.has_device_entry(handle) {
            return self.raster.get_cached_device_entry(handle);
        }
        
        // Fallback to synchronous upload if we couldn't queue an upload
        if !did_queue_upload {
            // Do synchronous upload
            let _ = self.raster.upload(device, encoder, handle, &mut self.atlas);
            
            if let Ok(mut tracker) = crate::image::IMAGE_DISPLAY_TRACKER.lock() {
                tracker.record_upload_complete();
            }
        }
        
        // Final attempt to get the entry
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
}
