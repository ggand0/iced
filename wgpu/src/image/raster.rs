use crate::core::image;
use crate::core::Size;
use crate::graphics;
use crate::graphics::image::image_rs;
use crate::image::atlas::{self, Atlas};
use image_rs::{ImageBuffer, Rgba};
use crate::core::image::Bytes;

use rustc_hash::{FxHashMap, FxHashSet};

/// Entry in cache corresponding to an image handle
#[derive(Debug)]
pub enum Memory {
    /// Image data on host
    Host(image_rs::ImageBuffer<Rgba<u8>, Bytes>),
    /// Storage entry
    Device(atlas::Entry),
    /// Image not found
    NotFound,
    /// Invalid image data
    Invalid,
}

impl Memory {
    /// Width and height of image
    pub fn dimensions(&self) -> Size<u32> {
        match self {
            Memory::Host(image) => {
                let (width, height) = image.dimensions();

                Size::new(width, height)
            }
            Memory::Device(entry) => entry.size(),
            Memory::NotFound => Size::new(1, 1),
            Memory::Invalid => Size::new(1, 1),
        }
    }
}

/// Caches image raster data
#[derive(Debug, Default)]
pub struct Cache {
    entries: FxHashMap<image::Id, Memory>,
    hits: FxHashSet<image::Id>,
    should_trim: bool,
}

impl Cache {
    /// Load image
    pub fn load(&mut self, handle: &image::Handle) -> &mut Memory {
        if self.contains(handle) {
            return self.get(handle).unwrap();
        }

        let memory = match graphics::image::load(handle) {
            Ok(image) => Memory::Host(image),
            Err(image_rs::error::ImageError::IoError(_)) => Memory::NotFound,
            Err(_) => Memory::Invalid,
        };

        self.should_trim = true;

        self.insert(handle, memory);
        self.get(handle).unwrap()
    }

    /// Load image and upload raster data
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &image::Handle,
        atlas: &mut Atlas,
    ) -> Option<&atlas::Entry> {
        let memory = self.load(handle);

        if let Memory::Host(image) = memory {
            let (width, height) = image.dimensions();

            // [ViewSkater] Track image uploads for image rendering FPS calculation
            let handle_hash = format!("{:?}", handle.id());            
            crate::image::record_image_upload(handle_hash, width, height);

            let entry = atlas.upload(device, encoder, width, height, image)?;

            *memory = Memory::Device(entry);
        }

        if let Memory::Device(allocation) = memory {
            Some(allocation)
        } else {
            None
        }
    }

    /// Trim cache misses from cache
    pub fn trim(&mut self, atlas: &mut Atlas) {
        // Only trim if new entries have landed in the `Cache`
        if !self.should_trim {
            return;
        }

        let hits = &self.hits;

        self.entries.retain(|k, memory| {
            let retain = hits.contains(k);

            if !retain {
                if let Memory::Device(entry) = memory {
                    atlas.remove(entry);
                }
            }

            retain
        });

        self.hits.clear();
        self.should_trim = false;
    }

    fn get(&mut self, handle: &image::Handle) -> Option<&mut Memory> {
        let _ = self.hits.insert(handle.id());

        self.entries.get_mut(&handle.id())
    }

    fn insert(&mut self, handle: &image::Handle, memory: Memory) {
        let _ = self.entries.insert(handle.id(), memory);
    }

    fn contains(&self, handle: &image::Handle) -> bool {
        self.entries.contains_key(&handle.id())
    }

    pub fn print_stats(&self) {
        println!(
            "Image cache stats: {} entries, {} hits", 
            self.entries.len(),
            self.hits.len()
        );
        
        // Count by memory type
        let host_count = self.entries.values()
            .filter(|mem| matches!(mem, Memory::Host(_)))
            .count();
        let device_count = self.entries.values()
            .filter(|mem| matches!(mem, Memory::Device(_)))
            .count();
            
        println!(
            "Memory locations: {} on host, {} on device",
            host_count, device_count
        );
    }

    // Add methods to access entries by state
    
    // Get a device entry if it exists
    pub fn get_cached_device_entry(&self, handle: &image::Handle) -> Option<&atlas::Entry> {
        if let Some(Memory::Device(entry)) = self.entries.get(&handle.id()) {
            Some(entry)
        } else {
            None
        }
    }
    
    // Get host memory if it exists
    pub fn get_cached_host_memory(&self, handle: &image::Handle) -> Option<&image_rs::ImageBuffer<Rgba<u8>, Bytes>> {
        if let Some(Memory::Host(data)) = self.entries.get(&handle.id()) {
            Some(data)
        } else {
            None
        }
    }
    
    // Insert a device entry directly
    pub fn insert_device_entry(&mut self, handle: &image::Handle, entry: atlas::Entry) {
        let _ = self.entries.insert(handle.id(), Memory::Device(entry));
    }

    // Check if entry exists in cache (any kind)
    pub fn has_cache_entry(&self, handle: &image::Handle) -> bool {
        self.entries.contains_key(&handle.id())
    }
    
    // Check if entry is already on device
    pub fn has_device_entry(&self, handle: &image::Handle) -> bool {
        matches!(self.entries.get(&handle.id()), Some(Memory::Device(_)))
    }
    
    // Check if we have host memory
    pub fn has_host_memory(&self, handle: &image::Handle) -> bool {
        matches!(self.entries.get(&handle.id()), Some(Memory::Host(_)))
    }
    
    // Get dimensions of an image if available
    pub fn get_image_dimensions(&self, handle: &image::Handle) -> Option<Size<u32>> {
        if let Some(Memory::Host(data)) = self.entries.get(&handle.id()) {
            Some(Size::new(data.width(), data.height()))
        } else {
            None
        }
    }
    
    // Get image bytes if available
    pub fn get_image_bytes(&self, handle: &image::Handle) -> Option<&[u8]> {
        if let Some(Memory::Host(data)) = self.entries.get(&handle.id()) {
            Some(data.as_raw())
        } else {
            None
        }
    }
}
