pub mod entry;

mod allocation;
mod allocator;
mod layer;

pub use allocation::Allocation;
pub use entry::Entry;
pub use layer::Layer;

use allocator::Allocator;

pub const SIZE: u32 = 2048;

use crate::core::Size;
use crate::graphics::color;

use std::sync::Arc;

use crate::image::compression;
use crate::engine::CompressionStrategy;

#[derive(Debug)]
pub struct Atlas {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    texture_bind_group: wgpu::BindGroup,
    texture_layout: Arc<wgpu::BindGroupLayout>,
    layers: Vec<Layer>,
    compression_strategy: CompressionStrategy,
}

impl Atlas {
    pub fn new(
        device: &wgpu::Device,
        backend: wgpu::Backend,
        texture_layout: Arc<wgpu::BindGroupLayout>,
        compression_strategy: CompressionStrategy,
    ) -> Self {
        let layers = match backend {
            // On the GL backend we start with 2 layers, to help wgpu figure
            // out that this texture is `GL_TEXTURE_2D_ARRAY` rather than `GL_TEXTURE_2D`
            // https://github.com/gfx-rs/wgpu/blob/004e3efe84a320d9331371ed31fa50baa2414911/wgpu-hal/src/gles/mod.rs#L371
            wgpu::Backend::Gl => vec![Layer::Empty, Layer::Empty],
            _ => vec![Layer::Empty],
        };

        let extent = wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: layers.len() as u32,
        };

        // Choose texture format based on compression strategy
        let format = match compression_strategy {
            CompressionStrategy::None => {
                if color::GAMMA_CORRECTION {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                }
            },
            CompressionStrategy::Bc1 => {
                // BC1 doesn't have an sRGB variant in wgpu (we'd need BC7 for that)
                wgpu::TextureFormat::Bc1RgbaUnorm
            },
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_wgpu::image texture atlas"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let texture_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_wgpu::image texture atlas bind group"),
                layout: &texture_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                }],
            });

        Atlas {
            texture,
            texture_view,
            texture_bind_group,
            texture_layout,
            layers,
            compression_strategy,
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.texture_bind_group
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Option<Entry> {
        let entry = {
            let current_size = self.layers.len();
            let entry = self.allocate(width, height)?;

            // We grow the internal texture after allocating if necessary
            let new_layers = self.layers.len() - current_size;
            self.grow(new_layers, device, encoder);

            entry
        };

        log::debug!("Allocated atlas entry: {entry:?}");

        match self.compression_strategy {
            CompressionStrategy::None => {
                // Original uncompressed upload path
                // It is a webgpu requirement that:
                //   BufferCopyView.layout.bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
                // So we calculate padded_width by rounding width up to the next
                // multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let padding = (align - (4 * width) % align) % align;
                let padded_width = (4 * width + padding) as usize;
                let padded_data_size = padded_width * height as usize;

                let mut padded_data = vec![0; padded_data_size];

                for row in 0..height as usize {
                    let offset = row * padded_width;

                    padded_data[offset..offset + 4 * width as usize].copy_from_slice(
                        &data[row * 4 * width as usize..(row + 1) * 4 * width as usize],
                    );
                }

                match &entry {
                    Entry::Contiguous(allocation) => {
                        self.upload_allocation(
                            &padded_data,
                            width,
                            height,
                            padding,
                            0,
                            allocation,
                            device,
                            encoder,
                        );
                    }
                    Entry::Fragmented { fragments, .. } => {
                        for fragment in fragments {
                            let (x, y) = fragment.position;
                            let offset = (y * padded_width as u32 + 4 * x) as usize;

                            self.upload_allocation(
                                &padded_data,
                                width,
                                height,
                                padding,
                                offset,
                                &fragment.allocation,
                                device,
                                encoder,
                            );
                        }
                    }
                }
            },
            CompressionStrategy::Bc1 => {
                // New compressed upload path
                self.upload_compressed(device, encoder, width, height, data, &entry);
            }
        }

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "Atlas layers: {} (busy: {}, allocations: {})",
                self.layer_count(),
                self.layers.iter().filter(|layer| !layer.is_empty()).count(),
                self.layers.iter().map(Layer::allocations).sum::<usize>(),
            );
        }

        Some(entry)
    }

    pub fn remove(&mut self, entry: &Entry) {
        log::debug!("Removing atlas entry: {entry:?}");

        match entry {
            Entry::Contiguous(allocation) => {
                self.deallocate(allocation);
            }
            Entry::Fragmented { fragments, .. } => {
                for fragment in fragments {
                    self.deallocate(&fragment.allocation);
                }
            }
        }
    }

    fn allocate(&mut self, width: u32, height: u32) -> Option<Entry> {
        // Allocate one layer if texture fits perfectly
        if width == SIZE && height == SIZE {
            let mut empty_layers = self
                .layers
                .iter_mut()
                .enumerate()
                .filter(|(_, layer)| layer.is_empty());

            if let Some((i, layer)) = empty_layers.next() {
                *layer = Layer::Full;

                return Some(Entry::Contiguous(Allocation::Full { layer: i }));
            }

            self.layers.push(Layer::Full);

            return Some(Entry::Contiguous(Allocation::Full {
                layer: self.layers.len() - 1,
            }));
        }

        // Split big textures across multiple layers
        if width > SIZE || height > SIZE {
            let mut fragments = Vec::new();
            let mut y = 0;

            while y < height {
                let height = std::cmp::min(height - y, SIZE);
                let mut x = 0;

                while x < width {
                    let width = std::cmp::min(width - x, SIZE);

                    let allocation = self.allocate(width, height)?;

                    if let Entry::Contiguous(allocation) = allocation {
                        fragments.push(entry::Fragment {
                            position: (x, y),
                            allocation,
                        });
                    }

                    x += width;
                }

                y += height;
            }

            return Some(Entry::Fragmented {
                size: Size::new(width, height),
                fragments,
            });
        }

        // Try allocating on an existing layer
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                Layer::Empty => {
                    let mut allocator = Allocator::new(SIZE);

                    if let Some(region) = allocator.allocate(width, height) {
                        *layer = Layer::Busy(allocator);

                        return Some(Entry::Contiguous(Allocation::Partial {
                            region,
                            layer: i,
                        }));
                    }
                }
                Layer::Busy(allocator) => {
                    if let Some(region) = allocator.allocate(width, height) {
                        return Some(Entry::Contiguous(Allocation::Partial {
                            region,
                            layer: i,
                        }));
                    }
                }
                Layer::Full => {}
            }
        }

        // Create new layer with atlas allocator
        let mut allocator = Allocator::new(SIZE);

        if let Some(region) = allocator.allocate(width, height) {
            self.layers.push(Layer::Busy(allocator));

            return Some(Entry::Contiguous(Allocation::Partial {
                region,
                layer: self.layers.len() - 1,
            }));
        }

        // We ran out of memory (?)
        None
    }

    fn deallocate(&mut self, allocation: &Allocation) {
        log::debug!("Deallocating atlas: {allocation:?}");

        match allocation {
            Allocation::Full { layer } => {
                self.layers[*layer] = Layer::Empty;
            }
            Allocation::Partial { layer, region } => {
                let layer = &mut self.layers[*layer];

                if let Layer::Busy(allocator) = layer {
                    allocator.deallocate(region);

                    if allocator.is_empty() {
                        *layer = Layer::Empty;
                    }
                }
            }
        }
    }

    fn upload_allocation(
        &mut self,
        data: &[u8],
        image_width: u32,
        image_height: u32,
        padding: u32,
        offset: usize,
        allocation: &Allocation,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        use wgpu::util::DeviceExt;

        let (x, y) = allocation.position();
        let Size { width, height } = allocation.size();
        let layer = allocation.layer();

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image upload buffer"),
                contents: data,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: offset as u64,
                    bytes_per_row: Some(4 * image_width + padding),
                    rows_per_image: Some(image_height),
                },
            },
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x,
                    y,
                    z: layer as u32,
                },
                aspect: wgpu::TextureAspect::default(),
            },
            extent,
        );
    }

    fn grow(
        &mut self,
        amount: usize,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if amount == 0 {
            return;
        }

        // Choose format based on compression strategy
        let format = match self.compression_strategy {
            CompressionStrategy::None => {
                if color::GAMMA_CORRECTION {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                }
            },
            CompressionStrategy::Bc1 => {
                wgpu::TextureFormat::Bc1RgbaUnorm
            },
        };

        let new_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_wgpu::image texture atlas"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: self.layers.len() as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let amount_to_copy = self.layers.len() - amount;

        for (i, layer) in
            self.layers.iter_mut().take(amount_to_copy).enumerate()
        {
            if layer.is_empty() {
                continue;
            }

            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::default(),
                },
                wgpu::ImageCopyTexture {
                    texture: &new_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::default(),
                },
                wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.texture = new_texture;
        self.texture_view =
            self.texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        self.texture_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_wgpu::image texture atlas bind group"),
                layout: &self.texture_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.texture_view,
                    ),
                }],
            });
    }

    // New method to handle compressed uploads
    fn upload_compressed(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        data: &[u8],
        entry: &Entry,
    ) {
        match &entry {
            Entry::Contiguous(allocation) => {
                self.upload_compressed_allocation(
                    device,
                    encoder,
                    width,
                    height, 
                    data,
                    allocation,
                );
            }
            Entry::Fragmented { fragments, .. } => {
                for fragment in fragments {
                    let (x, y) = fragment.position;
                    let fragment_width = fragment.allocation.size().width;
                    let fragment_height = fragment.allocation.size().height;
                    
                    // Extract fragment data from the original image
                    let mut fragment_data = Vec::with_capacity((fragment_width * fragment_height * 4) as usize);
                    for fy in 0..fragment_height {
                        for fx in 0..fragment_width {
                            let src_x = x + fx;
                            let src_y = y + fy;
                            if src_x < width && src_y < height {
                                let src_idx = ((src_y * width + src_x) * 4) as usize;
                                fragment_data.extend_from_slice(&data[src_idx..src_idx+4]);
                            } else {
                                // Padding for fragments that extend beyond the original image
                                fragment_data.extend_from_slice(&[0, 0, 0, 0]);
                            }
                        }
                    }
                    
                    self.upload_compressed_allocation(
                        device,
                        encoder,
                        fragment_width,
                        fragment_height,
                        &fragment_data,
                        &fragment.allocation,
                    );
                }
            }
        }
    }

    fn upload_compressed_allocation(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        data: &[u8],
        allocation: &Allocation,
    ) {
        use wgpu::util::DeviceExt;

        let (x, y) = allocation.position();
        let layer = allocation.layer();
        
        // BC1 requires coordinates to be aligned to 4-pixel blocks
        // Round down to nearest multiple of 4
        let aligned_x = (x / 4) * 4;
        let aligned_y = (y / 4) * 4;
        
        // Calculate offsets within the block
        let x_offset = x - aligned_x;
        let y_offset = y - aligned_y;
        
        // Convert to 4x4 blocks for BC1 compression
        let blocks_x = (width + 3) / 4;
        let blocks_y = (height + 3) / 4;
        
        // Create RGBA blocks from the raw data
        let mut rgba_blocks = Vec::with_capacity((blocks_x * blocks_y) as usize);
        
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut block = [[0u8; 4]; 16];
                for py in 0..4 {
                    for px in 0..4 {
                        let img_x = bx * 4 + px;
                        let img_y = by * 4 + py;
                        
                        if img_x < width && img_y < height {
                            let idx = ((img_y * width + img_x) * 4) as usize;
                            block[(py * 4 + px) as usize] = [
                                data[idx],
                                data[idx + 1],
                                data[idx + 2],
                                data[idx + 3],
                            ];
                        }
                    }
                }
                
                // Compress the block and add it to our list
                let compressed = compression::compress_bc1_block(
                    &block, 
                    compression::CompressionAlgorithm::RangeFit
                );
                rgba_blocks.push(compressed);
            }
        }
        
        // Flatten the blocks
        let compressed_data: Vec<u8> = rgba_blocks.into_iter().flat_map(|b| b.to_vec()).collect();
        
        // BC1 format is 8 bytes per 4x4 pixel block
        let bytes_per_row = blocks_x * 8;
        
        // Align to wgpu requirements
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padding = (align - (bytes_per_row % align)) % align;
        let padded_bytes_per_row = bytes_per_row + padding;
        
        // Create padded data if needed
        let upload_data = if padding == 0 {
            compressed_data
        } else {
            let mut padded_data = Vec::with_capacity((padded_bytes_per_row * blocks_y) as usize);
            for i in 0..blocks_y {
                let start = (i * bytes_per_row) as usize;
                let end = start + bytes_per_row as usize;
                padded_data.extend_from_slice(&compressed_data[start..end]);
                padded_data.extend(std::iter::repeat(0).take(padding as usize));
            }
            padded_data
        };
        
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compressed image upload buffer"),
            contents: &upload_data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        
        // Size in blocks (each block is 4x4 pixels)
        let width_blocks = (width + 3) / 4;
        let height_blocks = (height + 3) / 4;
        
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height_blocks),
                },
            },
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: aligned_x,
                    y: aligned_y,
                    z: layer as u32,
                },
                aspect: wgpu::TextureAspect::default(),
            },
            wgpu::Extent3d {
                width: width_blocks * 4,  // Convert back to pixels for the extent
                height: height_blocks * 4,
                depth_or_array_layers: 1,
            },
        );
    }
}
