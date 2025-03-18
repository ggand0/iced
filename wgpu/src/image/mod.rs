pub(crate) mod cache;
pub(crate) use cache::Cache;

mod atlas;

#[cfg(feature = "image")]
mod raster;

#[cfg(feature = "svg")]
mod vector;

use crate::core::{Rectangle, Size, Transformation};
use crate::Buffer;

use bytemuck::{Pod, Zeroable};

use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::sync::Mutex;
use once_cell::sync::Lazy;

pub use crate::graphics::Image;

pub type Batch = Vec<Image>;

#[derive(Debug)]
pub struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    backend: wgpu::Backend,
    nearest_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    texture_layout: Arc<wgpu::BindGroupLayout>,
    constant_layout: wgpu::BindGroupLayout,
    layers: Vec<Layer>,
    prepare_layer: usize,
}

impl Pipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        backend: wgpu::Backend,
    ) -> Self {
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let constant_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iced_wgpu::image constants layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<Uniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        let texture_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iced_wgpu::image texture atlas layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iced_wgpu::image pipeline layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[&constant_layout, &texture_layout],
            });

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iced_wgpu image shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    concat!(
                        include_str!("../shader/vertex.wgsl"),
                        "\n",
                        include_str!("../shader/image.wgsl"),
                    ),
                )),
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iced_wgpu::image pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array!(
                            // Position
                            0 => Float32x2,
                            // Center
                            1 => Float32x2,
                            // Scale
                            2 => Float32x2,
                            // Rotation
                            3 => Float32,
                            // Opacity
                            4 => Float32,
                            // Atlas position
                            5 => Float32x2,
                            // Atlas scale
                            6 => Float32x2,
                            // Layer
                            7 => Sint32,
                            // Snap
                            8 => Uint32,
                        ),
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        Pipeline {
            pipeline,
            backend,
            nearest_sampler,
            linear_sampler,
            texture_layout: Arc::new(texture_layout),
            constant_layout,
            layers: Vec::new(),
            prepare_layer: 0,
        }
    }

    pub fn create_cache(&self, device: &wgpu::Device) -> Cache {
        Cache::new(device, self.backend, self.texture_layout.clone())
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        cache: &mut Cache,
        images: &Batch,
        transformation: Transformation,
        scale: f32,
    ) {
        println!("iced_wgpu - Pipeline::prepare() - Processing {} images", images.len());
        let nearest_instances: &mut Vec<Instance> = &mut Vec::new();
        let linear_instances: &mut Vec<Instance> = &mut Vec::new();

        for image in images {
            match &image {
                #[cfg(feature = "image")]
                Image::Raster(image, bounds) => {
                    println!("iced_wgpu - Processing raster image with bounds: {:?}", bounds);
                    
                    if let Some(_atlas_entry) = cache.upload_raster(device, encoder, &image.handle) {
                        println!("iced_wgpu - Successfully uploaded raster image to atlas");
                    } else {
                        println!("iced_wgpu - Failed to upload raster image to atlas");
                    }
                    

                    if let Some(atlas_entry) =
                        cache.upload_raster(device, encoder, &image.handle)
                    {
                        add_instances(
                            [bounds.x, bounds.y],
                            [bounds.width, bounds.height],
                            f32::from(image.rotation),
                            image.opacity,
                            image.snap,
                            atlas_entry,
                            match image.filter_method {
                                crate::core::image::FilterMethod::Nearest => {
                                    nearest_instances
                                }
                                crate::core::image::FilterMethod::Linear => {
                                    linear_instances
                                }
                            },
                        );
                    }
                }
                #[cfg(not(feature = "image"))]
                Image::Raster { .. } => {}

                #[cfg(feature = "svg")]
                Image::Vector(svg, bounds) => {
                    let size = [bounds.width, bounds.height];

                    if let Some(atlas_entry) = cache.upload_vector(
                        device,
                        encoder,
                        &svg.handle,
                        svg.color,
                        size,
                        scale,
                    ) {
                        add_instances(
                            [bounds.x, bounds.y],
                            size,
                            f32::from(svg.rotation),
                            svg.opacity,
                            true,
                            atlas_entry,
                            nearest_instances,
                        );
                    }
                }
                #[cfg(not(feature = "svg"))]
                Image::Vector { .. } => {}
            }
        }

        if nearest_instances.is_empty() && linear_instances.is_empty() {
            return;
        }

        if self.layers.len() <= self.prepare_layer {
            self.layers.push(Layer::new(
                device,
                &self.constant_layout,
                &self.nearest_sampler,
                &self.linear_sampler,
            ));
        }

        let layer = &mut self.layers[self.prepare_layer];

        layer.prepare(
            device,
            encoder,
            belt,
            nearest_instances,
            linear_instances,
            transformation,
            scale,
        );

        self.prepare_layer += 1;
    }

    pub fn render<'a>(
        &'a self,
        cache: &'a Cache,
        layer: usize,
        bounds: Rectangle<u32>,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) -> usize {
        if let Some(layer) = self.layers.get(layer) {
            render_pass.set_pipeline(&self.pipeline);

            render_pass.set_scissor_rect(
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height,
            );

            render_pass.set_bind_group(1, cache.bind_group(), &[]);

            layer.render(render_pass)
        } else {
            0
        }
    }

    pub fn end_frame(&mut self) {
        self.prepare_layer = 0;
    }
}

#[derive(Debug)]
struct Layer {
    uniforms: wgpu::Buffer,
    nearest: Data,
    linear: Data,
}

impl Layer {
    fn new(
        device: &wgpu::Device,
        constant_layout: &wgpu::BindGroupLayout,
        nearest_sampler: &wgpu::Sampler,
        linear_sampler: &wgpu::Sampler,
    ) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced_wgpu::image uniforms buffer"),
            size: mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let nearest =
            Data::new(device, constant_layout, nearest_sampler, &uniforms);

        let linear =
            Data::new(device, constant_layout, linear_sampler, &uniforms);

        Self {
            uniforms,
            nearest,
            linear,
        }
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        nearest_instances: &[Instance],
        linear_instances: &[Instance],
        transformation: Transformation,
        scale_factor: f32,
    ) {
        let uniforms = Uniforms {
            transform: transformation.into(),
            scale_factor,
            _padding: [0.0; 3],
        };

        let bytes = bytemuck::bytes_of(&uniforms);

        belt.write_buffer(
            encoder,
            &self.uniforms,
            0,
            (bytes.len() as u64).try_into().expect("Sized uniforms"),
            device,
        )
        .copy_from_slice(bytes);

        self.nearest
            .upload(device, encoder, belt, nearest_instances);

        self.linear.upload(device, encoder, belt, linear_instances);
    }

    fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) -> usize {
        let total_instances = self.nearest.instance_count + self.linear.instance_count;
        
        println!("iced_wgpu - Layer::render() - Rendering nearest: {} instances, linear: {} instances", 
            self.nearest.instance_count, 
            self.linear.instance_count);
        
        self.nearest.render(render_pass);
        self.linear.render(render_pass);
        
        // Return the total instances so Pipeline can update metrics
        total_instances
    }
}

#[derive(Debug)]
struct Data {
    constants: wgpu::BindGroup,
    instances: Buffer<Instance>,
    instance_count: usize,
}

impl Data {
    pub fn new(
        device: &wgpu::Device,
        constant_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        uniforms: &wgpu::Buffer,
    ) -> Self {
        let constants = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iced_wgpu::image constants bind group"),
            layout: constant_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding {
                            buffer: uniforms,
                            offset: 0,
                            size: None,
                        },
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let instances = Buffer::new(
            device,
            "iced_wgpu::image instance buffer",
            Instance::INITIAL,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        Self {
            constants,
            instances,
            instance_count: 0,
        }
    }

    fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        instances: &[Instance],
    ) {
        self.instance_count = instances.len();

        if self.instance_count == 0 {
            return;
        }

        let _ = self.instances.resize(device, instances.len());
        let _ = self.instances.write(device, encoder, belt, 0, instances);
    }

    fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.instance_count == 0 {
            return;
        }

        render_pass.set_bind_group(0, &self.constants, &[]);
        render_pass.set_vertex_buffer(0, self.instances.slice(..));

        render_pass.draw(0..6, 0..self.instance_count as u32);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Instance {
    _position: [f32; 2],
    _center: [f32; 2],
    _size: [f32; 2],
    _rotation: f32,
    _opacity: f32,
    _position_in_atlas: [f32; 2],
    _size_in_atlas: [f32; 2],
    _layer: u32,
    _snap: u32,
}

impl Instance {
    pub const INITIAL: usize = 20;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Uniforms {
    transform: [f32; 16],
    scale_factor: f32,
    // Uniforms must be aligned to their largest member,
    // this uses a mat4x4<f32> which aligns to 16, so align to that
    _padding: [f32; 3],
}

fn add_instances(
    image_position: [f32; 2],
    image_size: [f32; 2],
    rotation: f32,
    opacity: f32,
    snap: bool,
    entry: &atlas::Entry,
    instances: &mut Vec<Instance>,
) {
    let center = [
        image_position[0] + image_size[0] / 2.0,
        image_position[1] + image_size[1] / 2.0,
    ];

    match entry {
        atlas::Entry::Contiguous(allocation) => {
            add_instance(
                image_position,
                center,
                image_size,
                rotation,
                opacity,
                snap,
                allocation,
                instances,
            );
        }
        atlas::Entry::Fragmented { fragments, size } => {
            let scaling_x = image_size[0] / size.width as f32;
            let scaling_y = image_size[1] / size.height as f32;

            for fragment in fragments {
                let allocation = &fragment.allocation;

                let [x, y] = image_position;
                let (fragment_x, fragment_y) = fragment.position;
                let Size {
                    width: fragment_width,
                    height: fragment_height,
                } = allocation.size();

                let position = [
                    x + fragment_x as f32 * scaling_x,
                    y + fragment_y as f32 * scaling_y,
                ];

                let size = [
                    fragment_width as f32 * scaling_x,
                    fragment_height as f32 * scaling_y,
                ];

                add_instance(
                    position, center, size, rotation, opacity, snap,
                    allocation, instances,
                );
            }
        }
    }
}

#[inline]
fn add_instance(
    position: [f32; 2],
    center: [f32; 2],
    size: [f32; 2],
    rotation: f32,
    opacity: f32,
    snap: bool,
    allocation: &atlas::Allocation,
    instances: &mut Vec<Instance>,
) {
    let (x, y) = allocation.position();
    let Size { width, height } = allocation.size();
    let layer = allocation.layer();

    let instance = Instance {
        _position: position,
        _center: center,
        _size: size,
        _rotation: rotation,
        _opacity: opacity,
        _position_in_atlas: [
            (x as f32 + 0.5) / atlas::SIZE as f32,
            (y as f32 + 0.5) / atlas::SIZE as f32,
        ],
        _size_in_atlas: [
            (width as f32 - 1.0) / atlas::SIZE as f32,
            (height as f32 - 1.0) / atlas::SIZE as f32,
        ],
        _layer: layer as u32,
        _snap: snap as u32,
    };

    instances.push(instance);
}

// Make frame counter public
pub static FRAME_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Store information about unique images displayed
pub static IMAGE_DISPLAY_TRACKER: Lazy<Mutex<ImageDisplayTracker>> = 
    Lazy::new(|| Mutex::new(ImageDisplayTracker::new()));

/// Increment the frame counter and return the new value
pub fn next_frame_id() -> usize {
    FRAME_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Get the current frame counter value
pub fn current_frame_id() -> usize {
    FRAME_COUNTER.load(Ordering::SeqCst)
}

/// Tracks when unique images are displayed to calculate true image rendering FPS
pub struct ImageDisplayTracker {
    // Set of currently displayed image identifiers
    current_frame_images: HashSet<String>,
    
    // Previous frame's image identifiers
    previous_frame_images: HashSet<String>,
    
    // Timestamps when display content changed
    change_timestamps: VecDeque<Instant>,
    
    // Window duration for FPS calculation
    window_duration: Duration,
    
    // Upload timestamps for FPS calculation
    upload_timestamps: VecDeque<Instant>,
    
    // Recently uploaded images
    uploaded_images: HashSet<String>,
    
    // Calculated FPS value
    fps: f64,
    
    // Debug counter
    frame_counter: usize,
}

impl ImageDisplayTracker {
    fn new() -> Self {
        Self {
            current_frame_images: HashSet::new(),
            previous_frame_images: HashSet::new(),
            change_timestamps: VecDeque::with_capacity(120),
            window_duration: Duration::from_secs(5),
            upload_timestamps: VecDeque::with_capacity(120),
            uploaded_images: HashSet::new(),
            fps: 0.0,
            frame_counter: 0,
        }
    }

    /// Register a new frame of image display
    pub fn start_frame(&mut self) {
        self.current_frame_images.clear();
        self.frame_counter += 1;
    }
    
    /// Register an image being displayed in the current frame
    pub fn register_image(&mut self, image_id: String, width: u32, height: u32) {
        // Create a meaningful ID including dimensions
        let full_id = format!("{}@{}x{}", image_id, width, height);
        let _ = self.current_frame_images.insert(full_id);
    }
    
    /// Record an image upload for FPS tracking
    pub fn record_image_upload(&mut self, handle_hash: String, width: u32, height: u32) {
        // Create meaningful identifier with dimensions
        let identifier = format!("{}@{}x{}", handle_hash, width, height);
        
        // Add to set of recently uploaded images - explicitly discard the result
        let _ = self.uploaded_images.insert(identifier);
        
        // Record timestamp
        self.upload_timestamps.push_back(Instant::now());
        
        // Calculate FPS
        self.calculate_fps();
        
        //println!("Image upload detected! Current upload FPS: {:.2}", self.fps);
    }
    
    /// Calculate FPS from upload timestamps
    fn calculate_fps(&mut self) {
        // Prune old timestamps
        let cutoff = Instant::now() - self.window_duration;
        while !self.upload_timestamps.is_empty() && 
              self.upload_timestamps.front().unwrap() < &cutoff {
            let _ = self.upload_timestamps.pop_front();
        }
        
        if self.upload_timestamps.len() > 1 {
            let oldest = self.upload_timestamps.front().unwrap();
            let newest = self.upload_timestamps.back().unwrap();
            let time_span = newest.duration_since(*oldest).as_secs_f64();
            
            if time_span > 0.0 {
                self.fps = (self.upload_timestamps.len() - 1) as f64 / time_span;
            }
        } else {
            self.fps = 0.0;
        }
    }
    
    /// End the frame and detect if content changed
    pub fn end_frame(&mut self) -> bool {
        // Compare with previous frame to detect change
        let changed = self.current_frame_images != self.previous_frame_images;
        
        if changed {
            //println!("DISPLAY CHANGED: Frame #{} - {} new images", 
            //         self.frame_counter, self.current_frame_images.len());
                     
            // Store timestamp of the change
            self.change_timestamps.push_back(Instant::now());
            
            // Prune old timestamps
            let cutoff = Instant::now() - self.window_duration;
            while !self.change_timestamps.is_empty() && 
                  self.change_timestamps.front().unwrap() < &cutoff {
                let _ = self.change_timestamps.pop_front();
            }
        }
        
        // Save current state for next comparison
        std::mem::swap(&mut self.current_frame_images, &mut self.previous_frame_images);
        
        changed
    }
    
    /// Get image rendering FPS based on content changes
    pub fn get_fps(&self) -> f64 {
        self.fps
    }
}

// Public API
pub fn start_display_frame() {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        tracker.start_frame();
    }
}

pub fn register_displayed_image(image_id: String, width: u32, height: u32) {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        tracker.register_image(image_id, width, height);
    }
}

pub fn end_display_frame() -> bool {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        return tracker.end_frame();
    }
    false
}

pub fn get_image_display_fps() -> f64 {
    if let Ok(tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        return tracker.get_fps();
    }
    0.0
}

// Function to record image uploads
pub fn record_image_upload(handle_hash: String, width: u32, height: u32) {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        tracker.record_image_upload(handle_hash, width, height);
    }
}
