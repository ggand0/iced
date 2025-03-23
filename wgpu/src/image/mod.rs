pub(crate) mod cache;
pub(crate) use cache::Cache;

mod atlas;
mod staging;

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
        let nearest_instances: &mut Vec<Instance> = &mut Vec::new();
        let linear_instances: &mut Vec<Instance> = &mut Vec::new();

        for image in images {
            match &image {
                #[cfg(feature = "image")]
                Image::Raster(image, bounds) => {
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
    ) {
        let render_start = Instant::now();
        
        if let Some(layer) = self.layers.get(layer) {
            render_pass.set_pipeline(&self.pipeline);

            render_pass.set_scissor_rect(
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height,
            );

            render_pass.set_bind_group(1, cache.bind_group(), &[]);

            layer.render(render_pass);
        }
        
        // Record render duration
        if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
            tracker.record_render_duration(render_start.elapsed());
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

    fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.nearest.render(render_pass);
        self.linear.render(render_pass);
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


// Store information about unique images displayed
pub static IMAGE_DISPLAY_TRACKER: Lazy<Mutex<ImageDisplayTracker>> = 
    Lazy::new(|| Mutex::new(ImageDisplayTracker::new()));

/// Tracks when unique images are displayed to calculate true image rendering FPS
#[derive(Debug)]
pub struct ImageDisplayTracker {    
    // Window duration for FPS calculation
    window_duration: Duration,
    
    // Upload timestamps for FPS calculation
    upload_timestamps: VecDeque<Instant>,
    
    // Recently uploaded images
    uploaded_images: HashSet<String>,
    
    // Calculated FPS value
    fps: f64,
    
    // Add new timing fields and stats
    upload_durations: VecDeque<Duration>,
    render_durations: VecDeque<Duration>,
    current_upload_start: Option<Instant>,
    current_render_start: Option<Instant>,
    max_render_duration: Duration,
    min_render_duration: Duration,
    pub total_frames_rendered: usize,
}

impl ImageDisplayTracker {
    fn new() -> Self {
        Self {
            window_duration: Duration::from_secs(2),
            upload_timestamps: VecDeque::with_capacity(120),
            uploaded_images: HashSet::new(),
            fps: 0.0,
            upload_durations: VecDeque::with_capacity(100),
            render_durations: VecDeque::with_capacity(100),
            current_upload_start: None,
            current_render_start: None,
            max_render_duration: Duration::from_millis(0),
            min_render_duration: Duration::from_secs(1000),
            total_frames_rendered: 0,
        }
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
        
        // Start timing the upload process
        self.current_upload_start = Some(Instant::now());
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

    /// Get image rendering FPS based on content changes
    #[allow(dead_code)]
    pub fn get_fps(&self) -> f64 {
        self.fps
    }


    /// Get a copy of the recent upload timestamps for syncing with application
    pub fn get_timestamps(&self) -> VecDeque<Instant> {
        self.upload_timestamps.clone()
    }
    
    /// Get the size of tracked timestamps
    pub fn timestamps_count(&self) -> usize {
        self.upload_timestamps.len()
    }
    
    /// Allow the application to initialize this tracker with external timestamps
    pub fn sync_from_external(&mut self, timestamps: VecDeque<Instant>) {
        // Only sync if we're getting meaningful data
        if !timestamps.is_empty() {
            self.upload_timestamps = timestamps;
            self.calculate_fps();
        }
    }

    // Fix the upload_durations trimming
    pub fn record_upload_complete(&mut self) {
        if let Some(start) = self.current_upload_start.take() {
            let duration = start.elapsed();
            self.upload_durations.push_back(duration);
            
            // Trim old entries - use let _ = to explicitly discard the result
            while self.upload_durations.len() > 100 {
                let _ = self.upload_durations.pop_front();
            }
        }
    }

    // Update record_render_duration to log outliers
    pub fn record_render_duration(&mut self, duration: Duration) {
        self.render_durations.push_back(duration);
        
        self.total_frames_rendered += 1;
        
        // Track min/max for outlier detection
        if duration > self.max_render_duration {
            self.max_render_duration = duration;
            println!("SLOW FRAME DETECTED: {:.2}ms", duration.as_secs_f64() * 1000.0);
        }
        
        if duration < self.min_render_duration {
            self.min_render_duration = duration;
        }
        
        // Log every 50th frame for monitoring
        if self.total_frames_rendered % 50 == 0 {
            let (avg_upload, avg_render) = self.get_timing_stats();
            println!("RENDER STATS: Frames: {}, FPS: {:.2}, Avg Render: {:.2}ms, Min: {:.2}ms, Max: {:.2}ms", 
                    self.total_frames_rendered, 
                    self.fps,
                    avg_render * 1000.0,
                    self.min_render_duration.as_secs_f64() * 1000.0,
                    self.max_render_duration.as_secs_f64() * 1000.0);
        }
        
        while self.render_durations.len() > 100 {
            let _ = self.render_durations.pop_front();
        }
    }

    // Start timing a render operation
    pub fn start_render_timing(&mut self) {
        self.current_render_start = Some(Instant::now());
    }
    
    // Complete timing a render operation
    pub fn complete_render_timing(&mut self) {
        if let Some(start) = self.current_render_start.take() {
            let duration = start.elapsed();
            self.record_render_duration(duration);
        }
    }
    
    // Enhanced method to get timing stats with more details
    pub fn get_detailed_timing_stats(&self) -> (f64, f64, f64, f64, f64) {
        let (avg_upload, avg_render) = self.get_timing_stats();
        
        let max_render = self.max_render_duration.as_secs_f64();
        let min_render = if self.total_frames_rendered > 0 {
            self.min_render_duration.as_secs_f64()
        } else {
            0.0
        };
        
        (self.fps, avg_upload, avg_render, min_render, max_render)
    }

    // Add method to get average timings
    pub fn get_timing_stats(&self) -> (f64, f64) {
        let avg_upload = if self.upload_durations.is_empty() {
            0.0
        } else {
            self.upload_durations.iter().sum::<Duration>().as_secs_f64() 
                / self.upload_durations.len() as f64
        };
        
        let avg_render = if self.render_durations.is_empty() {
            0.0
        } else {
            self.render_durations.iter().sum::<Duration>().as_secs_f64()
                / self.render_durations.len() as f64
        };
        
        (avg_upload, avg_render)
    }

    // Add method to start upload timing
    pub fn start_upload_timing(&mut self) {
        self.current_upload_start = Some(Instant::now());
    }
    
    // Add method to handle batch uploads
    pub fn record_batch_upload_complete(&mut self, count: usize) {
        if let Some(start) = self.current_upload_start.take() {
            let duration = start.elapsed();
            
            // Record the average duration per texture
            if count > 0 {
                let avg_duration = duration.div_f32(count as f32);
                self.upload_durations.push_back(avg_duration);
                
                while self.upload_durations.len() > 100 {
                    let _ = self.upload_durations.pop_front();
                }
                
                if avg_duration.as_millis() > 20 {
                    log::warn!("SLOW BATCH UPLOAD: {:.2}ms avg for {} textures", 
                             avg_duration.as_secs_f64() * 1000.0, count);
                }
            }
        }
    }
}


// Function to record image uploads
pub fn record_image_upload(handle_hash: String, width: u32, height: u32) {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        tracker.record_image_upload(handle_hash, width, height);
    }
}

// Add this new function to record rendering time measurements
pub fn record_image_render_duration(duration: Duration) {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        tracker.record_render_duration(duration);
    }
}

/// Get the current image rendering FPS 
/// This is a global function accessible to applications
pub fn get_image_display_fps() -> f64 {
    if let Ok(tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        return tracker.get_fps();
    }
    0.0
}

/// Get the internal timestamps from the image tracker
/// This allows applications to sync their own FPS calculations
pub fn get_image_upload_timestamps() -> VecDeque<Instant> {
    if let Ok(tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        return tracker.get_timestamps();
    }
    VecDeque::new()
}

/// Sync the tracker with external timestamps (for bidirectional sync)
pub fn sync_image_tracker_timestamps(timestamps: VecDeque<Instant>) {
    if let Ok(mut tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        tracker.sync_from_external(timestamps);
    }
}

// Add new function to get detailed performance metrics with logging
pub fn get_image_rendering_stats_with_logging() -> (f64, f64, f64) {
    if let Ok(tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        let fps = tracker.get_fps();
        let (avg_upload, avg_render) = tracker.get_timing_stats();
        
        println!("IMAGE PERFORMANCE: FPS: {:.2}, Upload: {:.2}ms, Render: {:.2}ms", 
                 fps, avg_upload * 1000.0, avg_render * 1000.0);
        
        // Log additional stats about recent frames
        if let Some(last_render) = tracker.render_durations.back() {
            println!("LAST FRAME: Render time: {:.2}ms", last_render.as_secs_f64() * 1000.0);
        }
        
        return (fps, avg_upload, avg_render);
    }
    (0.0, 0.0, 0.0)
}

// Fix the debug_image_upload_status function to use existing fields

pub fn debug_image_upload_status() {
    if let Ok(tracker) = IMAGE_DISPLAY_TRACKER.lock() {
        // Use the fields that actually exist in ImageDisplayTracker
        let uploads_pending = tracker.upload_timestamps.len();
        let total_frames = tracker.total_frames_rendered;
        
        println!("IMAGE UPLOAD STATUS:");
        println!("  Recent uploads: {}", uploads_pending);
        println!("  Total frames rendered: {}", total_frames);
        println!("  Current FPS: {:.2}", tracker.fps);
        
        // Get timing statistics
        let (avg_upload, avg_render) = tracker.get_timing_stats();
        println!("  Average upload time: {:.2}ms", avg_upload * 1000.0);
        println!("  Average render time: {:.2}ms", avg_render * 1000.0);
        println!("  Min render time: {:.2}ms", tracker.min_render_duration.as_secs_f64() * 1000.0);
        println!("  Max render time: {:.2}ms", tracker.max_render_duration.as_secs_f64() * 1000.0);
        
        // Show recent uploads
        if !tracker.uploaded_images.is_empty() {
            println!("RECENTLY UPLOADED IMAGES:");
            for image_identifier in &tracker.uploaded_images {
                println!("  {}", image_identifier);
            }
        }
    }
}
