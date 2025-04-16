use crate::buffer;
use crate::graphics::Antialiasing;
use crate::primitive;
use crate::quad;
use crate::text;
use crate::triangle;

#[derive(Debug, Clone)]
pub struct ImageConfig {
    pub atlas_size: u32,
    pub compression_strategy: CompressionStrategy,
}

#[cfg(feature = "image")]
impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            atlas_size: crate::image::atlas::SIZE,
            compression_strategy: CompressionStrategy::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// No compression, use RGBA8 formats (default)
    None,
    /// Use BC1 compression for textures
    Bc1,
}

#[allow(dead_code)]
#[allow(missing_debug_implementations)]
pub struct Engine {
    pub(crate) staging_belt: wgpu::util::StagingBelt,
    pub(crate) format: wgpu::TextureFormat,

    pub(crate) quad_pipeline: quad::Pipeline,
    pub(crate) text_pipeline: text::Pipeline,
    pub(crate) triangle_pipeline: triangle::Pipeline,
    #[cfg(any(feature = "image", feature = "svg"))]
    pub(crate) image_pipeline: crate::image::Pipeline,
    #[cfg(any(feature = "image", feature = "svg"))]
    pub(crate) image_config: ImageConfig,
    pub(crate) primitive_storage: primitive::Storage,
}

impl Engine {
    #[allow(unused_variables)]
    pub fn new(
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        antialiasing: Option<Antialiasing>, // TODO: Initialize AA pipelines lazily
        image_config: Option<ImageConfig>,
    ) -> Self {
        let text_pipeline = text::Pipeline::new(device, queue, format);
        let quad_pipeline = quad::Pipeline::new(device, format);
        let triangle_pipeline =
            triangle::Pipeline::new(device, format, antialiasing);

        #[cfg(any(feature = "image", feature = "svg"))]
        let image_pipeline = {
            let backend = _adapter.get_info().backend;

            crate::image::Pipeline::new(device, format, backend, image_config.as_ref())
        };

        Self {
            // TODO: Resize belt smartly (?)
            // It would be great if the `StagingBelt` API exposed methods
            // for introspection to detect when a resize may be worth it.
            staging_belt: wgpu::util::StagingBelt::new(
                buffer::MAX_WRITE_SIZE as u64,
            ),
            format,

            quad_pipeline,
            text_pipeline,
            triangle_pipeline,

            #[cfg(any(feature = "image", feature = "svg"))]
            image_pipeline,

            #[cfg(any(feature = "image", feature = "svg"))]
            image_config: image_config.unwrap_or_default(),

            primitive_storage: primitive::Storage::default(),
        }
    }

    #[cfg(any(feature = "image", feature = "svg"))]
    pub fn create_image_cache(
        &self,
        device: &wgpu::Device,
    ) -> crate::image::Cache {
        self.image_pipeline.create_cache(device)
    }

    /// Updates the image configuration settings
    /// 
    /// This allows changing compression strategy and other image-related settings
    /// at runtime.
    #[cfg(any(feature = "image", feature = "svg"))]
    pub fn update_image_config(
        &mut self,
        image_config: ImageConfig,
        device: &wgpu::Device,
    ) -> crate::image::Cache {
        // Update the stored config
        self.image_config = image_config.clone();
        self.image_pipeline.update_image_config(image_config);
        
        // Create a new cache with the updated settings
        // This will use the new compression strategy for future uploads
        self.create_image_cache(device)
    }

    /// Clears all stored data in the [`primitive::Storage`], releasing associated GPU resources
    pub fn clear_primitive_storage(&mut self) {
        self.primitive_storage.clear();
    }

    pub fn submit(
        &mut self,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
    ) -> wgpu::SubmissionIndex {
        self.staging_belt.finish();
        let index = queue.submit(Some(encoder.finish()));
        self.staging_belt.recall();

        self.quad_pipeline.end_frame();
        self.text_pipeline.end_frame();
        self.triangle_pipeline.end_frame();

        #[cfg(any(feature = "image", feature = "svg"))]
        self.image_pipeline.end_frame();

        index
    }
}
