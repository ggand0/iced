use crate::buffer;
use crate::graphics::Antialiasing;
use crate::primitive;
use crate::quad;
use crate::text;
use crate::triangle;

#[derive(Debug, Clone)]
pub struct ImageConfig {
    pub use_parallel_processing: bool,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            use_parallel_processing: true,
        }
    }
}

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
    pub fn new(
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        antialiasing: Option<Antialiasing>,
        image_config: Option<ImageConfig>,
    ) -> Self {
        let backend = _adapter.get_info().backend;
        println!("Using GPU backend: {:?}", backend);
        
        let text_pipeline = text::Pipeline::new(device, queue, format);
        let quad_pipeline = quad::Pipeline::new(device, format);
        let triangle_pipeline =
            triangle::Pipeline::new(device, format, antialiasing);

        #[cfg(any(feature = "image", feature = "svg"))]
        let image_pipeline = {
            let backend = _adapter.get_info().backend;

            crate::image::Pipeline::new(device, format, backend)
        };

        Self {
            staging_belt: wgpu::util::StagingBelt::new(
                if cfg!(target_os = "linux") {
                    buffer::MAX_WRITE_SIZE as u64 * 4 // Larger for Linux
                } else {
                    buffer::MAX_WRITE_SIZE as u64     // Normal size for other platforms
                }
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
        let mut cache = self.image_pipeline.create_cache(device);
        cache.set_parallel_processing(self.image_config.use_parallel_processing);
        cache
    }

    pub fn submit(
        &mut self,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
    ) -> wgpu::SubmissionIndex {
        let render_start = std::time::Instant::now();
        
        self.staging_belt.finish();
        let index = queue.submit(Some(encoder.finish()));
        self.staging_belt.recall();

        self.quad_pipeline.end_frame();
        self.text_pipeline.end_frame();
        self.triangle_pipeline.end_frame();

        #[cfg(any(feature = "image", feature = "svg"))]
        {
            // Record render time
            let render_duration = render_start.elapsed();
            if render_duration.as_millis() > 30 {
                println!("SLOW GPU SUBMIT: {:.2}ms", render_duration.as_secs_f64() * 1000.0);
            }
            crate::image::record_image_render_duration(render_duration);
            self.image_pipeline.end_frame();
        }

        index
    }
}
