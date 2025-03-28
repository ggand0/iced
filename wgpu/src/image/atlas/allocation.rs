use crate::core::Size;
use crate::image::atlas::allocator;

#[derive(Debug, Clone)]
pub enum Allocation {
    Partial {
        layer: usize,
        region: allocator::Region,
    },
    Full {
        layer: usize,
    },
}

impl Allocation {
    pub fn position(&self) -> (u32, u32) {
        match self {
            Allocation::Partial { region, .. } => region.position(),
            Allocation::Full { .. } => (0, 0),
        }
    }

    pub fn size(&self, atlas_size: u32) -> Size<u32> {
        match self {
            Allocation::Partial { region, .. } => region.size(),
            Allocation::Full { .. } => Size::new(atlas_size, atlas_size),
        }
    }

    pub fn layer(&self) -> usize {
        match self {
            Allocation::Partial { layer, .. } => *layer,
            Allocation::Full { layer } => *layer,
        }
    }
}
