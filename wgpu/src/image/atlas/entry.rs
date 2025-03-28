use crate::core::Size;
use crate::image::atlas;

#[derive(Debug)]
pub enum Entry {
    Contiguous(atlas::Allocation),
    Fragmented {
        size: Size<u32>,
        fragments: Vec<Fragment>,
    },
}

impl Entry {
    #[cfg(feature = "image")]
    pub fn size(&self, atlas_size: u32) -> Size<u32> {
        match self {
            Entry::Contiguous(allocation) => allocation.size(atlas_size),
            Entry::Fragmented { size, .. } => *size,
        }
    }
}

#[derive(Debug)]
pub struct Fragment {
    pub position: (u32, u32),
    pub allocation: atlas::Allocation,
}
