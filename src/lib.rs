#![feature(try_from, stdsimd, rust_2018_preview)]
#![warn(rust_2018_idioms)]

mod container;
mod conversion;
mod matrix;
mod rows;
mod vector;

pub use packed_simd::*;

pub use crate::conversion::{slice_as_flat, slice_as_flat_mut};
pub use crate::matrix::{ColumnOptimized, OptimizationStrategy, RowOptimized, SimdMatrix};
pub use crate::vector::SimdVector;

/// This is copy-paste from `packed_simd`, where this trait is unfortunately
/// sealed right now. In the future this might come from `std::simd`.
pub trait Simd {
    /// Element type of the SIMD vector
    type Element;
    /// The number of elements in the SIMD vector.
    const LANES: usize;
    /// The type: `[u32; Self::N]`.
    type LanesType;
}

macro_rules! impl_simd {
    ($simd:ty, $element:ty, $lanes:expr, $lanestype:ty) => {
        impl Simd for $simd {
            type Element = $element;
            const LANES: usize = $lanes;
            type LanesType = $lanestype;
        }
    };
}

impl_simd!(f32x16, f32, 16, [f32; 16]);
impl_simd!(f32x8, f32, 8, [f32; 8]);
impl_simd!(f32x4, f32, 4, [f32; 4]);
impl_simd!(f64x8, f64, 8, [f64; 8]);
impl_simd!(f64x4, f64, 4, [f64; 4]);
impl_simd!(f64x2, f64, 2, [f64; 2]);

// TODO: Implement heuristics for architecture / target features.

/// The "best know" `f32` type for this platform.
pub type f32s = f32x16;

/// The "best know" `f64` type for this platform.
pub type f64s = f64x8;
