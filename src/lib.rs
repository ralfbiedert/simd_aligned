//! # In One Sentence
//!
//! You want to use [std::simd](https://github.com/rust-lang-nursery/packed_simd/) but realized there is no simple, safe and fast way to align your `f32x8` (and friends) in memory _and_ treat them as regular `f32` slices for easy loading and manipulation; `simd_aligned` to the rescue.
//!
//!
//!
//! # Highlights
//!
//! * built on top of [std::simd](https://github.com/rust-lang-nursery/packed_simd/) for easy data handling
//! * supports everything from `u8x2` to `f64x8`
//! * think in flat slices (`&[f32]`), but get performance of properly aligned SIMD vectors (`&[f32x16]`)
//! * defines `u8s`, ..., `f36s` as "best guess" for current platform (WIP)
//! * provides N-dimensional [SimdVector] and NxM-dimensional [SimdMatrix].
//!
//!
//! **Note**: Right now this is an experimental crate. Features might be added or removed depending on how [std::simd](https://github.com/rust-lang-nursery/packed_simd/) evolves. At the end of the day it's just about being able to load and manipulate data without much fuzz.
//!
//!
//! # Examples
//!
//! Produces a vector that can hold `10` elements of type `f64`. Might internally
//! allocate `5` elements of type `f64x2`, or `3` of type `f64x4`, depending on the platform.
//! All elements are guaranteed to be properly aligned for fast access.
//!
//! ```rust
//! use packed_simd::*;
//! use simd_aligned::*;
//!
//! // Create vectors of `10` f64 elements with value `0.0`.
//! let mut v1 = SimdVector::<f64s>::with(0.0, 10);
//! let mut v2 = SimdVector::<f64s>::with(0.0, 10);
//!
//! // Get "flat", mutable view of the vector, and set individual elements:
//! let v1_m = v1.flat_mut();
//! let v2_m = v2.flat_mut();
//!
//! // Set some elements on v1
//! v1_m[0] = 0.0;
//! v1_m[4] = 4.0;
//! v1_m[8] = 8.0;
//!
//! // Set some others on v2
//! v2_m[1] = 0.0;
//! v2_m[5] = 5.0;
//! v2_m[9] = 9.0;
//!
//! let mut sum = f64s::splat(0.0);
//!
//! // Eventually, do something with the actual SIMD types. Does
//! // `std::simd` vector math, e.g., f64x8 + f64x8 in one operation:
//! sum = v1[0] + v2[0];
//! ```
//!
//! # Benchmarks
//!
//! There is no performance penalty for using `simd_aligned`, while retaining all the
//! simplicity of handling flat arrays.
//!
//! ```ignore
//! test vectors::packed       ... bench:          77 ns/iter (+/- 4)
//! test vectors::scalar       ... bench:       1,177 ns/iter (+/- 464)
//! test vectors::simd_aligned ... bench:          71 ns/iter (+/- 5)
//! ```

#![feature(try_from, stdsimd, rust_2018_preview)]
#![warn(rust_2018_idioms)]

mod container;
mod conversion;
mod matrix;
mod rows;
mod sealed;
mod vector;

use packed_simd::*;

pub use crate::conversion::{packed_as_flat, packed_as_flat_mut};
pub use crate::matrix::{ColumnOptimized, OptimizationStrategy, RowOptimized, SimdMatrix};
pub use crate::vector::SimdVector;

use crate::sealed::*;

macro_rules! impl_simd {
    ($simd:ty, $element:ty, $lanes:expr, $lanestype:ty) => {
        impl crate::sealed::Simd for $simd {
            type Element = $element;
            const LANES: usize = $lanes;
            type LanesType = $lanestype;

            fn splat(t: Self::Element) -> Self {
                Self::splat(t)
            }
        }
    };
}

impl_simd!(f32x16, f32, 16, [f32; 16]);
impl_simd!(f32x8, f32, 8, [f32; 8]);
impl_simd!(f32x4, f32, 4, [f32; 4]);

impl_simd!(f64x8, f64, 8, [f64; 8]);
impl_simd!(f64x4, f64, 4, [f64; 4]);
impl_simd!(f64x2, f64, 2, [f64; 2]);

impl_simd!(u8x16, u8, 16, [u8; 16]);

// TODO: Implement heuristics for architecture / target features.

/// The widest `f32x__` type natively supported on the current platform.
pub type f32s = f32x16;

/// The widest `f64x__` type natively supported on the current platform.
pub type f64s = f64x8;

/// The widest `u8x__` type natively supported on the current platform.
pub type u8s = u8x16;
