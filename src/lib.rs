//! # In One Sentence
//!
//! You want to use [`std::simd`](https://github.com/rust-lang-nursery/packed_simd/) but realized there is no simple, safe and fast way to align your `f32x8` (and friends) in memory _and_ treat them as regular `f32` slices for easy loading and manipulation; `simd_aligned` to the rescue.
//!
//!
//!
//! # Highlights
//!
//! * built on top of [`std::simd`](https://github.com/rust-lang-nursery/packed_simd/) for easy data handling
//! * supports everything from `u8x2` to `f64x8`
//! * think in flat slices (`&[f32]`), but get performance of properly aligned SIMD vectors (`&[f32x16]`)
//! * defines `u8s`, ..., `f36s` as "best guess" for current platform (WIP)
//! * provides N-dimensional [`VectorD`] and NxM-dimensional [`MatrixD`].
//!
//!
//! **Note**: Right now this is an experimental crate. Features might be added or removed depending on how [`std::simd`](https://github.com/rust-lang-nursery/packed_simd/) evolves. At the end of the day it's just about being able to load and manipulate data without much fuzz.
//!
//!
//! # Examples
//!
//! Produces a vector that can hold `10` elements of type `f64`. Might internally
//! allocate `5` elements of type `f64x2`, or `3` of type `f64x4`, depending on the platform.
//! All elements are guaranteed to be properly aligned for fast access.
//!
//! ```rust
//! #![feature(portable_simd)]
//! use std::simd::*;
//! use simd_aligned::*;
//!
//! // Create vectors of `10` f64 elements with value `0.0`.
//! let mut v1 = VectorD::<f64s>::with(0.0, 10);
//! let mut v2 = VectorD::<f64s>::with(0.0, 10);
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
//!
//! # FAQ
//!
//! ### How does it relate to [faster](https://github.com/AdamNiederer/faster) and [`std::simd`](https://github.com/rust-lang-nursery/packed_simd/)?
//!
//! * `simd_aligned` builds on top of `std::simd`. At aims to provide common, SIMD-aligned
//! data structure that support simple and safe scalar access patterns.
//!
//! * `faster` (as of today) is really good if you already have exiting flat slices in your code
//! and want operate them "full SIMD ahead". However, in particular when dealing with multiple
//! slices at the same time (e.g., kernel computations) the performance impact of unaligned arrays can
//! become a bit more noticeable (e.g., in the case of [ffsvm](https://github.com/ralfbiedert/ffsvm-rust/) up to 10% - 20%).

#![feature(portable_simd)]
#![warn(clippy::all)] // Enable ALL the warnings ...
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::module_inception)]

mod conversion;
mod matrix;
mod packed;
mod vector;

pub mod arch;
pub mod traits;


pub use crate::{
    arch::current::*,
    conversion::{packed_as_flat, packed_as_flat_mut},
    matrix::{AccessStrategy, Columns, MatrixD, MatrixFlat, MatrixFlatMut, Rows},
    vector::VectorD,
};

pub use std::simd::*;


pub trait SimdExt {
    type T;

    fn sum(&self) -> Self::T;
}

macro_rules! impl_simd {
    ($simd:ty, $element:ty, $lanes:expr, $lanestype:ty) => {
        impl crate::traits::Simd for $simd {
            type Element = $element;
            type LanesType = $lanestype;

            const LANES: usize = $lanes;

            fn splat(t: Self::Element) -> Self { Self::splat(t) }
        }

        impl SimdExt for $simd {
            type T = $element;

            fn sum(&self) -> Self::T {
                self.as_array().iter().sum()
            }
        }
    };
}

impl_simd!(u8x4, u8, 4, [u8; 4]);
impl_simd!(u8x8, u8, 8, [u8; 8]);
impl_simd!(u8x16, u8, 16, [u8; 16]);
impl_simd!(u8x32, u8, 32, [u8; 32]);

impl_simd!(i8x4, i8, 4, [i8; 4]);
impl_simd!(i8x8, i8, 8, [i8; 8]);
impl_simd!(i8x16, i8, 16, [i8; 16]);
impl_simd!(i8x32, i8, 32, [i8; 32]);

impl_simd!(u16x2, u16, 2, [u16; 2]);
impl_simd!(u16x4, u16, 4, [u16; 4]);
impl_simd!(u16x8, u16, 8, [u16; 8]);
impl_simd!(u16x16, u16, 16, [u16; 16]);

impl_simd!(i16x2, i16, 2, [i16; 2]);
impl_simd!(i16x4, i16, 4, [i16; 4]);
impl_simd!(i16x8, i16, 8, [i16; 8]);
impl_simd!(i16x16, i16, 16, [i16; 16]);

impl_simd!(u32x2, u32, 2, [u32; 2]);
impl_simd!(u32x4, u32, 4, [u32; 4]);
impl_simd!(u32x8, u32, 8, [u32; 8]);

impl_simd!(i32x2, i32, 2, [i32; 2]);
impl_simd!(i32x4, i32, 4, [i32; 4]);
impl_simd!(i32x8, i32, 8, [i32; 8]);

impl_simd!(u64x2, u64, 2, [u64; 2]);
impl_simd!(u64x4, u64, 4, [u64; 4]);

impl_simd!(i64x2, i64, 2, [i64; 2]);
impl_simd!(i64x4, i64, 4, [i64; 4]);

impl_simd!(f32x2, f32, 2, [f32; 2]);
impl_simd!(f32x4, f32, 4, [f32; 4]);
impl_simd!(f32x8, f32, 8, [f32; 8]);
impl_simd!(f32x16, f32, 16, [f32; 16]);

impl_simd!(f64x2, f64, 2, [f64; 2]);
impl_simd!(f64x4, f64, 4, [f64; 4]);
impl_simd!(f64x8, f64, 8, [f64; 8]);
