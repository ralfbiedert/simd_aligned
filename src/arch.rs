//! Contains vector definitions with a fixed bit width, reexported from [wide](https://crates.io/crates/wide)
#![allow(non_camel_case_types)]

pub use wide::{f32x4, f32x8, f64x2, f64x4, i16x16, i16x8, i32x4, i32x8, i64x2, i64x4, i8x16, i8x32, u16x16, u16x8, u32x4, u32x8, u64x2, u64x4, u8x16};

macro_rules! impl_simd {
    ($simd:ty, $element:ty, $lanes:expr, $lanestype:ty) => {
        impl crate::traits::Simd for $simd {
            type Element = $element;
            type LanesType = $lanestype;

            const LANES: usize = $lanes;

            fn splat(t: Self::Element) -> Self {
                Self::splat(t)
            }

            #[allow(clippy::transmute_ptr_to_ptr)]
            #[allow(clippy::missing_transmute_annotations)]
            fn as_array(&self) -> &[Self::Element] {
                let self_array = unsafe { std::mem::transmute::<_, &$lanestype>(self) };
                self_array.as_ref()
            }

            fn sum(&self) -> Self::Element {
                self.as_array().iter().sum()
            }
        }
    };
}

impl_simd!(u8x16, u8, 16, [u8; 16]);

impl_simd!(i8x16, i8, 16, [i8; 16]);
impl_simd!(i8x32, i8, 32, [i8; 32]);

impl_simd!(u16x8, u16, 8, [u16; 8]);
impl_simd!(u16x16, u16, 16, [u16; 16]);

impl_simd!(i16x8, i16, 8, [i16; 8]);
impl_simd!(i16x16, i16, 16, [i16; 16]);

impl_simd!(u32x4, u32, 4, [u32; 4]);
impl_simd!(u32x8, u32, 8, [u32; 8]);

impl_simd!(i32x4, i32, 4, [i32; 4]);
impl_simd!(i32x8, i32, 8, [i32; 8]);

impl_simd!(u64x2, u64, 2, [u64; 2]);
impl_simd!(u64x4, u64, 4, [u64; 4]);

impl_simd!(i64x2, i64, 2, [i64; 2]);
impl_simd!(i64x4, i64, 4, [i64; 4]);

impl_simd!(f32x4, f32, 4, [f32; 4]);
impl_simd!(f32x8, f32, 8, [f32; 8]);

impl_simd!(f64x2, f64, 2, [f64; 2]);
impl_simd!(f64x4, f64, 4, [f64; 4]);
