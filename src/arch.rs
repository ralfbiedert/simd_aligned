#![allow(non_camel_case_types)]

use packed_simd::*;

macro_rules! impl_vecs {
    ($u8s:ty, $i8s:ty, $u16s:ty, $i16s:ty, $u32s:ty, $i32s:ty, $u64s:ty, $i64s:ty, $f32s:ty, $f64s:ty) => {
        /// The widest `u8x__` type natively supported on the current platform.
        pub type u8s = $u8s;

        /// The widest `i8x__` type natively supported on the current platform.
        pub type i8s = $i8s;

        /// The widest `u16x__` type natively supported on the current platform.
        pub type u16s = $u16s;

        /// The widest `i16x__` type natively supported on the current platform.
        pub type i16s = $i16s;

        /// The widest `u32x__` type natively supported on the current platform.
        pub type u32s = $u32s;

        /// The widest `i32x__` type natively supported on the current platform.
        pub type i32s = $i32s;

        /// The widest `u64x__` type natively supported on the current platform.
        pub type u64s = $u64s;

        /// The widest `i64x__` type natively supported on the current platform.
        pub type i64s = $i64s;

        /// The widest `f32x__` type natively supported on the current platform.
        pub type f32s = $f32s;

        /// The widest `f64x__` type natively supported on the current platform.
        pub type f64s = $f64s;
    };
}

mod x86 {
    impl_vecs!(
        super::u8x16,
        super::i8x16,
        super::u16x8,
        super::i16x8,
        super::u32x4,
        super::i32x4,
        super::u64x2,
        super::i64x2,
        super::f32x8,
        super::f64x4
    );
}

pub mod myarch {
    // TODO: Implement heuristics for architecture / target features.
    pub use super::x86::*;
}
