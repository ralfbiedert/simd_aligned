//! Contains vector definitions with a fixed bit width.
//!
#![allow(non_camel_case_types)]

use packed_simd::*;

macro_rules! impl_vecs {
    ($u8s:ty, $i8s:ty, $u16s:ty, $i16s:ty, $u32s:ty, $i32s:ty, $u64s:ty, $i64s:ty, $f32s:ty, $f64s:ty) => {
        /// The widest `u8x` type natively supported on the current platform.
        pub type u8s = $u8s;

        /// The widest `i8x` type natively supported on the current platform.
        pub type i8s = $i8s;

        /// The widest `u16x` type natively supported on the current platform.
        pub type u16s = $u16s;

        /// The widest `i16x` type natively supported on the current platform.
        pub type i16s = $i16s;

        /// The widest `u32x` type natively supported on the current platform.
        pub type u32s = $u32s;

        /// The widest `i32x` type natively supported on the current platform.
        pub type i32s = $i32s;

        /// The widest `u64x` type natively supported on the current platform.
        pub type u64s = $u64s;

        /// The widest `i64x` type natively supported on the current platform.
        pub type i64s = $i64s;

        /// The widest `f32x` type natively supported on the current platform.
        pub type f32s = $f32s;

        /// The widest `f64x` type natively supported on the current platform.
        pub type f64s = $f64s;
    };
}

/// Vectors with fixed length of 128 bits.
pub mod x128 {
    impl_vecs!(
        super::super::u8x16,
        super::super::i8x16,
        super::super::u16x8,
        super::super::i16x8,
        super::super::u32x4,
        super::super::i32x4,
        super::super::u64x2,
        super::super::i64x2,
        super::super::f32x4,
        super::super::f64x2
    );
}

/// Vectors with fixed length of 256 bits.
pub mod x256 {
    impl_vecs!(
        super::super::u8x32,
        super::super::i8x32,
        super::super::u16x16,
        super::super::i16x16,
        super::super::u32x8,
        super::super::i32x8,
        super::super::u64x4,
        super::super::i64x4,
        super::super::f32x8,
        super::super::f64x4
    );
}

/// Vectors with fixed length of 512 bits.
pub mod x512 {
    impl_vecs!(
        super::super::u8x64,
        super::super::i8x64,
        super::super::u16x32,
        super::super::i16x32,
        super::super::u32x16,
        super::super::i32x16,
        super::super::u64x8,
        super::super::i64x8,
        super::super::f32x16,
        super::super::f64x8
    );
}

//     // TODO: Implement heuristics for architecture / target features.

/// Vectors for the current architecture.
pub mod current {

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod current {
        //! Vectors for the current arch.
        impl_vecs!(
            super::super::u8x32,
            super::super::i8x32,
            super::super::u16x16,
            super::super::i16x16,
            super::super::u32x8,
            super::super::i32x8,
            super::super::u64x4,
            super::super::i64x4,
            super::super::f32x8,
            super::super::f64x4
        );
    }

    #[cfg(any(target_arch = "aarch64"))]
    mod current {
        impl_vecs!(
            super::super::u8x16,
            super::super::i8x16,
            super::super::u16x8,
            super::super::i16x8,
            super::super::u32x4,
            super::super::i32x4,
            super::super::u64x2,
            super::super::i64x2,
            super::super::f32x4,
            super::super::f64x2
        );
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    mod current {
        impl_vecs!(
            super::super::u8x16,
            super::super::i8x16,
            super::super::u16x8,
            super::super::i16x8,
            super::super::u32x4,
            super::super::i32x4,
            super::super::u64x2,
            super::super::i64x2,
            super::super::f32x4,
            super::super::f64x2
        );
    }

    pub use current::*;
}
