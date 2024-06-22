//! Contains vector definitions with a fixed bit width.
//!
#![allow(non_camel_case_types)]


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
        crate::u8x16,
        crate::i8x16,
        crate::u16x8,
        crate::i16x8,
        crate::u32x4,
        crate::i32x4,
        crate::u64x2,
        crate::i64x2,
        crate::f32x4,
        crate::f64x2
    );
}

/// Vectors with fixed length of 256 bits.
pub mod x256 {
    impl_vecs!(
        crate::u8x32,
        crate::i8x32,
        crate::u16x16,
        crate::i16x16,
        crate::u32x8,
        crate::i32x8,
        crate::u64x4,
        crate::i64x4,
        crate::f32x8,
        crate::f64x4
    );
}

/// Vectors with fixed length of 512 bits.
pub mod x512 {
    impl_vecs!(
        crate::u8x64,
        crate::i8x64,
        crate::u16x32,
        crate::i16x32,
        crate::u32x16,
        crate::i32x16,
        crate::u64x8,
        crate::i64x8,
        crate::f32x16,
        crate::f64x8
    );
}

//     // TODO: Implement heuristics for architecture / target features.

/// Vectors for the current architecture.
pub mod current {

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod current {
        //! Vectors for the current arch.
        impl_vecs!(
            crate::u8x32,
            crate::i8x32,
            crate::u16x16,
            crate::i16x16,
            crate::u32x8,
            crate::i32x8,
            crate::u64x4,
            crate::i64x4,
            crate::f32x8,
            crate::f64x4
        );
    }

    #[cfg(any(target_arch = "aarch64"))]
    mod current {
        impl_vecs!(
            crate::u8x16,
            crate::i8x16,
            crate::u16x8,
            crate::i16x8,
            crate::u32x4,
            crate::i32x4,
            crate::u64x2,
            crate::i64x2,
            crate::f32x4,
            crate::f64x2
        );
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    mod current {
        impl_vecs!(
            crate::u8x16,
            crate::i8x16,
            crate::u16x8,
            crate::i16x8,
            crate::u32x4,
            crate::i32x4,
            crate::u64x2,
            crate::i64x2,
            crate::f32x4,
            crate::f64x2
        );
    }

    pub use current::*;
}
