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

//     // TODO: Implement heuristics for architecture / target features.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod myarch {
    impl_vecs!(
        super::u8x32,
        super::i8x32,
        super::u16x16,
        super::i16x16,
        super::u32x8,
        super::i32x8,
        super::u64x4,
        super::i64x4,
        super::f32x8,
        super::f64x4
    );
}

#[cfg(any(target_arch = "aarch64"))]
pub mod myarch {
    impl_vecs!(
        super::u8x16,
        super::i8x16,
        super::u16x8,
        super::i16x8,
        super::u32x4,
        super::i32x4,
        super::u64x2,
        super::i64x2,
        super::f32x4,
        super::f64x2
    );
}

#[cfg(
    not(
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64"
        )
    )
)]
pub mod myarch {
    impl_vecs!(
        super::u8x16,
        super::i8x16,
        super::u16x8,
        super::i16x8,
        super::u32x4,
        super::i32x4,
        super::u64x2,
        super::i64x2,
        super::f32x4,
        super::f64x2
    );
}
