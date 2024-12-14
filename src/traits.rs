//! Unified views on SIMD types.

/// This is mostly copy-paste from `packed_simd`, where this trait is unfortunately
/// sealed right now. In the future this might come from `std::simd`.
pub trait Simd {
    /// Element type of the SIMD vector
    type Element;
    /// The number of elements in the SIMD vector.
    const LANES: usize;
    /// The type: `[u32; Self::N]`.
    type LanesType;

    /// Added for convenience
    fn splat(t: Self::Element) -> Self;

    /// Added for convenience
    fn as_array(&self) -> &[Self::Element];

    fn sum(&self) -> Self::Element;
}
