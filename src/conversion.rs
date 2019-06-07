use crate::traits::Simd;

#[inline]
pub(crate) fn simd_container_flat_slice<T>(data: &[T], length: usize) -> &[T::Element]
where
    T: Simd + Default + Clone,
{
    let ptr = data.as_ptr() as *const T::Element;

    // This "should be safe(tm)" since:
    //
    // 1) a slice of `N x f32x8` elements are transformed into a slice of
    // `attributes * f32` elements, where `attributes <=  N * 8`.
    //
    // 2) The lifetime of the returned value should automatically match the self borrow.
    //
    // Having said this, as soon as `std::simd` (or similar) provides a safe way of handling
    // that for us, these lines should be removed!
    unsafe { std::slice::from_raw_parts(ptr, length) }
}

#[inline]
pub(crate) fn simd_container_flat_slice_mut<T>(data: &mut [T], length: usize) -> &mut [T::Element]
where
    T: Simd + Default + Clone,
{
    let mut_ptr = data.as_mut_ptr() as *mut T::Element;

    // See comment above
    unsafe { std::slice::from_raw_parts_mut(mut_ptr, length) }
}

/// Converts an slice of SIMD vectors into a flat slice of elements.
///
/// # Example
/// ```rust
/// use packed_simd::*;
/// use simd_aligned::*;
///
/// let packed = [f32x4::splat(0_f32); 4];
///
/// let flat = packed_as_flat(&packed);
///
/// assert_eq!(flat.len(), 16);
/// ```
#[inline]
pub fn packed_as_flat<T>(data: &[T]) -> &[T::Element]
where
    T: Simd + Default + Clone,
{
    simd_container_flat_slice(data, data.len() * T::LANES)
}

/// Converts a mutable slice of SIMD vectors into a flat slice of elements.
/// # Example
/// ```rust
/// use packed_simd::*;
/// use simd_aligned::*;
///
/// let mut packed = [f32x4::splat(0_f32); 4];
///
/// let flat = packed_as_flat_mut(&mut packed);
///
/// assert_eq!(flat.len(), 16);
/// ```
#[inline]
pub fn packed_as_flat_mut<T>(data: &mut [T]) -> &mut [T::Element]
where
    T: Simd + Default + Clone,
{
    simd_container_flat_slice_mut(data, data.len() * T::LANES)
}

#[cfg(test)]
mod test {
    use super::{packed_as_flat, packed_as_flat_mut};
    use packed_simd::*;

    #[test]
    fn slice_flattening() {
        let x_0 = [f32x4::splat(0.0); 0];
        let x_1 = [f32x4::splat(0.0); 1];

        let mut x_0_m = [f32x4::splat(0.0); 0];
        let mut x_1_m = [f32x4::splat(0.0); 1];

        let y_0 = packed_as_flat(&x_0);
        let y_1 = packed_as_flat(&x_1);

        let y_0_m = packed_as_flat_mut(&mut x_0_m);
        let y_1_m = packed_as_flat_mut(&mut x_1_m);

        assert_eq!(y_0.len(), 0);
        assert_eq!(y_1.len(), 4);
        assert_eq!(y_0_m.len(), 0);
        assert_eq!(y_1_m.len(), 4);
    }
}
