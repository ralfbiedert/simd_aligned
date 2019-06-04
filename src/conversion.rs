use super::Alignment;

#[inline]
pub(crate) fn simd_container_flat_slice<T>(data: &[T], length: usize) -> &[T::Type]
where
    T: Alignment,
    T::Type : Default
{
    let ptr = data.as_ptr() as *const T::Type;

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
pub(crate) fn simd_container_flat_slice_mut<T>(data: &mut [T], length: usize) -> &mut [T::Type]
where
    T: Alignment,
    T::Type : Default
{
    let mut_ptr = data.as_mut_ptr() as *mut T::Type;

    // See comment above
    unsafe { std::slice::from_raw_parts_mut(mut_ptr, length) }
}

