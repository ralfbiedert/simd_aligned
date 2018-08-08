use super::Simd;

#[inline]
crate fn simd_container_flat_slice<T>(data: &[T], length: usize) -> &[T::Element]
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
crate fn simd_container_flat_slice_mut<T>(data: &mut [T], length: usize) -> &mut [T::Element]
where
    T: Simd + Default + Clone,
{
    let mut_ptr = data.as_mut_ptr() as *mut T::Element;

    // See comment above
    unsafe { std::slice::from_raw_parts_mut(mut_ptr, length) }
}

#[inline]
pub fn slice_as_flat<T>(data: &[T]) -> &[T::Element]
where
    T: Simd + Default + Clone,
{
    simd_container_flat_slice(data, data.len() * T::LANES)
}

#[inline]
pub fn slice_as_flat_mut<T>(data: &mut [T]) -> &mut [T::Element]
where
    T: Simd + Default + Clone,
{
    simd_container_flat_slice_mut(data, data.len() * T::LANES)
}
