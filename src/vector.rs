use std::ops::{Deref, Index, IndexMut};

use super::container::Container;
use super::conversion::{simd_container_flat_slice, simd_container_flat_slice_mut};
use super::rows::SimdRows;
use super::Simd;

#[derive(Debug)]
pub struct SimdVector<T>
where
    T: Simd + Default + Clone,
{
    crate simd_rows: SimdRows<T, Vec<T>>,
}

impl<T> SimdVector<T>
where
    T: Simd + Default + Clone,
{
    #[inline]
    pub fn with_size(size: usize) -> Self {
        SimdVector {
            simd_rows: SimdRows::with(T::default(), 1, size),
        }
    }

    #[inline]
    pub fn flat(&self) -> &[T::Element] {
        simd_container_flat_slice(self.simd_rows.data.slice(), self.simd_rows.row_length)
    }

    #[inline]
    pub fn flat_mut(&mut self) -> &mut [T::Element] {
        simd_container_flat_slice_mut(self.simd_rows.data.slice_mut(), self.simd_rows.row_length)
    }
}

impl<T> Index<usize> for SimdVector<T>
where
    T: Simd + Default + Clone,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.simd_rows.data[index]
    }
}

impl<T> IndexMut<usize> for SimdVector<T>
where
    T: Simd + Default + Clone,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.simd_rows.data[index]
    }
}

impl<T> Deref for SimdVector<T>
where
    T: Simd + Default + Clone,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.simd_rows.data[..]
    }
}

mod test {
    use super::SimdVector;
    use crate::f32x4;
    use std::ops::Range;

    #[test]
    fn allocation_size() {
        let v_1 = SimdVector::<f32x4>::with_size(4);
        let v_2 = SimdVector::<f32x4>::with_size(5);

        assert_eq!(v_1.simd_rows.data.len(), 1);
        assert_eq!(v_2.simd_rows.data.len(), 2);
    }

    #[test]
    fn flat() {
        let mut v = SimdVector::<f32x4>::with_size(16);
        let r_m = v.flat_mut();

        assert_eq!(r_m.len(), 16);

        for x in r_m {
            *x = 1.0
        }

        let mut sum = 0.0;
        let r = v.flat();

        assert_eq!(r.len(), 16);

        for x in r {
            sum += x;
        }

        assert!((sum - 16.0).abs() <= std::f32::EPSILON);
    }

    #[test]
    fn deref() {
        let v = SimdVector::<f32x4>::with_size(16);
        assert_eq!(&v[0], &v[0]);
    }
}
