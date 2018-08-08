use std::ops::{Index, IndexMut};

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
