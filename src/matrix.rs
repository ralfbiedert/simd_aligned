use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use super::container::Container;
use super::conversion::{simd_container_flat_slice, simd_container_flat_slice_mut};
use super::rows::SimdRows;
use super::Simd;

pub trait Optimized {
    fn translate_indices_to_simdrows(x: usize, y: usize) -> (usize, usize);

    fn assert_column();

    fn assert_row();
}

pub struct RowOptimized;

pub struct ColumnOptimized;

impl Optimized for RowOptimized {
    #[inline]
    fn translate_indices_to_simdrows(x: usize, y: usize) -> (usize, usize) {
        (x, y)
    }

    fn assert_row() {}

    fn assert_column() {
        panic!("Asserting the matrix is column-optimized but it is actually row-optimized.");
    }
}

impl Optimized for ColumnOptimized {
    #[inline]
    fn translate_indices_to_simdrows(x: usize, y: usize) -> (usize, usize) {
        (y, x)
    }

    fn assert_row() {
        panic!("Asserting the matrix is row-optimized but it is actually column-optimized.");
    }

    fn assert_column() {}
}

#[derive(Debug)]
pub enum OptimizedFor {
    RowAccess,
    ColumnAccess,
}

#[derive(Debug)]
pub struct SimdMatrix<T, O>
where
    T: Simd + Default + Clone,
    O: Optimized,
{
    crate simd_rows: SimdRows<T, Vec<T>>,
    phantom: PhantomData<O>,
}

impl<T, O> SimdMatrix<T, O>
where
    T: Simd + Default + Clone,
    O: Optimized,
{
    #[inline]
    pub fn with_dimension(width: usize, height: usize, optimized_for: OptimizedFor) -> Self {
        let (x, y) = O::translate_indices_to_simdrows(width, height);

        SimdMatrix {
            simd_rows: SimdRows::with(T::default(), x, y),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        O::assert_row();
        &self.simd_rows.data.slice()[self.simd_rows.range_for_row(i)]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        O::assert_row();
        let range = self.simd_rows.range_for_row(i);
        &mut self.simd_rows.data.slice_mut()[range]
    }

    #[inline]
    pub fn row_as_flat(&self, i: usize) -> &[T::Element] {
        let row = self.row(i);
        simd_container_flat_slice(row, self.simd_rows.row_length)
    }

    #[inline]
    pub fn row_as_flat_mut(&mut self, i: usize) -> &mut [T::Element] {
        let length = self.simd_rows.row_length;
        let row = self.row_mut(i);
        simd_container_flat_slice_mut(row, length)
    }

    #[inline]
    pub fn column(&self, i: usize) -> &[T] {
        O::assert_column();
        &self.simd_rows.data.slice()[self.simd_rows.range_for_row(i)]
    }

    #[inline]
    pub fn column_mut(&mut self, i: usize) -> &mut [T] {
        O::assert_column();
        let range = self.simd_rows.range_for_row(i);
        &mut self.simd_rows.data.slice_mut()[range]
    }

    #[inline]
    pub fn column_as_flat(&self, i: usize) -> &[T::Element] {
        let column = self.column(i);
        simd_container_flat_slice(column, self.simd_rows.row_length)
    }

    #[inline]
    pub fn column_as_flat_mut(&mut self, i: usize) -> &mut [T::Element] {
        let length = self.simd_rows.row_length;
        let column = self.column_mut(i);
        simd_container_flat_slice_mut(column, length)
    }

    #[inline]
    pub fn flat(&self) -> SimdMatrixFlat<'_, T, O> {
        SimdMatrixFlat {
            matrix: self,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn flat_mut(&mut self) -> SimdMatrixFlatMut<'_, T, O> {
        SimdMatrixFlatMut {
            matrix: self,
            phantom: PhantomData,
        }
    }
}

pub struct SimdMatrixFlat<'a, T: 'a, O: 'a>
where
    T: Simd + Default + Clone,
    O: Optimized,
{
    matrix: &'a SimdMatrix<T, O>,
    phantom: PhantomData<O>, // Do we actually need this / is there a better way?
}

pub struct SimdMatrixFlatMut<'a, T: 'a, O: 'a>
where
    T: Simd + Default + Clone,
    O: Optimized,
{
    matrix: &'a mut SimdMatrix<T, O>,
    phantom: PhantomData<O>, // Do we actually need this / is there a better way?
}

// impl<'a, T, O> Index<(usize, usize)> for SimdMatrixFlat<'a, T, O>
// where
//     T: Simd + Default + Clone,
//     O: Optimized,
// {
//     type Output = T::Element;

//     #[inline]
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.simd_rows.data[index]
//     }
// }

// impl<'a, T, O> IndexMut<(usize, usize)> for SimdMatrixFlat<'a, T, O>
// where
//     T: Simd + Default + Clone,
//     O: Optimized,
// {
//     #[inline]
//     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
//         &mut self.simd_rows.data[index]
//     }
// }
