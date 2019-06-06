use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::traits::Simd;

use super::conversion::{simd_container_flat_slice, simd_container_flat_slice_mut};
use super::rows::SimdRows;

#[doc(hidden)]
pub trait OptimizationStrategy {
    fn translate_indices_to_simdrows(x: usize, y: usize) -> (usize, usize);

    fn assert_column();

    fn assert_row();
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct RowOptimized;

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ColumnOptimized;

impl OptimizationStrategy for RowOptimized {
    #[inline]
    fn translate_indices_to_simdrows(x: usize, y: usize) -> (usize, usize) {
        (x, y)
    }

    fn assert_row() {}

    fn assert_column() {
        panic!("Asserting the matrix is column-optimized but it is actually row-optimized.");
    }
}

impl OptimizationStrategy for ColumnOptimized {
    #[inline]
    fn translate_indices_to_simdrows(x: usize, y: usize) -> (usize, usize) {
        (y, x)
    }

    fn assert_row() {
        panic!("Asserting the matrix is row-optimized but it is actually column-optimized.");
    }

    fn assert_column() {}
}

/// A matrix with one axis aligned for fast and safe SIMD access that also provides a flat view on
/// its data.
///
/// You can use [SimdMatrix] when you need to deal with multiple SIMD vectors, but
/// want them arranged in a compact cache-friendly manner. Internally this struct is backed by a
/// continuous vector of aligned vectors, and dynamically sliced according to row / column access.
///
/// # Example
///
/// ```rust
/// use packed_simd::*;
/// use simd_aligned::*;
///
/// // Create a matrix of height 10x`f32` and width 5x`f32`, optimized for row access.
/// let mut m = SimdMatrix::<f32s, RowOptimized>::with_dimension(10, 5);
///
/// // A `RowOptimized` matrix provides `row` access. In this example, you could query
/// // rows `0` to `9` and receive vectors that can hold a total of at least `5` elements.
/// let _ = m.row(4);
///
/// // But accessing columns doesn't work, as there is no continuous view in memory.
/// // m.column(3); --> panic!
///
/// // However, you can always get a flat view of the matrix, for "scalar-speed"
/// // query and update all elements:
/// let mut m_flat = m.flat_mut();
///
/// m_flat[(2, 4)] = 42_f32;
/// ```
#[derive(Clone, Debug)]
pub struct SimdMatrix<T, O>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    pub(crate) simd_rows: SimdRows<T>,
    phantom: PhantomData<O>,
}

impl<T, O> SimdMatrix<T, O>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    #[inline]
    pub fn with_dimension(width: usize, height: usize) -> Self {
        let (x, y) = O::translate_indices_to_simdrows(width, height);

        SimdMatrix {
            simd_rows: SimdRows::with(T::default(), x, y),
            phantom: PhantomData,
        }
    }

    pub fn dimension(&self) -> (usize, usize) {
        O::translate_indices_to_simdrows(self.simd_rows.rows, self.simd_rows.row_length)
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        O::assert_row();
        let range = self.simd_rows.range_for_row(i);
        &self.simd_rows.data[range]
    }

    #[inline]
    pub fn row_iter(&self) -> SimdMatrixIter<'_, T, O> {
        O::assert_row();

        SimdMatrixIter {
            matrix: &self,
            index: 0,
        }
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        O::assert_row();
        let range = self.simd_rows.range_for_row(i);
        &mut self.simd_rows.data[range]
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
        let range = self.simd_rows.range_for_row(i);
        &self.simd_rows.data[range]
    }

    #[inline]
    pub fn column_iter(&self) -> SimdMatrixIter<'_, T, O> {
        O::assert_column();

        SimdMatrixIter {
            matrix: &self,
            index: 0,
        }
    }

    #[inline]
    pub fn column_mut(&mut self, i: usize) -> &mut [T] {
        O::assert_column();
        let range = self.simd_rows.range_for_row(i);
        &mut self.simd_rows.data[range]
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

/// Produced by [SimdMatrix::flat], this allow for flat matrix access.
pub struct SimdMatrixFlat<'a, T: 'a, O: 'a>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    matrix: &'a SimdMatrix<T, O>,
    phantom: PhantomData<O>, // Do we actually need this / is there a better way?
}

/// Provided by [SimdMatrix::flat_mut], this allow for flat, mutable matrix access.
pub struct SimdMatrixFlatMut<'a, T: 'a, O: 'a>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    matrix: &'a mut SimdMatrix<T, O>,
    phantom: PhantomData<O>, // Do we actually need this / is there a better way?
}

impl<'a, T, O> Index<(usize, usize)> for SimdMatrixFlat<'a, T, O>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    type Output = T::Element;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, x) = O::translate_indices_to_simdrows(index.0, index.1);
        let row_slice = self.matrix.simd_rows.row_as_flat(row);

        &row_slice[x]
    }
}

impl<'a, T, O> Index<(usize, usize)> for SimdMatrixFlatMut<'a, T, O>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    type Output = T::Element;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, x) = O::translate_indices_to_simdrows(index.0, index.1);
        let row_slice = self.matrix.simd_rows.row_as_flat(row);

        &row_slice[x]
    }
}

impl<'a, T, O> IndexMut<(usize, usize)> for SimdMatrixFlatMut<'a, T, O>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, x) = O::translate_indices_to_simdrows(index.0, index.1);
        let row_slice = self.matrix.simd_rows.row_as_flat_mut(row);

        &mut row_slice[x]
    }
}

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct SimdMatrixIter<'a, T: 'a, O: 'a>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    /// Reference to the matrix we iterate over.
    pub(crate) matrix: &'a SimdMatrix<T, O>,

    /// Current index of vector iteration.
    pub(crate) index: usize,
}

impl<'a, T, O> Iterator for SimdMatrixIter<'a, T, O>
where
    T: Simd + Default + Clone,
    O: OptimizationStrategy,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.matrix.simd_rows.rows {
            None
        } else {
            let range = self.matrix.simd_rows.range_for_row(self.index);
            self.index += 1;
            Some(&self.matrix.simd_rows.data[range])
        }
    }
}

#[cfg(test)]
mod test {
    use super::{ColumnOptimized, RowOptimized, SimdMatrix};
    use crate::*;

    #[test]
    fn allocation_size() {
        let m_1_1_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(1, 1);
        let m_1_1_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(1, 1);

        let m_5_5_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(5, 5);
        let m_5_5_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(5, 5);

        let m_1_4_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(1, 4);
        let m_4_1_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(4, 1);

        let m_4_1_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(4, 1);
        let m_1_4_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(1, 4);

        assert_eq!(m_1_1_r.simd_rows.data.len(), 1);
        assert_eq!(m_1_1_c.simd_rows.data.len(), 1);

        assert_eq!(m_5_5_r.simd_rows.data.len(), 10);
        assert_eq!(m_5_5_c.simd_rows.data.len(), 10);

        assert_eq!(m_1_4_r.simd_rows.data.len(), 1);
        assert_eq!(m_4_1_c.simd_rows.data.len(), 1);

        assert_eq!(m_4_1_r.simd_rows.data.len(), 4);
        assert_eq!(m_1_4_c.simd_rows.data.len(), 4);
    }

    #[test]

    fn access() {
        let mut m_5_5_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(5, 5);
        let mut m_5_5_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(5, 5);

        assert_eq!(m_5_5_c.column(0).len(), 2);
        assert_eq!(m_5_5_c.column_mut(0).len(), 2);
        assert_eq!(m_5_5_c.column_as_flat(0).len(), 5);
        assert_eq!(m_5_5_c.column_as_flat_mut(0).len(), 5);

        assert_eq!(m_5_5_r.row(0).len(), 2);
        assert_eq!(m_5_5_r.row_mut(0).len(), 2);
        assert_eq!(m_5_5_r.row_as_flat(0).len(), 5);
        assert_eq!(m_5_5_r.row_as_flat_mut(0).len(), 5);

        let mut count = 0;

        for _ in m_5_5_c.column_iter() {
            count += 1
        }

        for _ in m_5_5_r.row_iter() {
            count += 1
        }

        let r1 = m_5_5_r.row(3);
        let r2 = m_5_5_r.row(4);

        let mut sum = f32x4::splat(0_f32);

        for (x, y) in r1.iter().zip(r2) {
            sum += *x + *y;
        }

        assert_eq!(count, 10);
    }

    #[test]

    fn flattened() {
        let mut m_1_5_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(1, 5);
        let mut m_1_5_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(1, 5);
        let mut m_5_1_c = SimdMatrix::<f32x4, ColumnOptimized>::with_dimension(5, 1);
        let mut m_5_1_r = SimdMatrix::<f32x4, RowOptimized>::with_dimension(5, 1);

        let mut m_1_5_r_flat = m_1_5_r.flat_mut();
        let mut m_1_5_c_flat = m_1_5_c.flat_mut();
        let mut m_5_1_c_flat = m_5_1_c.flat_mut();
        let mut m_5_1_r_flat = m_5_1_r.flat_mut();

        m_1_5_r_flat[(0, 4)] = 1.0;
        m_1_5_c_flat[(0, 4)] = 1.0;
        m_5_1_r_flat[(4, 0)] = 1.0;
        m_5_1_c_flat[(4, 0)] = 1.0;
    }
}
