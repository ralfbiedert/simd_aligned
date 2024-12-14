use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::traits::Simd;

use super::{
    conversion::{simd_container_flat_slice, simd_container_flat_slice_mut},
    packed::PackedMxN,
};

#[doc(hidden)]
pub trait AccessStrategy {
    fn flat_to_packed(x: usize, y: usize) -> (usize, usize);
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct Rows;

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct Columns;

impl AccessStrategy for Rows {
    #[inline]
    fn flat_to_packed(x: usize, y: usize) -> (usize, usize) {
        (x, y)
    }
}

impl AccessStrategy for Columns {
    #[inline]
    fn flat_to_packed(x: usize, y: usize) -> (usize, usize) {
        (y, x)
    }
}

/// A dynamic (heap allocated) matrix with one axis aligned for fast and safe SIMD access that
/// also provides a flat view on its data.
///
/// You can use [`MatSimd`] when you need to deal with multiple SIMD vectors, but
/// want them arranged in a compact cache-friendly manner. Internally this struct is backed by a
/// continuous vector of aligned vectors, and dynamically sliced according to row / column access.
///
/// # Example
///
/// ```rust
/// use simd_aligned::{MatSimd, arch::f32x4, Rows};
///
/// // Create a matrix of height 10x`f32` and width 5x`f32`, optimized for row access.
/// let mut m = MatSimd::<f32x4, Rows>::with_dimension(10, 5);
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
pub struct MatSimd<T, A>
where
    T: Simd + Default + Clone,
    A: AccessStrategy,
{
    pub(crate) simd_rows: PackedMxN<T>,
    phantom: PhantomData<A>,
}

impl<T, O> MatSimd<T, O>
where
    T: Simd + Default + Clone,
    O: AccessStrategy,
{
    /// Creates a new [`MatSimd`] with the given dimension.
    #[inline]
    #[must_use]
    pub fn with_dimension(width: usize, height: usize) -> Self {
        let (x, y) = O::flat_to_packed(width, height);

        Self {
            simd_rows: PackedMxN::with(T::default(), x, y),
            phantom: PhantomData,
        }
    }

    /// Returns the size as (`rows`, `columns`).
    #[must_use]
    pub fn dimension(&self) -> (usize, usize) {
        O::flat_to_packed(self.simd_rows.rows, self.simd_rows.row_length)
    }

    /// Provides a flat, immutable view of the contained data.
    #[inline]
    #[must_use]
    pub const fn flat(&self) -> MatFlat<'_, T, O> {
        MatFlat {
            matrix: self,
            phantom: PhantomData,
        }
    }

    /// Provides a flat mutable view of the contained data.
    #[inline]
    pub fn flat_mut(&mut self) -> MatFlatMut<'_, T, O> {
        MatFlatMut {
            matrix: self,
            phantom: PhantomData,
        }
    }
}

impl<T> MatSimd<T, Rows>
where
    T: Simd + Default + Clone,
{
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &[T] {
        let range = self.simd_rows.range_for_row(i);
        &self.simd_rows.data[range]
    }

    #[inline]
    #[must_use]
    pub const fn row_iter(&self) -> Matrix2DIter<'_, T, Rows> {
        Matrix2DIter { matrix: self, index: 0 }
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let range = self.simd_rows.range_for_row(i);
        &mut self.simd_rows.data[range]
    }

    #[inline]
    #[must_use]
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
}

impl<T> MatSimd<T, Columns>
where
    T: Simd + Default + Clone,
{
    #[inline]
    #[must_use]
    pub fn column(&self, i: usize) -> &[T] {
        let range = self.simd_rows.range_for_row(i);
        &self.simd_rows.data[range]
    }

    #[inline]
    #[must_use]
    pub const fn column_iter(&self) -> Matrix2DIter<'_, T, Columns> {
        Matrix2DIter { matrix: self, index: 0 }
    }

    #[inline]
    #[must_use]
    pub fn column_mut(&mut self, i: usize) -> &mut [T] {
        let range = self.simd_rows.range_for_row(i);
        &mut self.simd_rows.data[range]
    }

    #[inline]
    #[must_use]
    pub fn column_as_flat(&self, i: usize) -> &[T::Element] {
        let column = self.column(i);
        simd_container_flat_slice(column, self.simd_rows.row_length)
    }

    #[inline]
    #[must_use]
    pub fn column_as_flat_mut(&mut self, i: usize) -> &mut [T::Element] {
        let length = self.simd_rows.row_length;
        let column = self.column_mut(i);
        simd_container_flat_slice_mut(column, length)
    }
}

/// Produced by [`MatSimd::flat`], this allow for flat matrix access.
pub struct MatFlat<'a, T, A>
where
    T: Simd + Default + Clone + 'a,
    A: AccessStrategy + 'a,
{
    matrix: &'a MatSimd<T, A>,
    phantom: PhantomData<A>, // Do we actually need this / is there a better way?
}

/// Provided by [`MatSimd::flat_mut`], this allow for flat, mutable matrix access.
pub struct MatFlatMut<'a, T, A>
where
    T: Simd + Default + Clone + 'a,
    A: AccessStrategy + 'a,
{
    matrix: &'a mut MatSimd<T, A>,
    phantom: PhantomData<A>, // Do we actually need this / is there a better way?
}

impl<T, A> Index<(usize, usize)> for MatFlat<'_, T, A>
where
    T: Simd + Default + Clone,
    A: AccessStrategy,
{
    type Output = T::Element;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, x) = A::flat_to_packed(index.0, index.1);
        let row_slice = self.matrix.simd_rows.row_as_flat(row);

        &row_slice[x]
    }
}

impl<T, A> Index<(usize, usize)> for MatFlatMut<'_, T, A>
where
    T: Simd + Default + Clone,
    A: AccessStrategy,
{
    type Output = T::Element;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, x) = A::flat_to_packed(index.0, index.1);
        let row_slice = self.matrix.simd_rows.row_as_flat(row);

        &row_slice[x]
    }
}

impl<T, A> IndexMut<(usize, usize)> for MatFlatMut<'_, T, A>
where
    T: Simd + Default + Clone,
    A: AccessStrategy,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, x) = A::flat_to_packed(index.0, index.1);
        let row_slice = self.matrix.simd_rows.row_as_flat_mut(row);

        &mut row_slice[x]
    }
}

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct Matrix2DIter<'a, T, O>
where
    T: Simd + Default + Clone + 'a,
    O: AccessStrategy + 'a,
{
    /// Reference to the matrix we iterate over.
    pub(crate) matrix: &'a MatSimd<T, O>,

    /// Current index of vector iteration.
    pub(crate) index: usize,
}

impl<'a, T, O> Iterator for Matrix2DIter<'a, T, O>
where
    T: Simd + Default + Clone,
    O: AccessStrategy,
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
    use super::{Columns, MatSimd, Rows};
    use crate::arch::f32x4;

    #[test]
    fn allocation_size() {
        let m_1_1_r = MatSimd::<f32x4, Rows>::with_dimension(1, 1);
        let m_1_1_c = MatSimd::<f32x4, Columns>::with_dimension(1, 1);

        let m_5_5_r = MatSimd::<f32x4, Rows>::with_dimension(5, 5);
        let m_5_5_c = MatSimd::<f32x4, Columns>::with_dimension(5, 5);

        let m_1_4_r = MatSimd::<f32x4, Rows>::with_dimension(1, 4);
        let m_4_1_c = MatSimd::<f32x4, Columns>::with_dimension(4, 1);

        let m_4_1_r = MatSimd::<f32x4, Rows>::with_dimension(4, 1);
        let m_1_4_c = MatSimd::<f32x4, Columns>::with_dimension(1, 4);

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
        let mut m_5_5_r = MatSimd::<f32x4, Rows>::with_dimension(5, 5);
        let mut m_5_5_c = MatSimd::<f32x4, Columns>::with_dimension(5, 5);

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
        let mut m_1_5_r = MatSimd::<f32x4, Rows>::with_dimension(1, 5);
        let mut m_1_5_c = MatSimd::<f32x4, Columns>::with_dimension(1, 5);
        let mut m_5_1_c = MatSimd::<f32x4, Columns>::with_dimension(5, 1);
        let mut m_5_1_r = MatSimd::<f32x4, Rows>::with_dimension(5, 1);

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
