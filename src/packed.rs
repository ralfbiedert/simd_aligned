use std::{ops::Range};

use super::traits::Simd;
use super::conversion::{simd_container_flat_slice, simd_container_flat_slice_mut};


#[derive(Clone, Debug)]
pub(crate) struct PackedMxN<T>
where
    T: Simd + Default + Clone,
{
    pub(crate) rows: usize,
    pub(crate) row_length: usize,
    pub(crate) vectors_per_row: usize,
    pub(crate) data: Vec<T>,
}

impl<T> PackedMxN<T>
where
    T: Simd + Default + Clone,
{
    #[inline]
    pub(crate) fn with(default: T, rows: usize, row_length: usize) -> Self {
        let vectors_per_row = match (row_length / T::LANES, row_length % T::LANES) {
            (x, 0) => x,
            (x, _) => x + 1,
        };
        
        Self {
            rows,
            row_length,
            vectors_per_row,
            data: vec![default; vectors_per_row * rows],
        }
    }

    /// Computes an offset for a vector and attribute.
    #[inline]
    pub(crate) fn row_start_offset(&self, row: usize) -> usize {
        row * self.vectors_per_row
    }

    /// Returns the range of SIMD vectors for the given row.
    #[inline]
    pub(crate) fn range_for_row(&self, row: usize) -> Range<usize> {
        let start = self.row_start_offset(row);
        let end = start + self.vectors_per_row;
        start..end
    }

    #[inline]
    pub(crate) fn row_as_flat_mut(&mut self, row: usize) -> &mut [T::Element] {
        let range = self.range_for_row(row);

        simd_container_flat_slice_mut(&mut self.data[range], self.row_length)
    }

    #[inline]
    pub(crate) fn row_as_flat(&self, row: usize) -> &[T::Element] {
        let range = self.range_for_row(row);
        simd_container_flat_slice(&self.data[range], self.row_length)
    }
}

#[cfg(test)]
mod test {
    use super::PackedMxN;
    use crate::f32x4;
    
    #[test]
    fn allocation_size() {
        let r_1 = PackedMxN::<f32x4>::with(f32x4::splat(0.0), 1, 4);
        let r_2 = PackedMxN::<f32x4>::with(f32x4::splat(0.0), 1, 5);

        assert_eq!(r_1.data.len(), 1);
        assert_eq!(r_2.data.len(), 2);
    }

    #[test]
    fn start_offset() {
        let r = PackedMxN::<f32x4>::with(f32x4::splat(0.0), 16, 16);

        assert_eq!(r.row_start_offset(0), 0);
        assert_eq!(r.row_start_offset(1), 4);
    }

    #[test]
    fn range() {
        let r = PackedMxN::<f32x4>::with(f32x4::splat(0.0), 16, 16);

        assert_eq!(r.range_for_row(2), 8..12);
    }

    #[test]
    fn slice() {
        let r = PackedMxN::<f32x4>::with(f32x4::splat(0.0), 16, 16);

        let s = r.row_as_flat(1);
        assert_eq!(s.len(), 16);
    }

    #[test]
    fn slice_mut() {
        let mut r = PackedMxN::<f32x4>::with(f32x4::splat(0.0), 16, 16);
        let s = r.row_as_flat_mut(1);

        assert_eq!(s.len(), 16);
    }
}
