use std::{ops::Range};

use super::{Alignment};

#[derive(Clone, Debug)]
pub(crate) struct SimdRows<T>
where
    T: Alignment
{
    pub(crate) rows: usize,
    pub(crate) row_length: usize,
    pub(crate) vectors_per_row: usize,
    pub(crate) data: Vec<T>,
}

pub struct Flat(usize);

impl<T> SimdRows<T>
where
    T: Alignment + Default + Clone, T::Type: Default + Clone
{
    #[inline]
    pub(crate) fn with(default: T::Type, rows: usize, row_length: Flat) -> SimdRows<T> {
        let vectors_per_row = match (row_length.0 / T::align(), row_length.0 % T::align()) {
            (x, 0) => x,
            (x, _) => x + 1,
        };
        
        SimdRows {
            rows,
            row_length: row_length.0,
            vectors_per_row,
            data: vec![Default::default(); vectors_per_row * rows],
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
    pub(crate) fn row_as_flat_mut(&mut self, row: usize) -> &mut [T::Type] {
        let range = self.range_for_row(row);
        let slice = &mut self.data[..];
        unimplemented!()
//        simd_container_flat_slice_mut(&mut slice[range], self.row_length)
    }

    #[inline]
    pub(crate) fn row_as_flat(&self, row: usize) -> &[T::Type] {
        let range = self.range_for_row(row);
        let slice = &self.data[..];

        unimplemented!()
//        simd_container_flat_slice(&slice[range], self.row_length)
    }
}

#[cfg(test)]
mod test {
    use super::{SimdRows, Flat};
    use crate::F32x2;
    
    #[test]
    fn allocation_size() {
        let r_1 = SimdRows::<F32x2>::with(0f32, 1, Flat(4));
        let r_2 = SimdRows::<F32x2>::with(0f32, 1, Flat(5));

        assert_eq!(r_1.data.len(), 1);
        assert_eq!(r_2.data.len(), 2);
    }

    #[test]
    fn start_offset() {
        let r = SimdRows::<F32x2>::with(0f32, 16, Flat(16));

        assert_eq!(r.row_start_offset(0), 0);
        assert_eq!(r.row_start_offset(1), 4);
    }

    #[test]
    fn range() {
        let r = SimdRows::<F32x2>::with(0f32, 16, Flat(16));

        assert_eq!(r.range_for_row(2), 8..12);
    }

    #[test]
    fn slice() {
        let r = SimdRows::<F32x2>::with(0f32, 16, Flat(16));

        let s = r.row_as_flat(1);
        assert_eq!(s.len(), 16);
    }

    #[test]
    fn slice_mut() {
        let mut r = SimdRows::<F32x2>::with(0f32, 16, Flat(16));
        let s = r.row_as_flat_mut(1);

        assert_eq!(s.len(), 16);
    }
}
