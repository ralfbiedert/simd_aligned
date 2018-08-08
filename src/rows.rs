use std::{
    fmt,
    iter::IntoIterator,
    marker::PhantomData,
    marker::{Copy, Sized},
    ops::Range,
    ops::{Index, IndexMut},
};

use super::container::Container;
use super::Simd;

#[derive(Clone, Debug)]
crate struct SimdRows<T, C>
where
    T: Simd + Default + Clone,
    C: Container<T>,
{
    crate rows: usize,
    crate row_length: usize,
    crate vectors_per_row: usize,
    crate data: C,
    phantom: PhantomData<T>, // Do we actually need this / is there a better way?
}

impl<T, C> SimdRows<T, C>
where
    T: Simd + Default + Clone,
    C: Container<T>,
{
    #[inline]
    crate fn with(default: T, rows: usize, row_length: usize) -> SimdRows<T, C> {
        let vectors_per_row = match (row_length / T::LANES, row_length % T::LANES) {
            (x, 0) => x,
            (x, _) => x + 1,
        };

        SimdRows {
            rows,
            row_length,
            vectors_per_row,
            phantom: PhantomData,
            data: C::with(default, vectors_per_row * rows),
        }
    }

    /// Computes an offset for a vector and attribute.
    #[inline]
    crate fn row_start_offset(&self, row: usize) -> usize {
        row * self.vectors_per_row
    }

    /// Returns the range of SIMD vectors for the given row.
    #[inline]
    crate fn range_for_row(&self, row: usize) -> Range<usize> {
        let start = self.row_start_offset(row);
        let end = start + self.vectors_per_row;
        start..end
    }
}
