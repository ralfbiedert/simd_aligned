use std::ops::{Deref, DerefMut, Index, IndexMut};

use crate::traits::Simd;

use super::{
    conversion::{simd_container_flat_slice, simd_container_flat_slice_mut},
    packed::PackedMxN,
};

/// A dynamic (heap allocated) vector aligned for fast and safe SIMD access that also provides a
/// flat view on its data.
///
/// # Example
///
/// ```rust
/// use simd_aligned::*;
///
/// // Create a vector of f64x__ elements that, in total, will hold space
/// // for at least 4 f64 values. Internally this might be one f64x4, two f64x2,
/// // or one f64x8 where the 2nd half is hidden, depending on the current architecture.
/// let mut v = VecD::<f64x4>::with(0_f64, 4);
///
/// // Get a 'flat view' (&[f64]) into the SIMD vectors and fill it.
/// v.flat_mut().clone_from_slice(&[0.0, 1.0, 2.0, 3.0]);
/// ```

#[derive(Clone, Debug)]
pub struct VecD<T>
where
    T: Simd + Default + Clone,
{
    pub(crate) simd_rows: PackedMxN<T>,
}

impl<T> VecD<T>
where
    T: Simd + Default + Clone,
{
    /// Produce a [`VecD`] with the given element `t` as default and a flat size of `size`.
    #[inline]
    pub fn with(t: T::Element, size: usize) -> Self {
        Self {
            simd_rows: PackedMxN::with(T::splat(t), 1, size),
        }
    }

    /// Get a flat view for this [`VecD`].
    #[inline]
    #[must_use]
    pub fn flat(&self) -> &[T::Element] {
        simd_container_flat_slice(&self.simd_rows.data[..], self.simd_rows.row_length)
    }

    /// Get a flat, mutable view for this [`VecD`].
    #[inline]
    pub fn flat_mut(&mut self) -> &mut [T::Element] {
        simd_container_flat_slice_mut(&mut self.simd_rows.data[..], self.simd_rows.row_length)
    }
}

impl<T> Index<usize> for VecD<T>
where
    T: Simd + Default + Clone,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.simd_rows.data[index]
    }
}

impl<T> IndexMut<usize> for VecD<T>
where
    T: Simd + Default + Clone,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.simd_rows.data[index]
    }
}

impl<T> Deref for VecD<T>
where
    T: Simd + Default + Clone,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.simd_rows.data[..]
    }
}

impl<T> DerefMut for VecD<T>
where
    T: Simd + Default + Clone,
{
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.simd_rows.data[..]
    }
}

#[cfg(test)]
mod test {
    use super::VecD;
    use crate::f32x4;

    #[test]
    fn allocation_size() {
        let v_1 = VecD::<f32x4>::with(0.0f32, 4);
        let v_2 = VecD::<f32x4>::with(0.0f32, 5);

        assert_eq!(v_1.simd_rows.data.len(), 1);
        assert_eq!(v_2.simd_rows.data.len(), 2);
    }

    #[test]
    fn flat() {
        let mut v = VecD::<f32x4>::with(10.0f32, 16);
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
        let v = VecD::<f32x4>::with(0.0f32, 16);
        assert_eq!(&v[0], &v[0]);
    }
}
