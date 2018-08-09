use std::ops::{Deref, DerefMut, Index, IndexMut};

use crate::sealed::Simd;

use super::container::Container;
use super::conversion::{simd_container_flat_slice, simd_container_flat_slice_mut};
use super::rows::SimdRows;

/// A vector aligned for fast and safe SIMD access that also provides a flat view on its data.
///
/// # Example
///
/// ```rust
/// use packed_simd::*;
/// use simd_aligned::*;
///
/// // Create a vector of f64x__ elements that, in total, will hold space
/// // for at least 4 f64 values. Internally this might be one f64x4, two f64x2,
/// // or one f64x8 where the 2nd half is hidden, depending on the current architecture.
/// let mut v = SimdVector::<f64s>::with(0_f64, 4);
///
/// // Get a 'flat view' (&[f64]) into the SIMD vectors and fill it.
/// v.flat_mut().clone_from_slice(&[0.0, 1.0, 2.0, 3.0]);
/// ```

#[derive(Clone, Debug)]
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
    /// Produce a [SimdVector] with the given element `t` as default and a flat size of `size`.
    #[inline]
    pub fn with(t: T::Element, size: usize) -> Self {
        SimdVector {
            simd_rows: SimdRows::with(T::splat(t), 1, size),
        }
    }

    /// Get a flat view for this [SimdVector].
    #[inline]
    pub fn flat(&self) -> &[T::Element] {
        simd_container_flat_slice(self.simd_rows.data.slice(), self.simd_rows.row_length)
    }

    /// Get a flat, mutable view for this [SimdVector].
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

impl<T> DerefMut for SimdVector<T>
where
    T: Simd + Default + Clone,
{
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.simd_rows.data[..]
    }
}

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct SimdVectorIter<'a, T: 'a>
where
    T: Simd + Default + Clone,
{
    /// Reference to the matrix we iterate over.
    crate vector: &'a SimdVector<T>,

    /// Current index of vector iteration.
    crate index: usize,
}

// #[derive(Debug)]
// pub struct SimdVectorIterMut<'a, T: 'a>
// where
//     T: Simd + Default + Clone,
// {
//     /// Reference to the matrix we iterate over.
//     crate vector: &'a mut SimdVector<T>,

//     /// Current index of vector iteration.
//     crate index: usize,
// }

// impl<'a, T> Iterator for SimdVectorIter<'a, T>
// where
//     T: Simd + Default + Clone,
// {
//     type Item = &'a T;

//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.vector.simd_rows.vectors_per_row {
//             None
//         } else {
//             let rval = Some(&self.vector.simd_rows.data[self.index]);
//             self.index += 1;
//             rval
//         }
//     }
// }

// impl<'a, T> Iterator for SimdVectorIterMut<'a, T>
// where
//     T: Simd + Default + Clone,
// {
//     type Item = &'a mut T;

//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         let index = self.index;

//         return None;

//         // if index >= self.vector.simd_rows.vectors_per_row {
//         //     return None;
//         // }

//         // self.index += 1;
//         // let rval = Some(&mut self.vector.simd_rows.data[index]);
//         // rval
//     }
// }

// impl<'a, T> IntoIterator for &'a SimdVector<T>
// where
//     T: Simd + Default + Clone,
// {
//     type Item = &'a T;
//     type IntoIter = SimdVectorIter<'a, T>;

//     fn into_iter(self) -> Self::IntoIter {
//         SimdVectorIter {
//             vector: &self,
//             index: 0,
//         }
//     }
// }

// impl<'a, T> IntoIterator for &'a mut SimdVector<T>
// where
//     T: Simd + Default + Clone,
// {
//     type Item = &'a mut T;
//     type IntoIter = SimdVectorIterMut<'a, T>;

//     fn into_iter(self) -> Self::IntoIter {
//         SimdVectorIterMut {
//             vector: self,
//             index: 0,
//         }
//     }
// }

mod test {
    use super::SimdVector;
    use crate::f32x4;
    use std::ops::Range;

    #[test]
    fn allocation_size() {
        let v_1 = SimdVector::<f32x4>::with(0.0f32, 4);
        let v_2 = SimdVector::<f32x4>::with(0.0f32, 5);

        assert_eq!(v_1.simd_rows.data.len(), 1);
        assert_eq!(v_2.simd_rows.data.len(), 2);
    }

    #[test]
    fn flat() {
        let mut v = SimdVector::<f32x4>::with(10.0f32, 16);
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
        let v = SimdVector::<f32x4>::with(0.0f32, 16);
        assert_eq!(&v[0], &v[0]);
    }
}
