use std::ops::{Deref, Index, IndexMut};

use super::container::Container;
use super::conversion::{simd_container_flat_slice, simd_container_flat_slice_mut};
use super::rows::SimdRows;
use super::Simd;

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
