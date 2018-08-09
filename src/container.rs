use crate::traits::Simd;

crate trait Container<T>
where
    T: Simd + Clone,
{
    fn with(default: T, size: usize) -> Self;

    fn slice(&self) -> &[T];

    fn slice_mut(&mut self) -> &mut [T];
}

impl<T> Container<T> for Vec<T>
where
    T: Simd + Clone,
{
    #[inline]
    fn with(default: T, size: usize) -> Self {
        vec![default; size]
    }

    #[inline(always)]
    fn slice(&self) -> &[T] {
        self
    }

    #[inline(always)]
    fn slice_mut(&mut self) -> &mut [T] {
        self
    }
}

// ---- Below here are some tests to understand how we could generalize stack and heap.

impl<T> Container<T> for [T; 4]
where
    T: Simd + Clone + Copy,
{
    fn with(default: T, _size: usize) -> Self {
        [default; 4]
    }

    fn slice(&self) -> &[T] {
        self
    }

    fn slice_mut(&mut self) -> &mut [T] {
        self
    }
}
