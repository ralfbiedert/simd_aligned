use simd_aligned::*;

fn vector() {
    // Create a vector of f64x__ elements that, in total, will hold space
    // for at least 4 f64 values. Internally this might be one f64x4, two f64x2,
    // or one f64x8 where the 2nd half is hidden, depending on the current architecture.
    let mut v = SimdVector::<f64s>::with_size(4);

    // Get a 'flat view' (&[f64]) into the SIMD vectors and fill it.
    v.flat_mut().clone_from_slice(&[0.0, 1.0, 2.0, 3.0]);
}

fn matrix() {
    // Create a matrix of height 10 and width 5, optimized for row access.
    // That means you will be able to access `m.row(4)`, and get a continuous view
    // of all `f32s` that constitute that row.
    let m = SimdMatrix::<f32s, RowOptimized>::with_dimension(10, 5);
    let _ = m.row(4);

    // Accessing the column does not work, as there is no continuous view in memory
    // m.column(3); --> panic!

    // However, you can always get a flat view of the matrix, for "normal-speed" query and update
    // of all elements:
    let m_flat = m.flat();
    let _ = m_flat[(3, 4)];
}

fn main() {
    vector();
    matrix();
}
