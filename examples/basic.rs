use simd_aligned::*;

fn add() {
    let mut v1 = SimdVector::<f64s>::with_size(10);
    let mut v2 = SimdVector::<f64s>::with_size(10);

    let v1_m = v1.flat_mut();
    let v2_m = v2.flat_mut();

    // Set some elements on v1
    v1_m[0] = 0.0;
    v1_m[4] = 4.0;
    v1_m[8] = 8.0;

    // Set some others on v2
    v2_m[1] = 0.0;
    v2_m[5] = 5.0;
    v2_m[9] = 9.0;

    // for x in v1.iter() {}

    let mut result = SimdVector::<f64s>::with_size(10);

    result[0] = v1[0] + v2[0];

    // for (x, y) in v1.iter().zip(v2.iter()) {
    //     sum += *x + *y;
    // }
}

fn vector() {
    // Create a vector of f64x__ elements that, in total, will hold space
    // for at least 4 f64 values. Internally this might be one f64x4, two f64x2,
    // or one f64x8 where the 2nd half is hidden, depending on the current architecture.
    let mut v = SimdVector::<f64s>::with_size(8);

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
    add();
    vector();
    matrix();
}
