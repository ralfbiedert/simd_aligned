use simd_aligned::*;

fn main() {
    // let array = SimdRows::<f32s>::with_dimension(10, 5);

    // let a = SimdArray::<f32s>::with_size(10);
    let mut v = SimdVector::<f32s>::with_size(6);
    v.flat_mut()
        .clone_from_slice(&[0_f32, 1_f32, 2_f32, 3_f32, 4_f32, 5_f32]);

    let mut m = SimdMatrix::<f32s, RowOptimized>::with_dimension(10, 5);
    m.row(0);
    m.row_mut(0);

    m.row_as_flat(0);
    m.row_as_flat_mut(0);

    m.column_as_flat(0);

    let mut mm = m.flat_mut();
    mm[(1, 4)] = 0.4;

    // v.as_flat();
    // v.as_flat_mut();

    // v[]
}
