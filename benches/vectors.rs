#![feature(test)]

mod vectors {

    use packed_simd::*;
    use simd_aligned::*;
    use test::Bencher;

    #[bench]
    fn scalar(b: &mut Bencher) {
        let x = vec![6; 1024];
        let y = vec![4; 1024];
        let mut z = vec![0; 1024];

        b.iter(|| {
            for (i, e) in z.iter_mut().enumerate() {
                *e = x[i] + y[i]
            }
        });
    }

    #[bench]
    fn packed(b: &mut Bencher) {
        let x = vec![u8x16::splat(6); 64];
        let y = vec![u8x16::splat(4); 64];
        let mut z = vec![u8x16::splat(0); 64];

        b.iter(|| {
            for (i, e) in z.iter_mut().enumerate() {
                *e = x[i] + y[i]
            }
        });
    }

    #[bench]
    fn simd_aligned(b: &mut Bencher) {
        let x = SimdVector::<u8x16>::with(6_u8, 1024);
        let y = SimdVector::<u8x16>::with(4_u8, 1024);
        let mut z = SimdVector::<u8x16>::with(0_u8, 1024);

        b.iter(|| {
            for (i, e) in z.iter_mut().enumerate() {
                *e = x[i] + y[i]
            }
        });
    }

}
