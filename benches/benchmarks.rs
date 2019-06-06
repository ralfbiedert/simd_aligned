#![feature(test)]

use criterion::{criterion_group, criterion_main, Criterion};

use packed_simd::*;
use simd_aligned::*;
use criterion::Benchmark;

fn criterion_benchmark(c: &mut Criterion) {
    
    c.bench("mode",
            Benchmark::new("scalar", |b| {
                let x = vec![6; 1024];
                let y = vec![4; 1024];
    
                b.iter(|| {
                    let mut z = vec![0; 1024];
        
                    for (i, e) in z.iter_mut().enumerate() {
                        *e = x[i] + y[i]
                    }
                });
            })
                .with_function("packed", |b| {
                    let x = vec![u8x16::splat(6); 64];
                    let y = vec![u8x16::splat(4); 64];
    
                    b.iter(|| {
                        let mut z = vec![u8x16::splat(0); 64];
        
                        for (i, e) in z.iter_mut().enumerate() {
                            *e = x[i] + y[i]
                        }
                    });
                })
                .with_function("simd_aligned", |b| {
                    let x = VecN::<u8x16>::with(6_u8, 1024);
                    let y = VecN::<u8x16>::with(4_u8, 1024);
    
                    b.iter(|| {
                        let mut z = vec![u8x16::splat(0); 64];
        
                        for (i, e) in z.iter_mut().enumerate() {
                            *e = x[i] + y[i]
                        }
                    });
                })
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
