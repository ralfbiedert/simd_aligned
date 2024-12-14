[![crates.io-badge]][crates.io-url]
[![docs.rs-badge]][docs.rs-url]
![license-badge]
[![rust-version-badge]][rust-version-url]
[![rust-build-badge]][rust-build-url]

## In One Sentence

You want to use safe SIMD datatypes from [`wide`](https://crates.io/crates/wide/) but realized there is no simple, safe and fast way to align your `f32x4` (and friends) in memory _and_ treat them as regular `f32` slices for easy loading and manipulation; `simd_aligned` to the rescue.


## Highlights

* built on top of [`wide`](https://crates.io/crates/wide/) for easy data handling
* supports everything from `u8x16` to `f64x4`
* think in flat slices (`&[f32]`), but get performance of properly aligned SIMD vectors (`&[f32x4]`)
* provides N-dimensional [`VecSimd`](https://docs.rs/simd_aligned/latest/simd_aligned/struct.VecSimd.html) and NxM-dimensional [`MatSimd`](https://docs.rs/simd_aligned/latest/simd_aligned/struct.MatSimd.html).

## Examples

Produces a vector that can hold `10` elements of type `f64`. All elements are guaranteed to be properly aligned for fast access.

```rust
use simd_aligned::*;

// Create vectors of `10` f64 elements with value `0.0`.
let mut v1 = VecSimd::<f64x4>::with(0.0, 10);
let mut v2 = VecSimd::<f64x4>::with(0.0, 10);

// Get "flat", mutable view of the vector, and set individual elements:
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

let mut sum = f64x4::splat(0.0);

// Eventually, do something with the actual SIMD types. Does
// `std::simd` vector math, e.g., f64x8 + f64x8 in one operation:
sum = v1[0] + v2[0];
```

## Benchmarks

There is no performance penalty for using `simd_aligned`, while retaining all the
simplicity of handling flat arrays.

```rust
test vectors::packed       ... bench:          77 ns/iter (+/- 4)
test vectors::scalar       ... bench:       1,177 ns/iter (+/- 464)
test vectors::simd_aligned ... bench:          71 ns/iter (+/- 5)
```

## Status

- December 2024: Compiles on stable.
- March 2023: Compiles again on latest Rust nightly.
- August 2018: Initial version.

## FAQ

#### How does it relate to [faster](https://github.com/AdamNiederer/faster) and [`std::simd`](https://github.com/rust-lang-nursery/packed_simd/)?

* `simd_aligned` builds on top of `std::simd`. At aims to provide common, SIMD-aligned
  data structure that support simple and safe scalar access patterns.

* `faster` (as of today) is good if you already have exiting flat slices in your code
  and want to operate them "full SIMD ahead". However, in particular when dealing with multiple
  slices at the same time (e.g., kernel computations) the performance impact of unaligned arrays can
  become a bit more noticeable (e.g., in the case of [ffsvm](https://github.com/ralfbiedert/ffsvm-rust/) up to 10% - 20%).

[crates.io-badge]: https://img.shields.io/crates/v/simd_aligned.svg
[crates.io-url]: https://crates.io/crates/simd_aligned
[license-badge]: https://img.shields.io/badge/license-BSD2-blue.svg
[docs.rs-badge]: https://docs.rs/simd_aligned/badge.svg
[docs.rs-url]: https://docs.rs/simd_aligned/
[rust-version-badge]: https://img.shields.io/badge/rust-1.83%2B-blue.svg?maxAge=3600
[rust-version-url]: https://github.com/ralfbiedert/simd_aligned
[rust-build-badge]: https://github.com/ralfbiedert/simd_aligned/actions/workflows/rust.yml/badge.svg
[rust-build-url]: https://github.com/ralfbiedert/simd_aligned/actions/workflows/rust.yml
