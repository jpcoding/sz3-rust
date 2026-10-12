//! Multilevel spline interpolation (SZ3's `InterpolationDecomposition`, ICDE'21 / SIGMOD'24).
//!
//! Level by level, from coarse to fine, every point midway between known points is predicted by
//! linear or cubic interpolation along one dimension at a time, then quantized.

use std::marker::PhantomData;

use super::{Decomposition, dims_array};
use crate::bytes::{ByteWriter, Reader};
use crate::config::Config;
use crate::error::{Result, corrupt};
use crate::quantizer::Quantizer;
use crate::types::SzFloat;

// Interpolation kernels, evaluated in the order and precision of SZ3's Interpolators.hpp.
#[inline(always)]
fn interp_linear<T: SzFloat>(a: T, b: T) -> T {
    (a + b) / T::c(2)
}
#[inline(always)]
fn interp_linear1<T: SzFloat>(a: T, b: T) -> T {
    T::from_f64(-0.5 * a.to_f64() + 1.5 * b.to_f64())
}
#[inline(always)]
fn interp_quad_1<T: SzFloat>(a: T, b: T, c: T) -> T {
    (T::c(3) * a + T::c(6) * b - c) / T::c(8)
}
#[inline(always)]
fn interp_quad_2<T: SzFloat>(a: T, b: T, c: T) -> T {
    (-a + T::c(6) * b + T::c(3) * c) / T::c(8)
}
#[inline(always)]
fn interp_quad_3<T: SzFloat>(a: T, b: T, c: T) -> T {
    (T::c(3) * a - T::c(10) * b + T::c(15) * c) / T::c(8)
}
#[inline(always)]
fn interp_cubic<T: SzFloat>(a: T, b: T, c: T, d: T) -> T {
    (-a + T::c(9) * b + T::c(9) * c - d) / T::c(16)
}

/// The grid geometry an interpolation pass walks.
struct Grid<const N: usize> {
    dims: [usize; N],
    offsets: [usize; N],
    sequences: Vec<[usize; N]>,
}

/// Calls `f` on `offset + sum(i_k * dim_offsets[k])` for `i_k` in `begins[k]..ends[k]` stepping
/// `strides[k]`, row-major (SZ3's `foreach`).
#[inline(always)]
fn foreach<const N: usize>(
    offset: usize,
    begins: &[usize; N],
    ends: &[usize; N],
    strides: &[usize; N],
    dim_offsets: &[usize; N],
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    if (0..N).any(|k| begins[k] >= ends[k]) {
        return Ok(());
    }
    let mut idx = *begins;
    loop {
        let mut base = offset;
        for k in 0..N - 1 {
            base += idx[k] * dim_offsets[k];
        }
        let mut i = begins[N - 1];
        while i < ends[N - 1] {
            f(base + i * dim_offsets[N - 1])?;
            i += strides[N - 1];
        }
        let mut d = N - 1;
        loop {
            if d == 0 {
                return Ok(());
            }
            d -= 1;
            idx[d] += strides[d];
            if idx[d] < ends[d] {
                break;
            }
            idx[d] = begins[d];
        }
    }
}

impl<const N: usize> Grid<N> {
    /// Interpolates along one dimension, the original (ICDE'21) scheme.
    fn interpolation_1d<T: SzFloat, F: FnMut(&mut [T], usize, T) -> Result<()>>(
        &self,
        data: &mut [T],
        begin: usize,
        end: usize,
        stride: usize,
        cubic: bool,
        f: &mut F,
    ) -> Result<()> {
        let n = (end - begin) / stride + 1;
        if n <= 1 {
            return Ok(());
        }
        let s3 = 3 * stride;
        let s5 = 5 * stride;
        if !cubic || n < 5 {
            let mut i = 1;
            while i + 1 < n {
                let d = begin + i * stride;
                let p = interp_linear(data[d - stride], data[d + stride]);
                f(data, d, p)?;
                i += 2;
            }
            if n.is_multiple_of(2) {
                let d = begin + (n - 1) * stride;
                let p = if n < 4 {
                    data[d - stride]
                } else {
                    interp_linear1(data[d - s3], data[d - stride])
                };
                f(data, d, p)?;
            }
        } else {
            let mut i = 3;
            while i + 3 < n {
                let d = begin + i * stride;
                let p = interp_cubic(data[d - s3], data[d - stride], data[d + stride], data[d + s3]);
                f(data, d, p)?;
                i += 2;
            }
            let d = begin + stride;
            let p = interp_quad_1(data[d - stride], data[d + stride], data[d + s3]);
            f(data, d, p)?;

            let d = begin + i * stride;
            let p = interp_quad_2(data[d - s3], data[d - stride], data[d + stride]);
            f(data, d, p)?;
            if n.is_multiple_of(2) {
                let d = begin + (n - 1) * stride;
                let p = interp_quad_3(data[d - s5], data[d - s3], data[d - stride]);
                f(data, d, p)?;
            }
        }
        Ok(())
    }

    /// All interpolations along `direction` over a sub-grid, moving along the fastest
    /// dimension (SIGMOD'24).
    #[allow(clippy::too_many_arguments)]
    fn interpolation_1d_fastest_dim_first<T: SzFloat, F: FnMut(&mut [T], usize, T) -> Result<()>>(
        &self,
        data: &mut [T],
        begin_idx: &[usize; N],
        end_idx: &[usize; N],
        direction: usize,
        strides: &mut [usize; N],
        math_stride: usize,
        cubic: bool,
        f: &mut F,
    ) -> Result<()> {
        if (0..N).any(|i| end_idx[i] < begin_idx[i]) {
            return Ok(());
        }
        let n = (end_idx[direction] - begin_idx[direction]) / math_stride + 1;
        if n <= 1 {
            return Ok(());
        }
        let stride = math_stride * self.offsets[direction];
        let mut begins = [0usize; N];
        let mut ends = [0usize; N];
        let mut dim_offsets = self.offsets;
        let mut offset = 0;
        for i in 0..N {
            ends[i] = end_idx[i] - begin_idx[i] + 1;
            offset += self.offsets[i] * begin_idx[i];
        }
        dim_offsets[direction] = stride;
        let s2 = 2 * stride;
        if !cubic {
            begins[direction] = 1;
            ends[direction] = n - 1;
            strides[direction] = 2;
            foreach(offset, &begins, &ends, strides, &dim_offsets, |d| {
                let p = interp_linear(data[d - stride], data[d + stride]);
                f(data, d, p)
            })?;
            if n.is_multiple_of(2) {
                begins[direction] = n - 1;
                ends[direction] = n;
                foreach(offset, &begins, &ends, strides, &dim_offsets, |d| {
                    let p = if n < 3 {
                        data[d - stride]
                    } else {
                        interp_linear1(data[d - s2], data[d - stride])
                    };
                    f(data, d, p)
                })?;
            }
        } else {
            let s3 = 3 * stride;
            begins[direction] = 3;
            ends[direction] = n.saturating_sub(3);
            strides[direction] = 2;
            foreach(offset, &begins, &ends, strides, &dim_offsets, |d| {
                let p = interp_cubic(data[d - s3], data[d - stride], data[d + stride], data[d + s3]);
                f(data, d, p)
            })?;
            let mut boundaries = vec![1usize];
            if n % 2 == 1 && n > 3 {
                boundaries.push(n - 2);
            }
            if n.is_multiple_of(2) && n > 4 {
                boundaries.push(n - 3);
            }
            if n.is_multiple_of(2) && n > 2 {
                boundaries.push(n - 1);
            }
            for b in boundaries {
                begins[direction] = b;
                ends[direction] = b + 1;
                foreach(offset, &begins, &ends, strides, &dim_offsets, |d| {
                    let p = if b >= 3 {
                        if b + 3 < n {
                            interp_cubic(data[d - s3], data[d - stride], data[d + stride], data[d + s3])
                        } else if b + 1 < n {
                            interp_quad_2(data[d - s3], data[d - stride], data[d + stride])
                        } else {
                            interp_linear1(data[d - s3], data[d - stride])
                        }
                    } else if b + 3 < n {
                        interp_quad_1(data[d - stride], data[d + stride], data[d + s3])
                    } else if b + 1 < n {
                        interp_linear(data[d - stride], data[d + stride])
                    } else {
                        data[d - stride]
                    };
                    f(data, d, p)
                })?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn interpolation<T: SzFloat, F: FnMut(&mut [T], usize, T) -> Result<()>>(
        &self,
        data: &mut [T],
        begin: [usize; N],
        end: [usize; N],
        cubic: bool,
        f: &mut F,
        direction: usize,
        stride: usize,
    ) -> Result<()> {
        let off = &self.offsets;
        if N == 1 {
            return self.interpolation_1d(data, begin[0], end[0], stride, cubic, f);
        }
        let dims = self.sequences[direction];
        let s2 = stride * 2;
        if N == 2 {
            let (d0, d1) = (dims[0], dims[1]);
            let mut j = if begin[d1] != 0 { begin[d1] + s2 } else { 0 };
            while j <= end[d1] {
                let bo = begin[d0] * off[d0] + j * off[d1];
                self.interpolation_1d(
                    data,
                    bo,
                    bo + (end[d0] - begin[d0]) * off[d0],
                    stride * off[d0],
                    cubic,
                    f,
                )?;
                j += s2;
            }
            let mut i = if begin[d0] != 0 { begin[d0] + stride } else { 0 };
            while i <= end[d0] {
                let bo = i * off[d0] + begin[d1] * off[d1];
                self.interpolation_1d(
                    data,
                    bo,
                    bo + (end[d1] - begin[d1]) * off[d1],
                    stride * off[d1],
                    cubic,
                    f,
                )?;
                i += stride;
            }
            return Ok(());
        }
        let mut strides = [0usize; N];
        let mut begin_idx = begin;
        let end_idx = end;
        strides[dims[0]] = 1;
        for i in 1..N {
            begin_idx[dims[i]] = if begin[dims[i]] != 0 { begin[dims[i]] + s2 } else { 0 };
            strides[dims[i]] = s2;
        }
        self.interpolation_1d_fastest_dim_first(data, &begin_idx, &end_idx, dims[0], &mut strides, stride, cubic, f)?;
        for i in 1..N {
            begin_idx[dims[i]] = begin[dims[i]];
            begin_idx[dims[i - 1]] = if begin[dims[i - 1]] != 0 {
                begin[dims[i - 1]] + stride
            } else {
                0
            };
            strides[dims[i - 1]] = stride;
            self.interpolation_1d_fastest_dim_first(
                data,
                &begin_idx,
                &end_idx,
                dims[i],
                &mut strides,
                stride,
                cubic,
                f,
            )?;
        }
        Ok(())
    }
}

/// All permutations of `0..N` in lexicographic order (as `std::next_permutation` produces them).
fn permutations<const N: usize>() -> Vec<[usize; N]> {
    let mut seq = [0usize; N];
    for (i, s) in seq.iter_mut().enumerate() {
        *s = i;
    }
    let mut out = vec![seq];
    loop {
        // next_permutation
        let Some(i) = (0..N.saturating_sub(1)).rev().find(|&i| seq[i] < seq[i + 1]) else {
            return out;
        };
        let j = (i + 1..N).rev().find(|&j| seq[j] > seq[i]).unwrap();
        seq.swap(i, j);
        seq[i + 1..].reverse();
        out.push(seq);
    }
}

pub struct InterpolationDecomposition<T: SzFloat, const N: usize, Q> {
    quantizer: Q,
    interp_level: i32,
    interp_id: i32,
    blocksize: u32,
    num_elements: usize,
    dims: [usize; N],
    direction_sequence_id: i32,
    anchor_stride: usize,
    eb_alpha: f64,
    eb_beta: f64,
    _t: PhantomData<T>,
}

const EB_RATIO: f64 = 0.5;

impl<T: SzFloat, const N: usize, Q: Quantizer<T>> InterpolationDecomposition<T, N, Q> {
    pub fn new(quantizer: Q) -> Self {
        InterpolationDecomposition {
            quantizer,
            interp_level: -1,
            interp_id: 0,
            blocksize: 32,
            num_elements: 0,
            dims: [0; N],
            direction_sequence_id: 0,
            anchor_stride: 0,
            eb_alpha: -1.0,
            eb_beta: -1.0,
            _t: PhantomData,
        }
    }

    fn init(&mut self) -> Result<Grid<N>> {
        if self.blocksize == 0 || !self.blocksize.is_multiple_of(2) {
            return corrupt("interpolation: block size must be a positive even number");
        }
        if !(0..2).contains(&self.interp_id) {
            return corrupt("interpolation: interpolator id is out of range");
        }
        self.num_elements = 1;
        self.interp_level = -1;
        let mut use_anchor = false;
        for &d in &self.dims {
            let l = (d as f64).log2().ceil();
            if (self.interp_level as f64) < l {
                self.interp_level = l as i32;
            }
            if d > self.anchor_stride {
                use_anchor = true;
            }
            self.num_elements = self
                .num_elements
                .checked_mul(d)
                .ok_or_else(|| crate::error::Error::Corrupt("interpolation: dimensions overflow".into()))?;
        }
        if !use_anchor {
            self.anchor_stride = 0;
        }
        if self.anchor_stride > 0 && !self.anchor_stride.is_power_of_two() {
            return corrupt("interpolation: anchor stride must be a power of two");
        }
        if self.anchor_stride > 0 {
            let max_level = (self.anchor_stride as f64).log2() as i32 + 1;
            if max_level <= self.interp_level {
                self.interp_level = max_level;
            }
        }
        if self.interp_level > 63 {
            return corrupt("interpolation: too many levels");
        }
        let mut offsets = [0usize; N];
        offsets[N - 1] = 1;
        for i in (0..N - 1).rev() {
            offsets[i] = offsets[i + 1] * self.dims[i + 1];
        }
        let sequences = permutations::<N>();
        if self.direction_sequence_id < 0 || self.direction_sequence_id as usize >= sequences.len() {
            return corrupt("interpolation: direction sequence id is out of range");
        }
        Ok(Grid {
            dims: self.dims,
            offsets,
            sequences,
        })
    }

    /// Error bound for a level, or `None` to keep the current one.
    fn level_eb(&self, eb: f64, level: i32) -> Option<f64> {
        if self.eb_alpha < 0.0 {
            Some(if level >= 3 { eb * EB_RATIO } else { eb })
        } else if self.eb_alpha >= 1.0 {
            // pow(double, int) in C++ is pow(double, double).
            let mut r = self.eb_alpha.powf((level - 1) as f64);
            if r > self.eb_beta {
                r = self.eb_beta;
            }
            Some(eb / r)
        } else {
            None
        }
    }

    /// Visits the anchor grid, then every level, calling `f` on each interpolated point.
    fn walk(
        &mut self,
        grid: &Grid<N>,
        data: &mut [T],
        eb: f64,
        anchor: &mut impl FnMut(&mut Q, &mut [T], usize) -> Result<()>,
        point: &mut impl FnMut(&mut Q, &mut [T], usize, T) -> Result<()>,
    ) -> Result<()> {
        if self.anchor_stride == 0 {
            point(&mut self.quantizer, data, 0, T::ZERO)?;
        } else {
            let begins = [0usize; N];
            let strides = [self.anchor_stride; N];
            let q = &mut self.quantizer;
            foreach(0, &begins, &grid.dims, &strides, &grid.offsets, |d| anchor(q, data, d))?;
            self.interp_level -= 1;
        }
        let cubic = self.interp_id == 1;
        let direction = self.direction_sequence_id as usize;
        let mut level = self.interp_level;
        while level > 0 && level <= self.interp_level {
            if let Some(e) = self.level_eb(eb, level) {
                self.quantizer.set_eb(e);
            }
            let stride = 1usize << (level - 1);
            let ibs = self.blocksize as usize * stride;
            let mut counts = [0usize; N];
            for i in 0..N {
                counts[i] = (self.dims[i] - 1) / ibs + 1;
            }
            let mut k = [0usize; N];
            let q = &mut self.quantizer;
            let mut visit = |data: &mut [T], d: usize, p: T| point(q, data, d, p);
            'blocks: loop {
                let mut start = [0usize; N];
                let mut end = [0usize; N];
                for i in 0..N {
                    start[i] = k[i] * ibs;
                    end[i] = (start[i] + ibs).min(self.dims[i] - 1);
                }
                grid.interpolation(data, start, end, cubic, &mut visit, direction, stride)?;
                let mut d = N;
                loop {
                    if d == 0 {
                        break 'blocks;
                    }
                    d -= 1;
                    k[d] += 1;
                    if k[d] < counts[d] {
                        break;
                    }
                    k[d] = 0;
                }
            }
            level -= 1;
        }
        Ok(())
    }
}

impl<T: SzFloat, const N: usize, Q: Quantizer<T>> Decomposition<T> for InterpolationDecomposition<T, N, Q> {
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>> {
        self.dims = dims_array::<N>(conf)?;
        self.interp_id = conf.interp_algo as i32;
        self.direction_sequence_id = conf.interp_direction as i32;
        self.anchor_stride = conf.interp_anchor_stride as i64 as usize;
        self.blocksize = 32;
        self.eb_alpha = conf.interp_alpha;
        self.eb_beta = conf.interp_beta;
        let grid = self.init()?;
        let mut bins = vec![0i32; self.num_elements];
        let eb = self.quantizer.eb();
        // Anchors are stored losslessly and keep bin 0, so only the index advances for them.
        let idx = std::cell::Cell::new(0usize);
        let mut anchor = |q: &mut Q, data: &mut [T], d: usize| {
            q.force_save_unpred(data[d]);
            idx.set(idx.get() + 1);
            Ok(())
        };
        let mut point = |q: &mut Q, data: &mut [T], d: usize, p: T| {
            let i = idx.get();
            bins[i] = q.quantize_and_overwrite(&mut data[d], p);
            idx.set(i + 1);
            Ok(())
        };
        self.walk(&grid, &mut data[..conf.num], eb, &mut anchor, &mut point)?;
        self.quantizer.set_eb(eb);
        Ok(bins)
    }

    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()> {
        // load() read the dimensions from the payload; the walk must stay inside conf's data.
        if conf.dims.len() != N || conf.dims[..] != self.dims[..] {
            return corrupt("interpolation: stored dimensions do not match the configuration");
        }
        let grid = self.init()?;
        if bins.len() < self.num_elements {
            return corrupt("interpolation: fewer bins than the grid consumes");
        }
        let eb = self.quantizer.eb();
        let idx = std::cell::Cell::new(0usize);
        let mut anchor = |q: &mut Q, data: &mut [T], d: usize| {
            data[d] = q.recover_unpred()?;
            idx.set(idx.get() + 1);
            Ok(())
        };
        let mut point = |q: &mut Q, data: &mut [T], d: usize, p: T| {
            let i = idx.get();
            let Some(&b) = bins.get(i) else {
                return corrupt("interpolation: ran out of bins");
            };
            data[d] = q.recover(p, b)?;
            idx.set(i + 1);
            Ok(())
        };
        self.walk(&grid, &mut out[..conf.num], eb, &mut anchor, &mut point)
    }

    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        for &d in &self.dims {
            out.put_u64(d as u64);
        }
        out.put_u32(self.blocksize);
        out.put_i32(self.interp_id);
        out.put_i32(self.direction_sequence_id);
        out.put_u64(self.anchor_stride as u64);
        out.put_f64(self.eb_alpha);
        out.put_f64(self.eb_beta);
        self.quantizer.save(out);
        Ok(())
    }

    fn load(&mut self, r: &mut Reader, _: &Config) -> Result<()> {
        for i in 0..N {
            self.dims[i] = usize::try_from(r.u64()?).or_else(|_| corrupt("dimension overflows"))?;
        }
        self.blocksize = r.u32()?;
        self.interp_id = r.i32()?;
        self.direction_sequence_id = r.i32()?;
        self.anchor_stride = usize::try_from(r.u64()?).or_else(|_| corrupt("anchor stride overflows"))?;
        self.eb_alpha = r.f64()?;
        self.eb_beta = r.f64()?;
        self.quantizer.load(r)
    }

    fn out_range(&self) -> (i32, i32) {
        self.quantizer.out_range()
    }

    fn size_est(&self) -> usize {
        N * 8 + 4 + 4 + 4 + self.quantizer.size_est() + 128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_order() {
        assert_eq!(
            permutations::<3>(),
            vec![[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        );
        assert_eq!(permutations::<1>(), vec![[0]]);
    }
}
