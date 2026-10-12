//! Padded N-d storage and the block iterator the blockwise decomposition walks
//! (SZ3's `block_data` / `block_iterator`).
//!
//! The data is copied into a buffer with `padding` zero layers before the start of every
//! dimension, so predictors can read neighbours at negative offsets without bounds logic.

use crate::types::SzFloat;

pub struct Padded<T, const N: usize> {
    pub buf: Vec<T>,
    /// Position of element (0, ..., 0).
    pub base: usize,
    pub dims: [usize; N],
    /// Strides of the padded layout.
    pub strides: [usize; N],
}

impl<T: SzFloat, const N: usize> Padded<T, N> {
    /// A zeroed padded buffer, filled from `data` when given.
    pub fn new(data: Option<&[T]>, dims: [usize; N], padding: usize) -> Self {
        let mut strides = [0; N];
        let mut s = 1;
        for i in (0..N).rev() {
            strides[i] = s;
            s *= dims[i] + padding;
        }
        let base = padding * strides.iter().sum::<usize>();
        let mut p = Padded {
            buf: vec![T::ZERO; s],
            base,
            dims,
            strides,
        };
        if let Some(data) = data {
            let rows = p.rows();
            let w = dims[N - 1];
            for (k, start) in rows.into_iter().enumerate() {
                p.buf[start..start + w].copy_from_slice(&data[k * w..(k + 1) * w]);
            }
        }
        p
    }

    /// Start of every innermost row, in row-major order.
    fn rows(&self) -> Vec<usize> {
        let nrows: usize = self.dims[..N - 1].iter().product();
        let mut out = Vec::with_capacity(nrows);
        let mut idx = [0usize; N];
        for _ in 0..nrows {
            out.push(self.base + (0..N - 1).map(|i| idx[i] * self.strides[i]).sum::<usize>());
            let mut d = N - 1;
            while d > 0 {
                d -= 1;
                idx[d] += 1;
                if idx[d] < self.dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    /// Copies the unpadded values out.
    pub fn copy_out(&self, out: &mut [T]) {
        let w = self.dims[N - 1];
        for (k, start) in self.rows().into_iter().enumerate() {
            out[k * w..(k + 1) * w].copy_from_slice(&self.buf[start..start + w]);
        }
    }

    pub fn block_iter(&self, block_size: usize) -> Block<N> {
        Block {
            offset: [0; N],
            block_size,
            dims: self.dims,
            strides: self.strides,
            base: self.base,
        }
    }
}

/// One block of a [`Padded`] grid; `next` moves to the following block in row-major order.
#[derive(Clone, Debug)]
pub struct Block<const N: usize> {
    pub offset: [usize; N],
    pub block_size: usize,
    pub dims: [usize; N],
    pub strides: [usize; N],
    pub base: usize,
}

impl<const N: usize> Block<N> {
    pub fn advance(&mut self) -> bool {
        let mut i = N - 1;
        self.offset[i] += self.block_size;
        while i > 0 && self.offset[i] >= self.dims[i] {
            self.offset[i] = 0;
            i -= 1;
            self.offset[i] += self.block_size;
        }
        self.offset[0] < self.dims[0]
    }

    /// Size of the block in each dimension.
    #[inline]
    pub fn sizes(&self) -> [usize; N] {
        let mut s = [0; N];
        for i in 0..N {
            s[i] = (self.offset[i] + self.block_size).min(self.dims[i]) - self.offset[i];
        }
        s
    }

    /// Buffer position of the element at `idx` within the block.
    #[inline]
    pub fn pos(&self, idx: &[usize; N]) -> usize {
        let mut p = self.base;
        for i in 0..N {
            p += (idx[i] + self.offset[i]) * self.strides[i];
        }
        p
    }

    /// Calls `f(position, index_in_block)` for every element, row-major.
    #[inline]
    pub fn for_each(&self, mut f: impl FnMut(usize, &[usize; N])) {
        let sizes = self.sizes();
        let mut idx = [0usize; N];
        loop {
            idx[N - 1] = 0;
            let row = self.pos(&idx);
            for k in 0..sizes[N - 1] {
                idx[N - 1] = k;
                f(row + k, &idx);
            }
            let mut d = N - 1;
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < sizes[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Calls `f` on the sample points SZ3 uses to estimate predictor errors: the block's
    /// diagonals.
    pub fn for_each_sampling(&self, mut f: impl FnMut(usize, &[usize; N])) {
        let m = self.sizes().into_iter().min().unwrap_or(0);
        if N == 1 {
            let mut idx = [0usize; N];
            f(self.pos(&idx), &idx);
            idx[0] = m - 1;
            f(self.pos(&idx), &idx);
            return;
        }
        for i in 0..m {
            let j = m - 1 - i;
            for mask in 0..(1usize << (N - 1)) {
                let mut idx = [i; N];
                for k in 1..N {
                    if (mask >> (N - 1 - k)) & 1 == 1 {
                        idx[k] = j;
                    }
                }
                f(self.pos(&idx), &idx);
            }
        }
    }
}
