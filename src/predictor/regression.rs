use std::marker::PhantomData;

use super::Predictor;
use crate::blockwise::Block;
use crate::bytes::{ByteWriter, Reader};
use crate::encoder::{Encoder, HuffmanEncoder};
use crate::error::{Result, corrupt};
use crate::quantizer::{LinearQuantizer, Quantizer};
use crate::types::SzFloat;

/// Fits a linear function of the index to each block; its quantized coefficients are stored.
pub struct RegressionPredictor<T: SzFloat, const N: usize, E = HuffmanEncoder> {
    quantizer_independent: LinearQuantizer<T>,
    quantizer_liner: LinearQuantizer<T>,
    coeff_bins: Vec<i32>,
    coeff_index: usize,
    prev_coeffs: [T; 5],
    current_coeffs: [T; 5],
    _encoder: PhantomData<E>,
}

impl<T: SzFloat, const N: usize, E: Encoder + Default> RegressionPredictor<T, N, E> {
    pub fn new(block_size: usize, eb: f64) -> Self {
        RegressionPredictor {
            quantizer_independent: LinearQuantizer::new(eb / (N + 1) as f64, 32768),
            quantizer_liner: LinearQuantizer::new(eb / (N + 1) as f64 / block_size as f64, 32768),
            coeff_bins: Vec::new(),
            coeff_index: 0,
            prev_coeffs: [T::ZERO; 5],
            current_coeffs: [T::ZERO; 5],
            _encoder: PhantomData,
        }
    }
}

impl<T: SzFloat, const N: usize, E: Encoder + Default> Predictor<T, N> for RegressionPredictor<T, N, E> {
    fn precompress(&mut self, block: &Block<N>, data: &[T]) -> bool {
        let sizes = block.sizes();
        let mut dims = [0f64; N];
        let mut num_elements = 1f64;
        for i in 0..N {
            dims[i] = sizes[i] as f64;
            if dims[i] <= 1.0 {
                return false;
            }
            num_elements *= dims[i];
        }
        let mut sum = [0f64; 5];
        block.for_each(|pos, idx| {
            let c = data[pos];
            for i in 0..N {
                // size_t * T is computed in T, then added to the double sum.
                sum[i] += (T::from_usize(idx[i]) * c).to_f64();
            }
            sum[N] += c.to_f64();
        });
        self.current_coeffs = [T::ZERO; 5];
        self.current_coeffs[N] = T::from_f64(sum[N] / num_elements);
        for i in 0..N {
            self.current_coeffs[i] =
                T::from_f64((2.0 * sum[i] / (dims[i] - 1.0) - sum[N]) * 6.0 / num_elements / (dims[i] + 1.0));
            self.current_coeffs[N] =
                T::from_f64(self.current_coeffs[N].to_f64() - (dims[i] - 1.0) * self.current_coeffs[i].to_f64() / 2.0);
        }
        true
    }

    fn precompress_block_commit(&mut self) {
        for i in 0..N {
            let b = self
                .quantizer_liner
                .quantize_and_overwrite(&mut self.current_coeffs[i], self.prev_coeffs[i]);
            self.coeff_bins.push(b);
        }
        let b = self
            .quantizer_independent
            .quantize_and_overwrite(&mut self.current_coeffs[N], self.prev_coeffs[N]);
        self.coeff_bins.push(b);
        self.prev_coeffs = self.current_coeffs;
    }

    fn predecompress(&mut self, block: &Block<N>) -> Result<bool> {
        if block.sizes().iter().any(|&s| s <= 1) {
            return Ok(false);
        }
        if self.coeff_index + N + 1 > self.coeff_bins.len() {
            return corrupt("ran out of regression coefficients while decompressing");
        }
        for i in 0..N {
            let b = self.coeff_bins[self.coeff_index];
            self.coeff_index += 1;
            self.current_coeffs[i] = self.quantizer_liner.recover(self.current_coeffs[i], b)?;
        }
        let b = self.coeff_bins[self.coeff_index];
        self.coeff_index += 1;
        self.current_coeffs[N] = self.quantizer_independent.recover(self.current_coeffs[N], b)?;
        Ok(true)
    }

    #[inline(always)]
    fn predict(&self, _: &[T], _: usize, idx: &[usize; N], _: &[usize; N]) -> T {
        let c = &self.current_coeffs;
        let mut p = c[0] * T::from_usize(idx[0]);
        for i in 1..N {
            p = p + c[i] * T::from_usize(idx[i]);
        }
        p + c[N]
    }

    #[inline(always)]
    fn estimate_error(&self, d: &[T], pos: usize, idx: &[usize; N], ds: &[usize; N]) -> T {
        (d[pos] - self.predict(d, pos, idx, ds)).abs()
    }

    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        out.put_u64(self.coeff_bins.len() as u64);
        if !self.coeff_bins.is_empty() {
            self.quantizer_independent.save(out);
            self.quantizer_liner.save(out);
            let mut enc = E::default();
            let states = self
                .quantizer_independent
                .out_range()
                .1
                .max(self.quantizer_liner.out_range().1);
            enc.preprocess_encode(&self.coeff_bins, states);
            enc.save(out);
            enc.encode(&self.coeff_bins, out)?;
        }
        Ok(())
    }

    fn load(&mut self, r: &mut Reader, max_blocks: usize) -> Result<()> {
        let n = r.u64()?;
        if n > ((N + 1) as u64).saturating_mul(max_blocks as u64) {
            return corrupt("more regression coefficients than blocks");
        }
        if n > 0 {
            self.quantizer_independent.load(r)?;
            self.quantizer_liner.load(r)?;
            let mut enc = E::default();
            enc.load(r)?;
            let n = usize::try_from(n).or_else(|_| corrupt("coefficient count overflows"))?;
            self.coeff_bins = enc.decode(r, n)?;
            self.current_coeffs = [T::ZERO; 5];
            self.coeff_index = 0;
        }
        Ok(())
    }

    fn size_est(&self) -> usize {
        if self.coeff_bins.is_empty() {
            return 8;
        }
        8 + self.quantizer_independent.size_est()
            + self.quantizer_liner.size_est()
            + HuffmanEncoder::size_bound(self.coeff_bins.len(), 65537)
    }
}
