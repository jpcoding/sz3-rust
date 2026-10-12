//! Error-bounded quantization of prediction errors (SZ3's `QuantizerInterface`).

use crate::bytes::{ByteWriter, Reader};
use crate::error::{Result, corrupt};
use crate::types::SzFloat;

/// Maps a value and its prediction to an integer bin, overwriting the value with what the
/// decompressor will reconstruct.
pub trait Quantizer<T: SzFloat> {
    /// Quantizes `data - pred`; `data` becomes the reconstructed value. Bin 0 means the value was
    /// stored losslessly ("unpredictable").
    fn quantize_and_overwrite(&mut self, data: &mut T, pred: T) -> i32;
    /// Reconstructs a value from its prediction and bin.
    fn recover(&mut self, pred: T, bin: i32) -> Result<T>;
    /// Stores a value losslessly; returns its bin (0).
    fn force_save_unpred(&mut self, v: T) -> i32;
    /// Next losslessly stored value.
    fn recover_unpred(&mut self) -> Result<T>;
    fn eb(&self) -> f64;
    fn set_eb(&mut self, eb: f64);
    /// Bins lie in `[0, out_range().1)`.
    fn out_range(&self) -> (i32, i32);
    fn save(&self, out: &mut Vec<u8>);
    fn load(&mut self, r: &mut Reader) -> Result<()>;
    fn size_est(&self) -> usize;
    /// For a linear quantizer: its parameters, unpredictable values and read position, for
    /// specialized loops.
    fn linear_kernel(&mut self) -> Option<(LinearKernel, &mut Vec<T>, &mut usize)> {
        None
    }
}

/// The arithmetic of [`LinearQuantizer`] with its parameters held by value.
#[derive(Clone, Copy, Debug)]
pub struct LinearKernel {
    pub error_bound: f64,
    pub error_bound_reciprocal: f64,
    pub radius: i32,
}

impl LinearKernel {
    #[inline(always)]
    pub fn recover_pred<T: SzFloat>(&self, pred: T, bin: i32) -> T {
        // In C++: pred + 2 * int64(bin - radius) * double(eb), in double, rounded to T.
        let delta = (2 * (bin as i64 - self.radius as i64)) as f64 * self.error_bound;
        T::from_f64(pred.to_f64() + delta)
    }

    /// Quantizes `data - pred`; `data` becomes the reconstructed value, or is stored in `unpred`
    /// and gets bin 0.
    #[inline(always)]
    pub fn quantize<T: SzFloat>(&self, data: &mut T, pred: T, unpred: &mut Vec<T>) -> i32 {
        let diff = *data - pred;
        // NaN fails the comparison and infinities exceed it: both are stored losslessly.
        let scaled = diff.abs().to_f64() * self.error_bound_reciprocal;
        if scaled < (self.radius as i64 * 2 - 1) as f64 {
            let half = ((scaled as i64 + 1) >> 1) as i32;
            let bin = if diff < T::ZERO {
                self.radius - half
            } else {
                self.radius + half
            };
            let dec = self.recover_pred(pred, bin);
            if (dec.to_f64() - data.to_f64()).abs() <= self.error_bound {
                *data = dec;
                return bin;
            }
        }
        unpred.push(*data);
        0
    }

    #[inline(always)]
    pub fn recover<T: SzFloat>(&self, pred: T, bin: i32, unpred: &[T], index: &mut usize) -> Result<T> {
        if bin != 0 {
            return Ok(self.recover_pred(pred, bin));
        }
        match unpred.get(*index) {
            Some(&v) => {
                *index += 1;
                Ok(v)
            }
            None => corrupt("ran out of unpredictable values while decompressing"),
        }
    }
}

/// Uniform quantizer with `2 * radius` bins of width `2 * eb` (`LinearQuantizer`).
#[derive(Clone, Debug)]
pub struct LinearQuantizer<T> {
    unpred: Vec<T>,
    index: usize,
    error_bound: f64,
    error_bound_reciprocal: f64,
    radius: i32,
}

const UID: u8 = 0b10;

impl<T: SzFloat> Default for LinearQuantizer<T> {
    fn default() -> Self {
        LinearQuantizer {
            unpred: Vec::new(),
            index: 0,
            error_bound: 1.0,
            error_bound_reciprocal: 1.0,
            radius: 32768,
        }
    }
}

impl<T: SzFloat> LinearQuantizer<T> {
    pub fn new(eb: f64, radius: i32) -> Self {
        LinearQuantizer {
            unpred: Vec::new(),
            index: 0,
            error_bound: eb,
            error_bound_reciprocal: 1.0 / eb,
            radius,
        }
    }

    #[inline(always)]
    fn kernel(&self) -> LinearKernel {
        LinearKernel {
            error_bound: self.error_bound,
            error_bound_reciprocal: self.error_bound_reciprocal,
            radius: self.radius,
        }
    }

    pub fn unpred_count(&self) -> usize {
        self.unpred.len()
    }
}

impl<T: SzFloat> Quantizer<T> for LinearQuantizer<T> {
    #[inline(always)]
    fn quantize_and_overwrite(&mut self, data: &mut T, pred: T) -> i32 {
        self.kernel().quantize(data, pred, &mut self.unpred)
    }

    #[inline(always)]
    fn recover(&mut self, pred: T, bin: i32) -> Result<T> {
        self.kernel().recover(pred, bin, &self.unpred, &mut self.index)
    }

    fn force_save_unpred(&mut self, v: T) -> i32 {
        self.unpred.push(v);
        0
    }

    #[inline(always)]
    fn recover_unpred(&mut self) -> Result<T> {
        match self.unpred.get(self.index) {
            Some(&v) => {
                self.index += 1;
                Ok(v)
            }
            None => corrupt("ran out of unpredictable values while decompressing"),
        }
    }

    fn eb(&self) -> f64 {
        self.error_bound
    }

    fn set_eb(&mut self, eb: f64) {
        self.error_bound = eb;
        self.error_bound_reciprocal = 1.0 / eb;
    }

    fn out_range(&self) -> (i32, i32) {
        (0, self.radius.wrapping_mul(2))
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.put_u8(UID);
        out.put_f64(self.error_bound);
        out.put_i32(self.radius);
        out.put_u64(self.unpred.len() as u64);
        out.put_slice(&self.unpred);
    }

    fn load(&mut self, r: &mut Reader) -> Result<()> {
        if r.u8()? != UID {
            return corrupt("LinearQuantizer uid mismatch");
        }
        self.error_bound = r.f64()?;
        self.error_bound_reciprocal = 1.0 / self.error_bound;
        self.radius = r.i32()?;
        let n = r.u64()?;
        self.unpred = r.vec(n)?;
        self.index = 0;
        Ok(())
    }

    fn size_est(&self) -> usize {
        1 + 8 + 4 + 8 + self.unpred.len() * T::SIZE
    }

    fn linear_kernel(&mut self) -> Option<(LinearKernel, &mut Vec<T>, &mut usize)> {
        Some((self.kernel(), &mut self.unpred, &mut self.index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_error() {
        let eb = 1e-3;
        let mut q = LinearQuantizer::<f32>::new(eb, 32768);
        let vals = [0.0f32, 1.0, -1.0, 0.0004, 1e10, f32::NAN, f32::INFINITY];
        let mut bins = Vec::new();
        let mut rec = Vec::new();
        for &v in &vals {
            let mut d = v;
            bins.push(q.quantize_and_overwrite(&mut d, 0.5));
            rec.push(d);
        }
        let mut buf = Vec::new();
        q.save(&mut buf);
        let mut d = LinearQuantizer::<f32>::default();
        d.load(&mut Reader::new(&buf)).unwrap();
        for (i, &b) in bins.iter().enumerate() {
            let r = d.recover(0.5, b).unwrap();
            assert!(r.to_bits() == rec[i].to_bits() || (r.is_nan() && rec[i].is_nan()));
            if vals[i].is_finite() {
                assert!(((r as f64) - vals[i] as f64).abs() <= eb);
            }
        }
    }
}
