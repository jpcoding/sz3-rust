use super::Decomposition;
use crate::bytes::Reader;
use crate::config::Config;
use crate::error::{Result, corrupt};
use crate::quantizer::Quantizer;
use crate::types::SzFloat;

/// Quantizes every value against a prediction of zero.
pub struct NoPredictionDecomposition<Q> {
    quantizer: Q,
}

impl<Q> NoPredictionDecomposition<Q> {
    pub fn new(quantizer: Q) -> Self {
        NoPredictionDecomposition { quantizer }
    }
}

impl<T: SzFloat, Q: Quantizer<T>> Decomposition<T> for NoPredictionDecomposition<Q> {
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>> {
        Ok(data[..conf.num]
            .iter_mut()
            .map(|d| self.quantizer.quantize_and_overwrite(d, T::ZERO))
            .collect())
    }

    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()> {
        if bins.len() < conf.num {
            return corrupt("no-prediction: fewer bins than the data has elements");
        }
        for (o, &b) in out[..conf.num].iter_mut().zip(bins) {
            *o = self.quantizer.recover(T::ZERO, b)?;
        }
        Ok(())
    }

    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        self.quantizer.save(out);
        Ok(())
    }

    fn load(&mut self, r: &mut Reader, _: &Config) -> Result<()> {
        self.quantizer.load(r)
    }

    fn out_range(&self) -> (i32, i32) {
        self.quantizer.out_range()
    }

    fn size_est(&self) -> usize {
        self.quantizer.size_est() + 64
    }
}
