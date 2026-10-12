use std::marker::PhantomData;

use super::{Decomposition, dims_array};
use crate::blockwise::Padded;
use crate::bytes::Reader;
use crate::config::Config;
use crate::error::{Result, corrupt};
use crate::predictor::{LorenzoPredictor, Predictor};
use crate::quantizer::Quantizer;
use crate::types::SzFloat;

/// Splits the data into blocks and predicts each with `P`, falling back to first-order Lorenzo
/// for blocks `P` declines.
pub struct BlockwiseDecomposition<T: SzFloat, const N: usize, P, Q> {
    predictor: P,
    quantizer: Q,
    fallback: LorenzoPredictor<T, N, 1>,
    padding: usize,
    _t: PhantomData<T>,
}

impl<T: SzFloat, const N: usize, P: Predictor<T, N>, Q: Quantizer<T>> BlockwiseDecomposition<T, N, P, Q> {
    pub fn new(conf: &Config, predictor: P, quantizer: Q) -> Self {
        let fallback = LorenzoPredictor::new(conf.abs_error_bound);
        // The fallback reads neighbours the predictor may not pad for.
        let padding = predictor.padding().max(fallback.padding());
        BlockwiseDecomposition {
            predictor,
            quantizer,
            fallback,
            padding,
            _t: PhantomData,
        }
    }

    fn block_size(conf: &Config) -> Result<usize> {
        if conf.block_size <= 0 {
            return corrupt("blockwise: block size must be positive");
        }
        Ok(conf.block_size as usize)
    }
}

impl<T: SzFloat, const N: usize, P: Predictor<T, N>, Q: Quantizer<T>> Decomposition<T>
    for BlockwiseDecomposition<T, N, P, Q>
{
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>> {
        let dims = dims_array::<N>(conf)?;
        let mut padded = Padded::new(Some(&data[..conf.num]), dims, self.padding);
        let mut block = padded.block_iter(Self::block_size(conf)?);
        let mut bins = Vec::with_capacity(conf.num);
        loop {
            let use_main = self.predictor.precompress(&block, &padded.buf);
            let p: &mut dyn Predictor<T, N> = if use_main {
                &mut self.predictor
            } else {
                &mut self.fallback
            };
            p.precompress_block_commit();
            let p: &dyn Predictor<T, N> = p;
            let buf = &mut padded.buf;
            let q = &mut self.quantizer;
            block.for_each(|pos, idx| {
                let pred = p.predict(buf, pos, idx, &block.strides);
                bins.push(q.quantize_and_overwrite(&mut buf[pos], pred));
            });
            if !block.advance() {
                break;
            }
        }
        Ok(bins)
    }

    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()> {
        if bins.len() < conf.num {
            return corrupt("blockwise: fewer bins than the grid consumes");
        }
        let dims = dims_array::<N>(conf)?;
        let mut padded = Padded::<T, N>::new(None, dims, self.padding);
        let mut block = padded.block_iter(Self::block_size(conf)?);
        let mut k = 0;
        loop {
            let use_main = self.predictor.predecompress(&block)?;
            let p: &dyn Predictor<T, N> = if use_main { &self.predictor } else { &self.fallback };
            let buf = &mut padded.buf;
            let q = &mut self.quantizer;
            let mut err = Ok(());
            block.for_each(|pos, idx| {
                let pred = p.predict(buf, pos, idx, &block.strides);
                match q.recover(pred, bins[k]) {
                    Ok(v) => buf[pos] = v,
                    Err(e) => err = Err(e),
                }
                k += 1;
            });
            err?;
            if !block.advance() {
                break;
            }
        }
        padded.copy_out(&mut out[..conf.num]);
        Ok(())
    }

    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        self.fallback.save(out)?;
        self.predictor.save(out)?;
        self.quantizer.save(out);
        Ok(())
    }

    fn load(&mut self, r: &mut Reader, conf: &Config) -> Result<()> {
        self.fallback.load(r, conf.num)?;
        self.predictor.load(r, conf.num)?;
        self.quantizer.load(r)
    }

    fn out_range(&self) -> (i32, i32) {
        self.quantizer.out_range()
    }

    fn size_est(&self) -> usize {
        self.fallback.size_est() + self.predictor.size_est() + self.quantizer.size_est()
    }
}
