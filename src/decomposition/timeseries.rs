//! Time-series prediction for `(timesteps, points)` data (SZ3's `TimeSeriesDecomposition`): the
//! first timestep is predicted spatially (or from a given reference timestep), every later value
//! from the same point one timestep earlier.

use std::marker::PhantomData;

use super::Decomposition;
use crate::blockwise::Padded;
use crate::bytes::Reader;
use crate::config::Config;
use crate::error::{Error, Result, corrupt};
use crate::predictor::{LorenzoPredictor, Predictor};
use crate::quantizer::Quantizer;
use crate::types::SzFloat;

pub struct TimeSeriesDecomposition<T: SzFloat, P, Q> {
    predictor: P,
    fallback: LorenzoPredictor<T, 1, 1>,
    quantizer: Q,
    /// The reference for timestep 0; without one, timestep 0 is predicted spatially.
    ts0: Option<Vec<T>>,
    _t: PhantomData<T>,
}

impl<T: SzFloat, P: Predictor<T, 1>, Q: Quantizer<T>> TimeSeriesDecomposition<T, P, Q> {
    pub fn new(conf: &Config, predictor: P, quantizer: Q, ts0: Option<Vec<T>>) -> Self {
        TimeSeriesDecomposition {
            predictor,
            fallback: LorenzoPredictor::new(conf.abs_error_bound),
            quantizer,
            ts0,
            _t: PhantomData,
        }
    }

    fn shape(conf: &Config) -> Result<(usize, usize, usize)> {
        if conf.dims.len() != 2 {
            return Err(Error::InvalidArgument("timestep prediction requires 2d dataset".into()));
        }
        if conf.block_size <= 0 {
            return Err(Error::InvalidArgument("block size must be positive".into()));
        }
        Ok((conf.dims[0], conf.dims[1], conf.block_size as usize))
    }
}

impl<T: SzFloat, P: Predictor<T, 1>, Q: Quantizer<T>> Decomposition<T> for TimeSeriesDecomposition<T, P, Q> {
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>> {
        let (steps, points, bs) = Self::shape(conf)?;
        let mut bins = Vec::with_capacity(conf.num);
        let ts0_recon: Vec<T> = if let Some(ts0) = &self.ts0 {
            for j in 0..points {
                bins.push(self.quantizer.quantize_and_overwrite(&mut data[j], ts0[j]));
            }
            data[..points].to_vec()
        } else {
            // SZ3 pads for the predictor alone; the fallback's reach is covered too, which only
            // differs where SZ3 would read before its buffer.
            let padding = self.predictor.padding().max(self.fallback.padding());
            let mut padded = Padded::<T, 1>::new(Some(&data[..points]), [points], padding);
            let mut block = padded.block_iter(bs);
            loop {
                let use_main = self.predictor.precompress(&block, &padded.buf);
                let p: &mut dyn Predictor<T, 1> = if use_main {
                    &mut self.predictor
                } else {
                    &mut self.fallback
                };
                p.precompress_block_commit();
                let p: &dyn Predictor<T, 1> = p;
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
            let mut v = vec![T::ZERO; points];
            padded.copy_out(&mut v);
            v
        };
        for j in 0..points {
            let mut prev = ts0_recon[j];
            for i in 1..steps {
                let idx = i * points + j;
                bins.push(self.quantizer.quantize_and_overwrite(&mut data[idx], prev));
                prev = data[idx];
            }
        }
        Ok(bins)
    }

    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()> {
        let (steps, points, bs) = Self::shape(conf)?;
        if bins.len() < conf.num {
            return corrupt("time series: fewer bins than the grid consumes");
        }
        let mut k = 0;
        if let Some(ts0) = &self.ts0 {
            for j in 0..points {
                out[j] = self.quantizer.recover(ts0[j], bins[k])?;
                k += 1;
            }
        } else {
            let padding = self.predictor.padding().max(self.fallback.padding());
            let mut padded = Padded::<T, 1>::new(None, [points], padding);
            let mut block = padded.block_iter(bs);
            loop {
                let use_main = self.predictor.predecompress(&block)?;
                let p: &dyn Predictor<T, 1> = if use_main { &self.predictor } else { &self.fallback };
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
            padded.copy_out(&mut out[..points]);
        }
        for j in 0..points {
            for i in 1..steps {
                let idx = i * points + j;
                out[idx] = self.quantizer.recover(out[idx - points], bins[k])?;
                k += 1;
            }
        }
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
