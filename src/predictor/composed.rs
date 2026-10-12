use std::marker::PhantomData;

use super::Predictor;
use crate::blockwise::Block;
use crate::bytes::{ByteWriter, Reader};
use crate::encoder::{Encoder, HuffmanEncoder};
use crate::error::{Result, corrupt};
use crate::types::SzFloat;

/// Picks, per block, the predictor with the smallest estimated error on sample points.
pub struct ComposedPredictor<T: SzFloat, const N: usize, E = HuffmanEncoder> {
    predictors: Vec<Box<dyn Predictor<T, N>>>,
    selection: Vec<i32>,
    sid: usize,
    current_index: usize,
    _encoder: PhantomData<E>,
}

impl<T: SzFloat, const N: usize, E: Encoder + Default> ComposedPredictor<T, N, E> {
    pub fn new(predictors: Vec<Box<dyn Predictor<T, N>>>) -> Self {
        assert!(!predictors.is_empty(), "Empty predictor list for ComposedPredictor.");
        ComposedPredictor {
            predictors,
            selection: Vec::new(),
            sid: 0,
            current_index: 0,
            _encoder: PhantomData,
        }
    }
}

impl<T: SzFloat, const N: usize, E: Encoder + Default> Predictor<T, N> for ComposedPredictor<T, N, E> {
    fn precompress(&mut self, block: &Block<N>, data: &[T]) -> bool {
        let mut errors = vec![0f64; self.predictors.len()];
        let mut valid = vec![false; self.predictors.len()];
        for (i, p) in self.predictors.iter_mut().enumerate() {
            valid[i] = p.precompress(block, data);
            if valid[i] {
                let p = &**p;
                let mut e = 0f64;
                block.for_each_sampling(|pos, idx| {
                    e += p.estimate_error(data, pos, idx, &block.strides).to_f64();
                });
                errors[i] = e;
            } else {
                errors[i] = f64::MAX;
            }
        }
        // First minimum, as std::min_element.
        let mut sid = 0;
        for i in 1..errors.len() {
            if errors[i] < errors[sid] {
                sid = i;
            }
        }
        self.sid = sid;
        valid[sid]
    }

    fn precompress_block_commit(&mut self) {
        self.selection.push(self.sid as i32);
        self.predictors[self.sid].precompress_block_commit();
    }

    fn predecompress(&mut self, block: &Block<N>) -> Result<bool> {
        let Some(&s) = self.selection.get(self.current_index) else {
            return corrupt("ran out of predictor selections while decompressing");
        };
        self.current_index += 1;
        if s < 0 || s as usize >= self.predictors.len() {
            return corrupt("predictor selection index is out of range");
        }
        self.sid = s as usize;
        self.predictors[self.sid].predecompress(block)
    }

    #[inline(always)]
    fn predict(&self, d: &[T], pos: usize, idx: &[usize; N], ds: &[usize; N]) -> T {
        self.predictors[self.sid].predict(d, pos, idx, ds)
    }

    fn estimate_error(&self, d: &[T], pos: usize, idx: &[usize; N], ds: &[usize; N]) -> T {
        self.predictors[self.sid].estimate_error(d, pos, idx, ds)
    }

    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        for p in &mut self.predictors {
            p.save(out)?;
        }
        out.put_u64(self.selection.len() as u64);
        if !self.selection.is_empty() {
            let mut enc = E::default();
            enc.preprocess_encode(&self.selection, self.predictors.len() as i32);
            enc.save(out);
            enc.encode(&self.selection, out)?;
        }
        Ok(())
    }

    fn load(&mut self, r: &mut Reader, max_blocks: usize) -> Result<()> {
        for p in &mut self.predictors {
            p.load(r, max_blocks)?;
        }
        let n = r.u64()?;
        if n > max_blocks as u64 {
            return corrupt("more predictor selections than blocks");
        }
        if n > 0 {
            let mut enc = E::default();
            enc.load(r)?;
            let n = usize::try_from(n).or_else(|_| corrupt("selection count overflows"))?;
            self.selection = enc.decode(r, n)?;
        }
        Ok(())
    }

    fn padding(&self) -> usize {
        self.predictors.iter().map(|p| p.padding()).max().unwrap_or(0)
    }

    fn size_est(&self) -> usize {
        let mut total = 8 + self.predictors.iter().map(|p| p.size_est()).sum::<usize>();
        if !self.selection.is_empty() {
            total += HuffmanEncoder::size_bound(self.selection.len(), self.predictors.len());
        }
        total
    }
}
