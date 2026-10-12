use super::Predictor;
use crate::blockwise::Block;
use crate::bytes::Reader;
use crate::error::Result;
use crate::types::SzFloat;

/// N-dimensional Lorenzo predictor of order `L` (1 or 2).
#[derive(Clone, Debug)]
pub struct LorenzoPredictor<T, const N: usize, const L: usize> {
    noise: T,
}

impl<T: SzFloat, const N: usize, const L: usize> LorenzoPredictor<T, N, L> {
    pub fn new(eb: f64) -> Self {
        let k = match (L, N) {
            (1, 1) => 0.5,
            (1, 2) => 0.81,
            (1, 3) => 1.22,
            (1, 4) => 1.79,
            (2, 1) => 1.08,
            (2, 2) => 2.76,
            (2, 3) => 6.8,
            _ => 0.0,
        };
        LorenzoPredictor {
            noise: T::from_f64(k * eb),
        }
    }
}

impl<T: SzFloat, const N: usize, const L: usize> Predictor<T, N> for LorenzoPredictor<T, N, L> {
    fn precompress(&mut self, _: &Block<N>, _: &[T]) -> bool {
        true
    }
    fn precompress_block_commit(&mut self) {}
    fn predecompress(&mut self, _: &Block<N>) -> Result<bool> {
        Ok(true)
    }

    #[inline(always)]
    fn predict(&self, d: &[T], pos: usize, _: &[usize; N], ds: &[usize; N]) -> T {
        let ds: &[usize] = ds;
        // Neighbour offsets as in SZ3's prev1..prev4 (including their argument-to-stride order),
        // summed in the same order so the result is bit-identical.
        let p1 = |i: usize| d[pos - i];
        let p2 = |j: usize, i: usize| d[pos - (j * ds[0] + i)];
        let p3 = |k: usize, j: usize, i: usize| d[pos - (k * ds[1] + j * ds[0] + i)];
        let p4 = |t: usize, k: usize, j: usize, i: usize| d[pos - (t * ds[2] + k * ds[1] + j * ds[0] + i)];
        let c = T::c;
        match (N, L) {
            (1, 1) => p1(1),
            (2, 1) => p2(0, 1) + p2(1, 0) - p2(1, 1),
            (3, 1) => p3(0, 0, 1) + p3(0, 1, 0) + p3(1, 0, 0) - p3(0, 1, 1) - p3(1, 0, 1) - p3(1, 1, 0) + p3(1, 1, 1),
            (4, 1) => {
                p4(0, 0, 0, 1) + p4(0, 0, 1, 0) - p4(0, 0, 1, 1) + p4(0, 1, 0, 0) - p4(0, 1, 0, 1) - p4(0, 1, 1, 0)
                    + p4(0, 1, 1, 1)
                    + p4(1, 0, 0, 0)
                    - p4(1, 0, 0, 1)
                    - p4(1, 0, 1, 0)
                    + p4(1, 0, 1, 1)
                    - p4(1, 1, 0, 0)
                    + p4(1, 1, 0, 1)
                    + p4(1, 1, 1, 0)
                    - p4(1, 1, 1, 1)
            }
            (1, 2) => c(2) * p1(1) - p1(2),
            (2, 2) => {
                c(2) * p2(0, 1) - p2(0, 2) + c(2) * p2(1, 0) - c(4) * p2(1, 1) + c(2) * p2(1, 2) - p2(2, 0)
                    + c(2) * p2(2, 1)
                    - p2(2, 2)
            }
            (3, 2) => {
                c(2) * p3(0, 0, 1) - p3(0, 0, 2) + c(2) * p3(0, 1, 0) - c(4) * p3(0, 1, 1) + c(2) * p3(0, 1, 2)
                    - p3(0, 2, 0)
                    + c(2) * p3(0, 2, 1)
                    - p3(0, 2, 2)
                    + c(2) * p3(1, 0, 0)
                    - c(4) * p3(1, 0, 1)
                    + c(2) * p3(1, 0, 2)
                    - c(4) * p3(1, 1, 0)
                    + c(8) * p3(1, 1, 1)
                    - c(4) * p3(1, 1, 2)
                    + c(2) * p3(1, 2, 0)
                    - c(4) * p3(1, 2, 1)
                    + c(2) * p3(1, 2, 2)
                    - p3(2, 0, 0)
                    + c(2) * p3(2, 0, 1)
                    - p3(2, 0, 2)
                    + c(2) * p3(2, 1, 0)
                    - c(4) * p3(2, 1, 1)
                    + c(2) * p3(2, 1, 2)
                    - p3(2, 2, 0)
                    + c(2) * p3(2, 2, 1)
                    - p3(2, 2, 2)
            }
            // SZ3 has no 4-D second-order stencil and predicts 0.
            _ => T::ZERO,
        }
    }

    #[inline(always)]
    fn estimate_error(&self, d: &[T], pos: usize, idx: &[usize; N], ds: &[usize; N]) -> T {
        (d[pos] - self.predict(d, pos, idx, ds)).abs() + self.noise
    }

    fn save(&mut self, _: &mut Vec<u8>) -> Result<()> {
        Ok(())
    }
    fn load(&mut self, _: &mut Reader, _: usize) -> Result<()> {
        Ok(())
    }
    fn padding(&self) -> usize {
        2
    }
}
