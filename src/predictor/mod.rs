//! Predictors for the blockwise decomposition (SZ3's `PredictorInterface`).

mod composed;
mod lorenzo;
mod regression;

pub use composed::ComposedPredictor;
pub use lorenzo::LorenzoPredictor;
pub use regression::RegressionPredictor;

use crate::blockwise::Block;
use crate::bytes::Reader;
use crate::error::Result;
use crate::types::SzFloat;

/// Predicts each value of a block from its already-reconstructed neighbours.
///
/// `data` is the padded buffer, `pos` the element's position in it, `idx` its index within the
/// block, `strides` the padded strides.
pub trait Predictor<T: SzFloat, const N: usize> {
    /// Fits the predictor to a block before compressing it; `false` declines the block.
    fn precompress(&mut self, block: &Block<N>, data: &[T]) -> bool;
    /// The block was compressed with this predictor: keep what `load` will need.
    fn precompress_block_commit(&mut self);
    /// Prepares to decompress a block; `false` means the compressor declined it.
    fn predecompress(&mut self, block: &Block<N>) -> Result<bool>;
    fn predict(&self, data: &[T], pos: usize, idx: &[usize; N], strides: &[usize; N]) -> T;
    fn estimate_error(&self, data: &[T], pos: usize, idx: &[usize; N], strides: &[usize; N]) -> T;
    fn save(&mut self, out: &mut Vec<u8>) -> Result<()>;
    /// Reads what `save` wrote; `max_blocks` bounds the block count the stream may claim.
    fn load(&mut self, r: &mut Reader, max_blocks: usize) -> Result<()>;
    /// Zero layers this predictor reads before each dimension.
    fn padding(&self) -> usize {
        0
    }
    fn size_est(&self) -> usize {
        0
    }
}
