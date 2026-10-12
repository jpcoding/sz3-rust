//! Decompositions turn data into integer bins (SZ3's `DecompositionInterface`).

mod biomd;
mod blockwise;
mod interpolation;
mod nopred;
mod timeseries;

pub use biomd::{BioMdDecomposition, NUM_STREAMS as BIOMD_NUM_STREAMS};
pub use blockwise::BlockwiseDecomposition;
pub use interpolation::InterpolationDecomposition;
pub use nopred::NoPredictionDecomposition;
pub use timeseries::TimeSeriesDecomposition;

use crate::bytes::Reader;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::types::SzFloat;

/// Predicts and quantizes the whole data set into bins, keeping the side information
/// (unpredictable values, predictor coefficients, ...) it needs to invert the process.
pub trait Decomposition<T: SzFloat> {
    /// Compresses `data`, which is overwritten with the decompressed values where the
    /// decomposition works in place.
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>>;
    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()>;
    fn save(&mut self, out: &mut Vec<u8>) -> Result<()>;
    /// Reads what `save` wrote; `conf` is the trusted configuration of the data.
    fn load(&mut self, r: &mut Reader, conf: &Config) -> Result<()>;
    /// Bins lie in `[out_range().0, out_range().1)`.
    fn out_range(&self) -> (i32, i32);
    fn size_est(&self) -> usize;
    /// The most bins a valid stream for `conf` holds; a stream claiming more is corrupt.
    fn max_bins(&self, conf: &Config) -> usize {
        conf.num
    }
}

pub(crate) fn dims_array<const N: usize>(conf: &Config) -> Result<[usize; N]> {
    if conf.dims.len() != N {
        return Err(Error::Corrupt(
            "configuration dimension count does not match the data".into(),
        ));
    }
    let mut d = [0; N];
    d.copy_from_slice(&conf.dims);
    Ok(d)
}
