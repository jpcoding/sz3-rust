//! Glues a decomposition, an encoder and a lossless stage into a compressor
//! (SZ3's `SZGenericCompressor`).
//!
//! Stream layout (before the lossless stage): decomposition side information, encoder table,
//! bin count (u64), encoded bins.

use std::marker::PhantomData;

use crate::bytes::{ByteWriter, Reader};
use crate::config::Config;
use crate::decomposition::Decomposition;
use crate::encoder::Encoder;
use crate::error::{Error, Result, corrupt};
use crate::lossless::Lossless;
use crate::types::SzFloat;

pub struct GenericCompressor<T, D, E, L> {
    pub decomposition: D,
    pub encoder: E,
    pub lossless: L,
    _t: PhantomData<T>,
}

impl<T: SzFloat, D: Decomposition<T>, E: Encoder, L: Lossless> GenericCompressor<T, D, E, L> {
    pub fn new(decomposition: D, encoder: E, lossless: L) -> Self {
        GenericCompressor {
            decomposition,
            encoder,
            lossless,
            _t: PhantomData,
        }
    }

    /// The bytes handed to the lossless stage.
    pub fn encode(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<u8>> {
        let bins = self.decomposition.compress(conf, data)?;
        let (lo, hi) = self.decomposition.out_range();
        if lo != 0 {
            return Err(Error::InvalidArgument(
                "the decomposition's output range must start from 0".into(),
            ));
        }
        self.encoder.preprocess_encode(&bins, hi);
        let mut buf = Vec::with_capacity(self.decomposition.size_est() + self.encoder.size_est() + bins.len());
        self.decomposition.save(&mut buf)?;
        self.encoder.save(&mut buf);
        buf.put_u64(bins.len() as u64);
        self.encoder.encode(&bins, &mut buf)?;
        Ok(buf)
    }

    /// Compresses `data` (which may be overwritten) and appends the stream to `out`; fails with
    /// [`Error::BufferTooSmall`] past `cap` bytes.
    pub fn compress(&mut self, conf: &Config, data: &mut [T], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
        let buf = self.encode(conf, data)?;
        self.lossless.compress(&buf, out, cap)
    }

    pub fn decompress(&mut self, conf: &Config, cmp: &[u8], out: &mut [T]) -> Result<()> {
        // A valid stream holds at most every value plus bins and tables; refuse to allocate more.
        let max_len = conf.num.saturating_mul(16 + T::SIZE * 2).saturating_add(1 << 20);
        let buf = self.lossless.decompress(cmp, max_len)?;
        let mut r = Reader::new(&buf);
        self.decomposition.load(&mut r, conf)?;
        self.encoder.load(&mut r)?;
        let n = r.u64()?;
        if n > conf.num as u64 {
            return corrupt("more bins than the data has elements");
        }
        let bins = self.encoder.decode(&mut r, n as usize)?;
        drop(buf);
        self.decomposition.decompress(conf, &bins, out)
    }
}

impl<T: SzFloat> Decomposition<T> for Box<dyn Decomposition<T>> {
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>> {
        (**self).compress(conf, data)
    }
    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()> {
        (**self).decompress(conf, bins, out)
    }
    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        (**self).save(out)
    }
    fn load(&mut self, r: &mut Reader, conf: &Config) -> Result<()> {
        (**self).load(r, conf)
    }
    fn out_range(&self) -> (i32, i32) {
        (**self).out_range()
    }
    fn size_est(&self) -> usize {
        (**self).size_est()
    }
}
