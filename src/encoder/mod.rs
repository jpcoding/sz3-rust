//! Entropy coding of quantization bins (SZ3's `EncoderInterface`).

mod bypass;
mod huffman;
mod huffman_v1;
mod segmented;

pub use bypass::BypassEncoder;
pub use huffman::HuffmanEncoder;
pub use huffman_v1::HuffmanEncoderV1;
pub use segmented::SegmentedEncoder;

use crate::bytes::Reader;
use crate::error::{Error, Result, corrupt};

/// Turns integer bins into a compact byte stream and back.
pub trait Encoder {
    /// Builds the code from the bins to be encoded. `state_num > 0` says the bins lie in
    /// `[0, state_num)`; it is only a hint.
    fn preprocess_encode(&mut self, bins: &[i32], state_num: i32);
    /// Appends the code table.
    fn save(&self, out: &mut Vec<u8>);
    /// Appends the encoded bins; they must be the ones `preprocess_encode` saw.
    fn encode(&self, bins: &[i32], out: &mut Vec<u8>) -> Result<()>;
    /// Reads the code table.
    fn load(&mut self, r: &mut Reader) -> Result<()>;
    /// Reads `n` bins.
    fn decode(&self, r: &mut Reader, n: usize) -> Result<Vec<i32>>;
    /// Reads `n` bins as a stream decoded on demand; by default all at once.
    fn decode_stream<'a>(&'a self, r: &mut Reader<'a>, n: usize) -> Result<BinStream<'a>> {
        Ok(BinStream::from_vec(self.decode(r, n)?))
    }
    /// Upper bound of the size of `save`.
    fn size_est(&self) -> usize {
        0
    }
}

/// Fills its slice with the next bins of a stream.
type Fill<'a> = Box<dyn FnMut(&mut [i32]) -> Result<()> + 'a>;

/// Bins read in order, decoded a chunk at a time, so a decomposition that consumes them in order
/// never holds them all.
pub struct BinStream<'a> {
    buf: Vec<i32>,
    at: usize,
    total: usize,
    decoded: usize,
    fill: Option<Fill<'a>>,
    err: Option<Error>,
}

const CHUNK: usize = 1 << 14;

impl<'a> BinStream<'a> {
    /// Bins already decoded.
    pub fn from_vec(v: Vec<i32>) -> Self {
        BinStream {
            total: v.len(),
            decoded: v.len(),
            buf: v,
            at: 0,
            fill: None,
            err: None,
        }
    }

    /// `total` bins that `fill` decodes in order.
    pub fn new(total: usize, fill: Fill<'a>) -> Self {
        BinStream {
            buf: Vec::new(),
            at: 0,
            total,
            decoded: 0,
            fill: Some(fill),
            err: None,
        }
    }

    /// The number of bins in the stream.
    pub fn len(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// The decoded bins not yet consumed.
    #[inline(always)]
    pub fn rest(&self) -> &[i32] {
        &self.buf[self.at..]
    }

    /// Marks all but the last `left` bins of [`BinStream::rest`] consumed.
    #[inline(always)]
    pub fn keep(&mut self, left: usize) {
        self.at = self.buf.len() - left.min(self.buf.len());
    }

    /// Drops the rest of the chunk and decodes the next one: empty at the end of the stream or
    /// after an error, which [`BinStream::finish`] reports.
    #[cold]
    #[inline(never)]
    pub fn refill(&mut self) -> &[i32] {
        self.at = 0;
        self.buf.clear();
        let m = (self.total - self.decoded).min(CHUNK);
        let Some(fill) = self.fill.as_mut() else {
            return &[];
        };
        if m == 0 || self.err.is_some() {
            return &[];
        }
        self.buf.resize(m, 0);
        if let Err(e) = fill(&mut self.buf) {
            self.err = Some(e);
            self.buf.clear();
            return &[];
        }
        self.decoded += m;
        &self.buf
    }

    /// The next bin; 0 past the end, which [`BinStream::finish`] reports.
    #[inline(always)]
    pub fn next_bin(&mut self) -> i32 {
        if let Some(&b) = self.buf.get(self.at) {
            self.at += 1;
            return b;
        }
        match self.refill().first() {
            Some(&b) => {
                self.at = 1;
                b
            }
            None => {
                if self.err.is_none() {
                    self.err = Some(Error::Corrupt("fewer bins than the decomposition consumes".into()));
                }
                0
            }
        }
    }

    /// All bins not yet consumed.
    pub fn take_all(&mut self) -> Result<Vec<i32>> {
        if self.at == 0 && self.decoded == self.total {
            return Ok(std::mem::take(&mut self.buf));
        }
        let mut v = self.rest().to_vec();
        while !self.refill().is_empty() {
            v.extend_from_slice(&self.buf);
        }
        self.buf.clear();
        match self.err.take() {
            Some(e) => Err(e),
            None => Ok(v),
        }
    }

    /// Decodes what is left, so a damaged tail is still found, and returns the first error.
    pub fn finish(mut self) -> Result<()> {
        if let Some(e) = self.err.take() {
            return Err(e);
        }
        while !self.refill().is_empty() {}
        match self.err {
            Some(e) => Err(e),
            None if self.decoded < self.total => corrupt("bin stream ended early"),
            None => Ok(()),
        }
    }
}
