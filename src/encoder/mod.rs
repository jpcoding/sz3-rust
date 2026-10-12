//! Entropy coding of quantization bins (SZ3's `EncoderInterface`).

mod huffman;
mod huffman_v1;

pub use huffman::HuffmanEncoder;
pub use huffman_v1::HuffmanEncoderV1;

use crate::bytes::Reader;
use crate::error::Result;

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
    /// Upper bound of the size of `save`.
    fn size_est(&self) -> usize {
        0
    }
}
