//! Stores the bins as they are, 4 bytes each (SZ3's `BypassEncoder`).

use super::Encoder;
use crate::bytes::{ByteWriter, Reader};
use crate::error::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct BypassEncoder;

impl Encoder for BypassEncoder {
    fn preprocess_encode(&mut self, _: &[i32], _: i32) {}
    fn save(&self, _: &mut Vec<u8>) {}
    fn encode(&self, bins: &[i32], out: &mut Vec<u8>) -> Result<()> {
        out.put_slice(bins);
        Ok(())
    }
    fn load(&mut self, _: &mut Reader) -> Result<()> {
        Ok(())
    }
    fn decode(&self, r: &mut Reader, n: usize) -> Result<Vec<i32>> {
        r.vec(n as u64)
    }
}
