//! The last stage of the pipeline: a general-purpose lossless codec over the encoded bytes.

use crate::bytes::ByteWriter;
use crate::error::{Error, Result, corrupt};

/// SZ3's `LosslessInterface`.
pub trait Lossless {
    /// Compresses `src`, appending to `out`. Fails with [`Error::BufferTooSmall`] when the result
    /// would exceed `cap` bytes.
    fn compress(&self, src: &[u8], out: &mut Vec<u8>, cap: usize) -> Result<usize>;
    /// Reverse of `compress`; `max_len` bounds the size the stream may declare.
    fn decompress(&self, src: &[u8], max_len: usize) -> Result<Vec<u8>>;
}

/// zstd behind an 8-byte decompressed-size header (`Lossless_zstd`).
#[derive(Clone, Copy, Debug)]
pub struct Zstd {
    pub level: i32,
}

impl Default for Zstd {
    fn default() -> Self {
        Zstd { level: 3 }
    }
}

impl Zstd {
    pub fn compress_bound(len: usize) -> usize {
        zstd_safe::compress_bound(len)
    }
}

impl Lossless for Zstd {
    fn compress(&self, src: &[u8], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
        if cap < 8 {
            return Err(Error::BufferTooSmall);
        }
        // zstd writes from the start of a Vec, so the frame goes to its own buffer.
        let mut frame: Vec<u8> = Vec::with_capacity(zstd_safe::compress_bound(src.len()));
        match zstd_safe::compress(&mut frame, src, self.level) {
            Ok(n) if n <= cap - 8 => {
                out.put_u64(src.len() as u64);
                out.extend_from_slice(&frame);
                Ok(n + 8)
            }
            Ok(_) => Err(Error::BufferTooSmall),
            Err(code) => Err(Error::Lossless(zstd_safe::get_error_name(code).to_string())),
        }
    }

    fn decompress(&self, src: &[u8], max_len: usize) -> Result<Vec<u8>> {
        if src.len() < 8 {
            return corrupt("lossless: compressed data is smaller than the size header");
        }
        let len = u64::from_le_bytes(src[..8].try_into().unwrap());
        if len > max_len as u64 {
            return corrupt("lossless: declared decompressed size exceeds the allowed capacity");
        }
        let len = len as usize;
        let mut out: Vec<u8> = Vec::with_capacity(len);
        match zstd_safe::decompress(&mut out, &src[8..]) {
            Ok(n) if n == len => Ok(out),
            Ok(_) => corrupt("lossless: stream does not decompress to the size it declares"),
            Err(code) => Err(Error::Lossless(zstd_safe::get_error_name(code).to_string())),
        }
    }
}

/// Copies bytes through unchanged (`Lossless_bypass`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bypass;

impl Lossless for Bypass {
    fn compress(&self, src: &[u8], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
        if src.len() > cap {
            return Err(Error::BufferTooSmall);
        }
        out.extend_from_slice(src);
        Ok(src.len())
    }

    fn decompress(&self, src: &[u8], max_len: usize) -> Result<Vec<u8>> {
        if src.len() > max_len {
            return corrupt("bypass lossless: payload exceeds the allowed capacity");
        }
        Ok(src.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstd_round_trip() {
        let data: Vec<u8> = (0..10000u32).map(|i| (i % 17) as u8).collect();
        let mut out = Vec::new();
        let n = Zstd::default().compress(&data, &mut out, usize::MAX).unwrap();
        assert_eq!(n, out.len());
        assert_eq!(Zstd::default().decompress(&out, usize::MAX).unwrap(), data);
        let mut small = Vec::new();
        assert_eq!(
            Zstd::default().compress(&data, &mut small, 10),
            Err(Error::BufferTooSmall)
        );
        assert!(small.is_empty());
    }
}
