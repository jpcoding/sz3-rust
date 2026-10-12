//! Per-segment entropy coding (SZ3's `SegmentedEncoder`).
//!
//! The bins are a sequence of segments `[n, n bins]`; each non-empty segment gets its own encoder, so
//! a decomposition that concatenates streams of different statistics gets a code per stream.
//! `save` writes the segment count (u32), then per segment its size as a LEB128 varint and, unless
//! empty, its encoder's table; `encode` writes the encoders' payloads in segment order.

use super::Encoder;
use crate::bytes::{ByteWriter, Reader, read_varint, write_varint};
use crate::error::{Error, Result, corrupt};

pub struct SegmentedEncoder<E> {
    encoders: Vec<E>,
    segments: Vec<Vec<i32>>,
    sizes: Vec<u64>,
    max_segments: u32,
}

impl<E: Encoder + Default> SegmentedEncoder<E> {
    pub fn new(max_segments: u32) -> Self {
        SegmentedEncoder {
            encoders: Vec::new(),
            segments: Vec::new(),
            sizes: Vec::new(),
            max_segments,
        }
    }
}

impl<E: Encoder + Default> Encoder for SegmentedEncoder<E> {
    fn preprocess_encode(&mut self, bins: &[i32], state_num: i32) {
        self.segments.clear();
        self.encoders.clear();
        let mut k = 0;
        while k < bins.len() {
            let n = bins[k];
            // The decomposition writes the segments; a bad one is a bug there.
            assert!(
                n >= 0 && (n as usize) < bins.len() - k,
                "SegmentedEncoder: a segment runs past the end"
            );
            let seg = bins[k + 1..k + 1 + n as usize].to_vec();
            if n > 0 {
                let mut e = E::default();
                e.preprocess_encode(&seg, state_num);
                self.encoders.push(e);
            }
            self.segments.push(seg);
            k += n as usize + 1;
        }
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.put_u32(self.segments.len() as u32);
        let mut e = 0;
        for s in &self.segments {
            write_varint(s.len() as u64, out);
            if !s.is_empty() {
                self.encoders[e].save(out);
                e += 1;
            }
        }
    }

    fn encode(&self, _: &[i32], out: &mut Vec<u8>) -> Result<()> {
        let mut e = 0;
        for s in &self.segments {
            if !s.is_empty() {
                self.encoders[e].encode(s, out)?;
                e += 1;
            }
        }
        Ok(())
    }

    fn load(&mut self, r: &mut Reader) -> Result<()> {
        let count = r.u32()?;
        if count as usize > r.remaining() {
            return corrupt("SegmentedEncoder: more segments than bytes");
        }
        if count > self.max_segments {
            return corrupt("SegmentedEncoder: more segments than the data has");
        }
        self.sizes.clear();
        self.encoders.clear();
        for _ in 0..count {
            let size = read_varint(r)?;
            if size > i32::MAX as u64 {
                return corrupt("SegmentedEncoder: a segment length beyond INT_MAX");
            }
            self.sizes.push(size);
            if size > 0 {
                let mut e = E::default();
                e.load(r)?;
                self.encoders.push(e);
            }
        }
        Ok(())
    }

    fn decode(&self, r: &mut Reader, n: usize) -> Result<Vec<i32>> {
        let mut total = self.sizes.len() as u64;
        for &s in &self.sizes {
            total = total
                .checked_add(s)
                .ok_or_else(|| Error::Corrupt("segment sizes overflow".into()))?;
        }
        if total != n as u64 {
            return corrupt("SegmentedEncoder: the segment lengths do not add up to the bin count");
        }
        let mut out = Vec::with_capacity(n);
        let mut e = 0;
        for &s in &self.sizes {
            out.push(s as i32);
            if s > 0 {
                out.extend(self.encoders[e].decode(r, s as usize)?);
                e += 1;
            }
        }
        Ok(out)
    }

    fn size_est(&self) -> usize {
        4 + 5 * self.segments.len() + self.encoders.iter().map(|e| e.size_est()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::HuffmanEncoder;

    #[test]
    fn round_trip() {
        let bins = vec![3, 1, 2, 3, 0, 2, 100000, 7, 1, 5];
        let mut enc = SegmentedEncoder::<HuffmanEncoder>::new(12);
        enc.preprocess_encode(&bins, 0);
        let mut buf = Vec::new();
        enc.save(&mut buf);
        enc.encode(&bins, &mut buf).unwrap();
        let mut dec = SegmentedEncoder::<HuffmanEncoder>::new(12);
        let mut r = Reader::new(&buf);
        dec.load(&mut r).unwrap();
        assert_eq!(dec.decode(&mut r, bins.len()).unwrap(), bins);
        assert_eq!(r.remaining(), 0);
    }
}
