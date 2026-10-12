//! Canonical Huffman coder for integer bins, in SZ3's 3.4.0 format.
//!
//! `save` writes the number of distinct bins D as a u32; if D > 0, the smallest bin; if D > 1, for
//! each bin in ascending order the Elias-gamma code of its gap from the one before (none for the
//! first), then the Elias-gamma code of 1 + the zigzag of its code length minus the previous one's.
//! `encode` writes the payload's bit count as a u64, then the codes MSB-first.
//!
//! Code lengths, capped at 32 bits, are those of a Huffman tree built in (frequency, bin) order, so
//! every platform writes the same bytes.

use std::collections::HashMap;

use super::Encoder;
use crate::bytes::{ByteWriter, Reader};
use crate::error::{Error, Result, corrupt};

const MAX_LEN: usize = 32;
const TABLE_BITS: u32 = 12;

#[derive(Clone, Copy, Default)]
struct Entry {
    sym: i32,
    /// 0: the code is longer than the table, or is not a code.
    len: u8,
}

#[derive(Clone)]
pub struct HuffmanEncoder {
    offset: i32,
    /// Distinct `bin - offset`, ascending.
    syms: Vec<u64>,
    /// Code length of each.
    lens: Vec<u8>,
    header_bits: u64,
    payload_bits: u64,
    count: [u64; MAX_LEN + 1],
    // encoder: (code << 8) | length, indexed by bin - offset (dense) or keyed by it (sparse)
    dense_codes: Vec<u64>,
    sparse_codes: HashMap<u32, u64>,
    dense: bool,
    // decoder
    max_len: u32,
    table_bits: u32,
    first: [u64; MAX_LEN + 1],
    base: [u64; MAX_LEN + 1],
    limit: [u64; MAX_LEN + 1],
    sorted: Vec<i32>,
    table: Vec<Entry>,
}

#[inline]
fn gamma_bits(v: u64) -> u64 {
    2 * (64 - v.leading_zeros() as u64) - 1
}

#[inline]
fn zigzag(d: i32) -> u64 {
    if d >= 0 {
        2 * d as u64
    } else {
        2 * (-(d as i64)) as u64 - 1
    }
}

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    nb: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        BitWriter { out, acc: 0, nb: 0 }
    }
    /// n <= 32
    fn put(&mut self, v: u64, n: u32) {
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        self.acc = (self.acc << n) | (v & mask);
        self.nb += n;
        while self.nb >= 8 {
            self.nb -= 8;
            self.out.push((self.acc >> self.nb) as u8);
        }
    }
    /// v >= 1
    fn gamma(&mut self, v: u64) {
        let mut n = 64 - v.leading_zeros();
        let mut z = n - 1;
        while z > 0 {
            let k = z.min(32);
            self.put(0, k);
            z -= k;
        }
        if n > 32 {
            self.put(v >> 32, n - 32);
            n = 32;
        }
        self.put(v, n);
    }
    fn finish(self) {
        if self.nb > 0 {
            self.out.push((self.acc << (8 - self.nb)) as u8);
        }
    }
}

struct BitReader<'a> {
    p: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn bit(&mut self) -> Result<u64> {
        if self.pos >= self.p.len() * 8 {
            return corrupt("Huffman: truncated code table");
        }
        let b = (self.p[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(b as u64)
    }
    fn gamma(&mut self) -> Result<u64> {
        let mut z = 0;
        while self.bit()? == 0 {
            z += 1;
            if z > 63 {
                return corrupt("Huffman: invalid gamma code");
            }
        }
        let mut v = 1u64;
        for _ in 0..z {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }
    fn bytes(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

impl Default for HuffmanEncoder {
    fn default() -> Self {
        HuffmanEncoder {
            offset: 0,
            syms: Vec::new(),
            lens: Vec::new(),
            header_bits: 0,
            payload_bits: 0,
            count: [0; MAX_LEN + 1],
            dense_codes: Vec::new(),
            sparse_codes: HashMap::new(),
            dense: true,
            max_len: 0,
            table_bits: 0,
            first: [0; MAX_LEN + 1],
            base: [0; MAX_LEN + 1],
            limit: [0; MAX_LEN + 1],
            sorted: Vec::new(),
            table: Vec::new(),
        }
    }
}

impl HuffmanEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Upper bound of `save` plus `encode` for `num_bins` bins holding at most
    /// `distinct_symbols` values.
    pub fn size_bound(num_bins: usize, distinct_symbols: usize) -> usize {
        let d = num_bins.min(distinct_symbols);
        4 + 4 + (d * (16 * 4 + 10)).div_ceil(8) + 8 + (num_bins * MAX_LEN).div_ceil(8)
    }

    fn build_lengths(&mut self, freq: &[u64]) -> Result<()> {
        let d = self.syms.len();
        if d > u32::MAX as usize {
            return Err(Error::InvalidArgument(
                "Huffman: more distinct bins than a uint32 counts".into(),
            ));
        }
        self.lens = vec![0; d];
        if d == 1 {
            return Ok(());
        }
        // Ascending frequency, ties by bin.
        let mut order: Vec<u32> = (0..d as u32).collect();
        order.sort_unstable_by_key(|&i| (freq[i as usize], i));
        let mut a: Vec<u64> = order.iter().map(|&i| freq[i as usize]).collect();
        minimum_redundancy(&mut a);

        // Clamp to MAX_LEN, then move the deepest leaves shorter than MAX_LEN down until Kraft holds.
        let mut count = [0u64; MAX_LEN + 1];
        for &l in &a {
            count[(l as usize).min(MAX_LEN)] += 1;
        }
        let mut kraft: u64 = 0;
        for l in 1..=MAX_LEN {
            kraft += count[l] << (MAX_LEN - l);
        }
        while kraft > (1u64 << MAX_LEN) {
            let mut l = MAX_LEN - 1;
            while count[l] == 0 {
                l -= 1;
            }
            count[l] -= 1;
            count[l + 1] += 1;
            kraft -= 1u64 << (MAX_LEN - l - 1);
        }
        let mut i = 0;
        for l in (1..=MAX_LEN).rev() {
            for _ in 0..count[l] {
                let o = order[i] as usize;
                self.lens[o] = l as u8;
                self.payload_bits += freq[o] * l as u64;
                i += 1;
            }
        }
        self.header_bits = 0;
        for j in 0..d {
            if j > 0 {
                self.header_bits += gamma_bits(self.syms[j] - self.syms[j - 1]);
            }
            let prev = if j > 0 { self.lens[j - 1] as i32 } else { 0 };
            self.header_bits += gamma_bits(zigzag(self.lens[j] as i32 - prev) + 1);
        }
        Ok(())
    }

    /// Codes of one length are consecutive and follow bin order.
    fn canonical(&mut self) -> [u64; MAX_LEN + 1] {
        self.count = [0; MAX_LEN + 1];
        for &l in &self.lens {
            self.count[l as usize] += 1;
        }
        self.count[0] = 0;
        let mut first = [0u64; MAX_LEN + 1];
        let mut code = 0u64;
        for l in 1..=MAX_LEN {
            first[l] = code;
            code = (code + self.count[l]) << 1;
        }
        first
    }

    fn build_codes(&mut self) {
        if self.syms.len() < 2 {
            return;
        }
        let mut first = self.canonical();
        for i in 0..self.syms.len() {
            let l = self.lens[i] as usize;
            let e = (first[l] << 8) | l as u64;
            first[l] += 1;
            if self.dense {
                self.dense_codes[self.syms[i] as usize] = e;
            } else {
                self.sparse_codes.insert(self.syms[i] as u32, e);
            }
        }
    }

    fn read_table(&mut self, r: &mut BitReader, d: usize) -> Result<()> {
        self.syms = vec![0; d];
        self.lens = vec![0; d];
        // offset + bin must stay an i32.
        let max_sym = (i32::MAX as u32).wrapping_sub(self.offset as u32) as u64;
        let mut s: u64 = 0;
        let mut prev: i32 = 0;
        for i in 0..d {
            if i > 0 {
                let g = r.gamma()?;
                if g > max_sym - s {
                    return corrupt("Huffman: bin out of range");
                }
                s += g;
            }
            let z = r.gamma()? - 1;
            if z > 2 * MAX_LEN as u64 {
                return corrupt("Huffman: invalid code length");
            }
            let len = prev
                + if z & 1 == 1 {
                    -(z.div_ceil(2) as i32)
                } else {
                    (z / 2) as i32
                };
            if len < 1 || len > MAX_LEN as i32 {
                return corrupt("Huffman: invalid code length");
            }
            self.syms[i] = s;
            self.lens[i] = len as u8;
            prev = len;
        }
        Ok(())
    }

    fn build_decoder(&mut self) -> Result<()> {
        let d = self.syms.len();
        if d < 2 {
            return Ok(());
        }
        self.first = self.canonical();
        let mut kraft: u64 = 0;
        for l in 1..=MAX_LEN {
            kraft += self.count[l] << (MAX_LEN - l);
            if self.count[l] != 0 {
                self.max_len = l as u32;
            }
        }
        if kraft > (1u64 << MAX_LEN) {
            return corrupt("Huffman: code lengths overfull");
        }
        self.base[1] = 0;
        for l in 2..=MAX_LEN {
            self.base[l] = self.base[l - 1] + self.count[l - 1];
        }
        // Codes of length l are below limit[l] once left-aligned in 32 bits.
        for l in 1..=MAX_LEN {
            self.limit[l] = (self.first[l] + self.count[l]) << (MAX_LEN - l);
        }
        let mut pos = self.base;
        self.sorted = vec![0; d];
        for i in 0..d {
            let l = self.lens[i] as usize;
            self.sorted[pos[l] as usize] = (self.offset as u32).wrapping_add(self.syms[i] as u32) as i32;
            pos[l] += 1;
        }
        self.table_bits = self.max_len.min(TABLE_BITS);
        let tb = self.table_bits as usize;
        self.table = vec![Entry::default(); 1 << tb];
        for l in 1..=tb {
            for k in 0..self.count[l] {
                let lo = ((self.first[l] + k) << (tb - l)) as usize;
                let e = Entry {
                    sym: self.sorted[(self.base[l] + k) as usize],
                    len: l as u8,
                };
                self.table[lo..lo + (1 << (tb - l))].fill(e);
            }
        }
        Ok(())
    }

    /// A code longer than the table, with its bits at the top of `acc`.
    #[inline]
    fn decode_long(&self, acc: u64) -> Result<(i32, u32)> {
        let w = acc >> 32;
        let mut len = self.table_bits + 1;
        for l in self.table_bits + 1..self.max_len {
            len += (w >= self.limit[l as usize]) as u32;
        }
        if len as usize > MAX_LEN || w >= self.limit[len as usize] {
            return corrupt("Huffman: invalid code in payload");
        }
        let l = len as usize;
        let idx = self.base[l] + ((w >> (32 - len)) - self.first[l]);
        Ok((self.sorted[idx as usize], len))
    }
}

impl HuffmanEncoder {
    /// Decodes `out.len()` symbols from `p`; returns the bits consumed.
    ///
    /// While 8 bytes remain, one big-endian load feeds several symbols. With `ZERO_RUNS` (a
    /// one-bit code exists), a byte of zeros is a run of that code's symbol, and a byte of ones a
    /// run of the second one-bit symbol when there are two; such runs are filled in bulk.
    fn decode_payload<const ZERO_RUNS: bool>(&self, p: &[u8], out: &mut [i32]) -> Result<u64> {
        let n = out.len();
        let nbytes = p.len();
        let tb = self.table_bits;
        let ml = self.max_len;
        let table = &self.table[..];
        let zero_sym = self.sorted[0];
        let one_runs = self.count[1] == 2;
        let one_sym = if one_runs { self.sorted[1] } else { 0 };
        let mut pos: u64 = 0;
        let mut i = 0usize;
        while i < n && (pos >> 3) as usize + 8 <= nbytes {
            let byte = (pos >> 3) as usize;
            let mut acc = u64::from_be_bytes(p[byte..byte + 8].try_into().unwrap()) << (pos & 7);
            let mut avail = 64 - (pos & 7) as u32;
            loop {
                let len;
                let top = acc >> 56;
                if ZERO_RUNS && (top == 0 || (one_runs && top == 0xff)) {
                    let ones = top != 0;
                    let w = if ones { !acc } else { acc };
                    let run = (w.leading_zeros()).min(avail) as usize;
                    let run = run.min(n - i);
                    out[i..i + run].fill(if ones { one_sym } else { zero_sym });
                    len = run as u32;
                    acc = if len < 64 { acc << len } else { 0 };
                    i += run;
                } else {
                    let e = table[(acc >> (64 - tb)) as usize];
                    if e.len != 0 {
                        out[i] = e.sym;
                        len = e.len as u32;
                    } else {
                        let (s, l) = self.decode_long(acc)?;
                        out[i] = s;
                        len = l;
                    }
                    acc <<= len;
                    i += 1;
                }
                avail -= len;
                pos += len as u64;
                if avail < ml || i >= n {
                    break;
                }
            }
        }
        while i < n {
            let byte = (pos >> 3) as usize;
            if byte >= nbytes {
                return corrupt("Huffman: payload length does not match its codes");
            }
            let mut tail = [0u8; 8];
            let k = (nbytes - byte).min(8);
            tail[..k].copy_from_slice(&p[byte..byte + k]);
            let acc = u64::from_be_bytes(tail) << (pos & 7);
            let e = table[(acc >> (64 - tb)) as usize];
            let (s, l) = if e.len != 0 {
                (e.sym, e.len as u32)
            } else {
                self.decode_long(acc)?
            };
            out[i] = s;
            pos += l as u64;
            i += 1;
        }
        Ok(pos)
    }
}

/// Moffat and Katajainen, "In-place calculation of minimum-redundancy codes" (1995): `a` holds the
/// weights in ascending order and receives the code lengths.
fn minimum_redundancy(a: &mut [u64]) {
    let n = a.len();
    let mut root = 0usize;
    let mut leaf = 2usize;
    a[0] += a[1];
    for next in 1..n - 1 {
        if leaf >= n || a[root] < a[leaf] {
            a[next] = a[root];
            a[root] = next as u64;
            root += 1;
        } else {
            a[next] = a[leaf];
            leaf += 1;
        }
        if leaf >= n || (root < next && a[root] < a[leaf]) {
            a[next] += a[root];
            a[root] = next as u64;
            root += 1;
        } else {
            a[next] += a[leaf];
            leaf += 1;
        }
    }
    a[n - 2] = 0;
    for k in (0..n - 2).rev() {
        a[k] = a[a[k] as usize] + 1;
    }
    let (mut avbl, mut used, mut depth, mut next) = (1usize, 0usize, 0u64, n);
    let mut r = n - 1;
    while avbl > 0 {
        while r > 0 && a[r - 1] == depth {
            used += 1;
            r -= 1;
        }
        while avbl > used {
            next -= 1;
            a[next] = depth;
            avbl -= 1;
        }
        avbl = 2 * used;
        depth += 1;
        used = 0;
    }
}

impl Encoder for HuffmanEncoder {
    fn preprocess_encode(&mut self, bins: &[i32], _state_num: i32) {
        *self = HuffmanEncoder::new();
        if bins.is_empty() {
            return;
        }
        let (mut mn, mut mx) = (bins[0], bins[0]);
        for &b in bins {
            mn = mn.min(b);
            mx = mx.max(b);
        }
        self.offset = mn;
        let off = mn as u32;
        let top = (mx as u32).wrapping_sub(off) as u64;
        self.dense = top < 4096.max(2 * bins.len() as u64);
        let mut freq = Vec::new();
        if self.dense {
            let mut f = vec![0u64; top as usize + 1];
            for &b in bins {
                f[(b as u32).wrapping_sub(off) as usize] += 1;
            }
            for (k, &c) in f.iter().enumerate() {
                if c != 0 {
                    self.syms.push(k as u64);
                    freq.push(c);
                }
            }
            self.dense_codes = vec![0; top as usize + 1];
        } else {
            let mut m: HashMap<u32, u64> = HashMap::new();
            for &b in bins {
                *m.entry((b as u32).wrapping_sub(off)).or_insert(0) += 1;
            }
            let mut kv: Vec<(u32, u64)> = m.into_iter().collect();
            kv.sort_unstable();
            for (k, c) in kv {
                self.syms.push(k as u64);
                freq.push(c);
            }
        }
        // At most 2^32 distinct i32 bins, so this cannot fail.
        self.build_lengths(&freq).expect("distinct bins fit in u32");
        self.build_codes();
    }

    fn size_est(&self) -> usize {
        if self.syms.is_empty() {
            return 4;
        }
        4 + 4 + self.header_bits.div_ceil(8) as usize
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.put_u32(self.syms.len() as u32);
        if self.syms.is_empty() {
            return;
        }
        out.put_i32(self.offset);
        if self.syms.len() == 1 {
            return;
        }
        let mut w = BitWriter::new(out);
        for i in 0..self.syms.len() {
            if i > 0 {
                w.gamma(self.syms[i] - self.syms[i - 1]);
            }
            let prev = if i > 0 { self.lens[i - 1] as i32 } else { 0 };
            w.gamma(zigzag(self.lens[i] as i32 - prev) + 1);
        }
        w.finish();
    }

    fn encode(&self, bins: &[i32], out: &mut Vec<u8>) -> Result<()> {
        out.put_u64(self.payload_bits);
        let not_seen = || Error::InvalidArgument("Huffman: bin not seen by preprocess_encode".into());
        if self.syms.len() < 2 {
            if bins.iter().any(|&b| self.syms.is_empty() || b != self.offset) {
                return Err(not_seen());
            }
            return Ok(());
        }
        let start = out.len();
        out.reserve(self.payload_bits.div_ceil(8) as usize + 8);
        let off = self.offset as u32;
        let mut acc: u64 = 0;
        let mut nb: u32 = 0;
        let mut put = |e: u64, out: &mut Vec<u8>| {
            let len = (e & 0xff) as u32;
            acc = (acc << len) | (e >> 8);
            nb += len;
            if nb >= 32 {
                nb -= 32;
                out.extend_from_slice(&((acc >> nb) as u32).to_be_bytes());
            }
        };
        if self.dense {
            let tab = &self.dense_codes;
            for &b in bins {
                let s = (b as u32).wrapping_sub(off) as usize;
                let e = *tab.get(s).ok_or_else(not_seen)?;
                put(e, out);
            }
        } else {
            for &b in bins {
                let e = self
                    .sparse_codes
                    .get(&(b as u32).wrapping_sub(off))
                    .copied()
                    .unwrap_or(0);
                put(e, out);
            }
        }
        let bits = (out.len() - start) as u64 * 8 + nb as u64;
        while nb >= 8 {
            nb -= 8;
            out.push((acc >> nb) as u8);
        }
        if nb > 0 {
            out.push((acc << (8 - nb)) as u8);
        }
        // A bin preprocess_encode did not count has a zero-length entry.
        if bits != self.payload_bits {
            return Err(Error::InvalidArgument(
                "Huffman: bins differ from preprocess_encode's".into(),
            ));
        }
        Ok(())
    }

    fn load(&mut self, r: &mut Reader) -> Result<()> {
        *self = HuffmanEncoder::new();
        let d = r.u32()? as usize;
        if d > 0 {
            self.offset = r.i32()?;
        }
        if d == 1 {
            self.syms = vec![0];
            self.lens = vec![0];
        } else if d > 1 {
            // Every bin after the first takes at least two bits of the table.
            if d - 1 > r.remaining() * 4 {
                return corrupt("Huffman: code table exceeds the buffer");
            }
            let mut br = BitReader { p: r.rest(), pos: 0 };
            self.read_table(&mut br, d)?;
            let used = br.bytes();
            r.skip(used)?;
        }
        self.build_decoder()
    }

    fn decode(&self, r: &mut Reader, n: usize) -> Result<Vec<i32>> {
        let bits = r.u64()?;
        if bits > r.remaining() as u64 * 8 {
            return corrupt("Huffman: payload exceeds the buffer");
        }
        let nbytes = bits.div_ceil(8) as usize;
        let p = r.take(nbytes)?;
        if self.syms.len() < 2 {
            if bits != 0 || (self.syms.is_empty() && n != 0) {
                return corrupt("Huffman: payload does not match the code table");
            }
            return Ok(vec![self.offset; n]);
        }
        if bits < n as u64 || bits / MAX_LEN as u64 > n as u64 {
            return corrupt("Huffman: payload does not match the value count");
        }
        let mut out = vec![0i32; n];
        let pos = if self.count[1] != 0 {
            self.decode_payload::<true>(p, &mut out)?
        } else {
            self.decode_payload::<false>(p, &mut out)?
        };
        if pos != bits {
            return corrupt("Huffman: payload length does not match its codes");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(bins: &[i32]) -> Vec<u8> {
        let mut enc = HuffmanEncoder::new();
        enc.preprocess_encode(bins, 0);
        let mut buf = Vec::new();
        enc.save(&mut buf);
        enc.encode(bins, &mut buf).unwrap();
        let mut dec = HuffmanEncoder::new();
        let mut r = Reader::new(&buf);
        dec.load(&mut r).unwrap();
        assert_eq!(dec.decode(&mut r, bins.len()).unwrap(), bins);
        assert_eq!(r.remaining(), 0);
        buf
    }

    #[test]
    fn small_cases() {
        round_trip(&[]);
        round_trip(&[5, 5, 5]);
        round_trip(&[1, 2]);
        round_trip(&[i32::MIN, i32::MAX, 0, 0, 0]);
    }

    #[test]
    fn skewed_distribution() {
        let mut bins = Vec::new();
        let mut x: u64 = 12345;
        for _ in 0..100_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((x >> 33) % 1000) as i32;
            bins.push(32768 + if v < 900 { 0 } else { v - 950 });
        }
        round_trip(&bins);
    }

    #[test]
    fn fibonacci_weights_hit_length_cap() {
        // Fibonacci frequencies give a maximally deep tree, beyond 32 bits.
        let mut bins = Vec::new();
        let (mut a, mut b) = (1u64, 1u64);
        for s in 0..40 {
            for _ in 0..a.min(2_000_000) {
                bins.push(s);
            }
            let c = a + b;
            a = b;
            b = c;
        }
        round_trip(&bins);
    }

    #[test]
    fn sparse_bins() {
        let bins: Vec<i32> = (0..1000).map(|i| i * 100_000 - 50_000_000).collect();
        round_trip(&bins);
    }
}
