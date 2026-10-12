//! Little-endian serialization, as SZ3 stores every multi-byte value.

use crate::error::{Result, corrupt};

/// Appends little-endian values to a byte vector.
pub trait ByteWriter {
    fn put_u8(&mut self, v: u8);
    fn put_u32(&mut self, v: u32);
    fn put_i32(&mut self, v: i32);
    fn put_u64(&mut self, v: u64);
    fn put_f64(&mut self, v: f64);
    fn put<T: Pod>(&mut self, v: T);
    fn put_slice<T: Pod>(&mut self, v: &[T]);
}

impl ByteWriter for Vec<u8> {
    #[inline]
    fn put_u8(&mut self, v: u8) {
        self.push(v);
    }
    #[inline]
    fn put_u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put_i32(&mut self, v: i32) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put_u64(&mut self, v: u64) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put_f64(&mut self, v: f64) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    fn put<T: Pod>(&mut self, v: T) {
        v.write_le(self);
    }
    fn put_slice<T: Pod>(&mut self, v: &[T]) {
        self.reserve(v.len() * T::SIZE);
        for &x in v {
            x.write_le(self);
        }
    }
}

/// Plain values with a fixed little-endian encoding.
pub trait Pod: Copy + Default + Send + Sync + 'static {
    const SIZE: usize;
    fn write_le(self, out: &mut Vec<u8>);
    fn read_le(b: &[u8]) -> Self;
}

macro_rules! impl_pod {
    ($($t:ty),*) => {$(
        impl Pod for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            #[inline]
            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            #[inline]
            fn read_le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b[..Self::SIZE].try_into().unwrap())
            }
        }
    )*};
}
impl_pod!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// Bounds-checked little-endian reader over untrusted bytes.
#[derive(Clone)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The unread bytes.
    #[inline]
    pub fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return corrupt("attempt to read past the end of the compressed buffer");
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }

    #[inline]
    pub fn get<T: Pod>(&mut self) -> Result<T> {
        Ok(T::read_le(self.take(T::SIZE)?))
    }

    pub fn u8(&mut self) -> Result<u8> {
        self.get()
    }
    pub fn u32(&mut self) -> Result<u32> {
        self.get()
    }
    pub fn i32(&mut self) -> Result<i32> {
        self.get()
    }
    pub fn u64(&mut self) -> Result<u64> {
        self.get()
    }
    pub fn f64(&mut self) -> Result<f64> {
        self.get()
    }

    pub fn vec<T: Pod>(&mut self, n: u64) -> Result<Vec<T>> {
        if n > (self.remaining() / T::SIZE) as u64 {
            return corrupt("array length exceeds the compressed buffer");
        }
        let bytes = self.take(n as usize * T::SIZE)?;
        Ok(bytes.chunks_exact(T::SIZE).map(T::read_le).collect())
    }
}

/// Bits needed for the largest value (SZ3's `vector_bit_width`).
pub fn vector_bit_width(data: &[usize]) -> u8 {
    let max = data.iter().copied().max().unwrap_or(0);
    (usize::BITS - max.leading_zeros()) as u8
}

/// Packs each value in `bit_width` bits, LSB first (SZ3's `vector2bytes`).
pub fn vector2bytes(data: &[usize], bit_width: u8, out: &mut Vec<u8>) {
    let total_bits = data.len() * bit_width as usize;
    let start = out.len();
    out.resize(start + total_bits.div_ceil(8), 0);
    for (i, &v) in data.iter().enumerate() {
        for j in 0..bit_width as usize {
            if (v as u64 >> j) & 1 == 1 {
                let bit = i * bit_width as usize + j;
                out[start + bit / 8] |= 1 << (bit % 8);
            }
        }
    }
}

/// Reverse of [`vector2bytes`].
pub fn bytes2vector(r: &mut Reader, bit_width: u8, n: usize) -> Result<Vec<usize>> {
    let total = (n * bit_width as usize).div_ceil(8);
    let bytes = r.take(total)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v: u64 = 0;
        for j in 0..bit_width as usize {
            let bit = i * bit_width as usize + j;
            v |= (((bytes[bit / 8] >> (bit % 8)) & 1) as u64) << j;
        }
        out.push(v as usize);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_pack_round_trip() {
        let dims = vec![100usize, 500, 3, 70000];
        let w = vector_bit_width(&dims);
        assert_eq!(w, 17);
        let mut out = Vec::new();
        vector2bytes(&dims, w, &mut out);
        assert_eq!(out.len(), (4 * 17usize).div_ceil(8));
        let mut r = Reader::new(&out);
        assert_eq!(bytes2vector(&mut r, w, 4).unwrap(), dims);
    }
}
