//! The SZ3 stream: a 16-byte header, the algorithm's payload, then the configuration.
//!
//! ```text
//! [ magic u32 | data version u32 | payload size u64 | payload | config ]
//! ```

use crate::algo;
use crate::bytes::{ByteWriter, Pod, Reader};
use crate::config::{
    ALGO_MAP, Config, SZ3_DATA_VER, SZ3_DATA_VER_OLDEST, SZ3_MAGIC_NUMBER, SZ3_VERSION, algo as A, enum_to_string,
    version_int, version_str,
};
use crate::encoder::{HuffmanEncoder, HuffmanEncoderV1};
use crate::error::{Error, Result, corrupt};
use crate::lossless::{Lossless, Zstd};
use crate::stats::cal_abs_error_bound;
use crate::types::{SzData, SzFloat};

macro_rules! with_n {
    ($n:expr, $N:ident => $body:expr) => {
        match $n {
            1 => {
                const $N: usize = 1;
                $body
            }
            2 => {
                const $N: usize = 2;
                $body
            }
            3 => {
                const $N: usize = 3;
                $body
            }
            4 => {
                const $N: usize = 4;
                $body
            }
            _ => Err(Error::InvalidArgument(
                "Data dimension higher than 4 is not supported.".into(),
            )),
        }
    };
}

fn to_bytes<F: Pod>(data: &[F]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() * F::SIZE);
    v.put_slice(data);
    v
}

/// Upper bound on the compressed size (`SZ_compress_size_bound`).
pub fn compress_size_bound<T: SzData>(conf: &Config) -> usize {
    if conf.openmp {
        4096 + size_bound_omp(conf, T::FLOAT_SIZE)
    } else {
        4096 + conf.size_est() + Zstd::compress_bound(conf.num * T::FLOAT_SIZE)
    }
}

/// Threads for `openmp = true`, as an OpenMP build picks them: `OMP_NUM_THREADS`, else one per
/// core.
pub fn num_threads() -> usize {
    std::env::var("OMP_NUM_THREADS")
        .ok()
        .and_then(|v| v.split(',').next().and_then(|n| n.trim().parse::<usize>().ok()))
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

/// The dimension the chunked layout splits, the chunk count, and (outer, extent, inner) sizes.
fn omp_split(conf: &Config) -> (usize, usize, usize, usize) {
    let split = (conf.cmpr_algo == A::BIOMD && conf.n == 3) as usize;
    let ext = conf.dims[split];
    let outer = if split == 1 { conf.dims[0] } else { 1 };
    (split, outer, ext, conf.num / (outer * ext))
}

fn size_bound_omp(conf: &Config, float_size: usize) -> usize {
    let (_, _, ext, _) = omp_split(conf);
    let nt = num_threads().min(ext);
    let chunk = ext.div_ceil(nt) * (conf.num / ext);
    4 + nt * conf.size_est() + 2 * nt * 8 + nt * Zstd::compress_bound(chunk * float_size)
}

/// Compresses `data` (at least `conf.num` values, slowest dimension first) into an SZ3 stream.
pub fn compress<T: SzData>(conf: &Config, data: &[T]) -> Result<Vec<u8>> {
    let mut conf = conf.clone();
    compress_with(&mut conf, data)
}

/// As [`compress`], leaving in `conf` the configuration that was stored (absolute error bound,
/// chosen algorithm and settings).
pub fn compress_with<T: SzData>(conf: &mut Config, data: &[T]) -> Result<Vec<u8>> {
    if conf.n == 0 || conf.dims.is_empty() {
        return Err(Error::InvalidArgument("the configuration has no dimensions".into()));
    }
    if data.len() < conf.num {
        return Err(Error::InvalidArgument(format!(
            "{} values given for {} elements",
            data.len(),
            conf.num
        )));
    }
    let data = &data[..conf.num];
    let bound = compress_size_bound::<T>(conf);
    let cap = bound - 16 - conf.size_est() * 2;

    let mut out = Vec::with_capacity(bound / 2);
    out.put_u32(SZ3_MAGIC_NUMBER);
    out.put_u32(version_int(SZ3_DATA_VER));
    out.put_u64(0);
    let size = compress_impl(conf, data, &mut out, cap)?;
    out[8..16].copy_from_slice(&(size as u64).to_le_bytes());
    conf.save(&mut out)?;
    Ok(out)
}

fn compress_impl<T: SzData>(conf: &mut Config, data: &[T], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
    if let Some(f) = T::as_f32_slice(data) {
        return compress_float(conf, f, out, cap);
    }
    if let Some(f) = T::as_f64_slice(data) {
        return compress_float(conf, f, out, cap);
    }
    // Integers go through the float type that holds them, within floor(bound) + 0.49, so rounding
    // each decompressed value lands within the bound.
    if T::FLOAT_SIZE == 4 {
        let v: Vec<f32> = data.iter().map(|x| x.to_f64() as f32).collect();
        cal_abs_error_bound(conf, &v, None)?;
        conf.abs_error_bound = conf.abs_error_bound.floor() + 0.49;
        compress_float(conf, &v, out, cap)
    } else {
        let v: Vec<f64> = data.iter().map(|x| x.to_f64()).collect();
        if T::SIZE == 8 && v.iter().any(|x| x.abs() >= 9007199254740992.0) {
            return Err(Error::InvalidArgument(
                "8-byte integers must lie within +-2^53, where doubles hold them exactly".into(),
            ));
        }
        cal_abs_error_bound(conf, &v, None)?;
        conf.abs_error_bound = conf.abs_error_bound.floor() + 0.49;
        compress_float(conf, &v, out, cap)
    }
}

fn compress_float<F: SzFloat>(conf: &mut Config, data: &[F], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
    if conf.openmp {
        compress_omp(conf, data, out, cap, num_threads())
    } else {
        dispatch_compress(conf, data, out, cap)
    }
}

/// SZ3's OpenMP layout (`SZ_compress_OMP`): the data is split into `threads` chunks along the first
/// dimension (the atoms for BioMD 3-D data), compressed in parallel; written as the chunk count
/// (i32), each chunk's config, each chunk's size (u64), then the chunks.
pub fn compress_omp<F: SzFloat>(
    conf: &mut Config,
    data: &[F],
    out: &mut Vec<u8>,
    cap: usize,
    threads: usize,
) -> Result<usize> {
    let (split, outer, ext, inner) = omp_split(conf);
    let nt = threads.max(1).min(ext);
    if conf.error_bound_mode != crate::config::eb::ABS {
        // Range from per-chunk extremes, as SZ3 computes it.
        let mut lo_v = Vec::with_capacity(nt);
        let mut hi_v = Vec::with_capacity(nt);
        for i in 0..nt {
            let chunk = &data[i * conf.num / nt..(i + 1) * conf.num / nt];
            let (mut mn, mut mx) = (chunk[0], chunk[0]);
            for &v in &chunk[1..] {
                if v < mn {
                    mn = v;
                }
                if !(v < mx) {
                    mx = v;
                }
            }
            lo_v.push(mn);
            hi_v.push(mx);
        }
        let mx = hi_v.iter().copied().fold(hi_v[0], |a, b| if a < b { b } else { a });
        let mn = lo_v.iter().copied().fold(lo_v[0], |a, b| if b < a { b } else { a });
        cal_abs_error_bound(conf, data, Some(mx - mn))?;
    }
    let base = conf.clone();
    let results: Vec<Result<(Config, Vec<u8>)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..nt)
            .map(|i| {
                let base = &base;
                s.spawn(move || -> Result<(Config, Vec<u8>)> {
                    let lo = i * ext / nt;
                    let hi = (i + 1) * ext / nt;
                    let mut dims = base.dims.clone();
                    dims[split] = hi - lo;
                    let mut c = base.clone();
                    c.set_dims(&dims);
                    let cap = 8 + Zstd::compress_bound(c.num * F::SIZE);
                    let gathered;
                    let chunk: &[F] = if outer > 1 {
                        let run = (hi - lo) * inner;
                        gathered = (0..outer)
                            .flat_map(|o| data[(o * ext + lo) * inner..][..run].iter().copied())
                            .collect::<Vec<F>>();
                        &gathered
                    } else {
                        &data[lo * inner..hi * inner]
                    };
                    let mut buf = Vec::new();
                    dispatch_compress(&mut c, chunk, &mut buf, cap)?;
                    Ok((c, buf))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("compression thread panicked"))
            .collect()
    });
    let chunks = results.into_iter().collect::<Result<Vec<_>>>()?;
    let header: usize = 4 + nt * 8 + chunks.iter().map(|(c, _)| c.size_est()).sum::<usize>();
    let total: usize = chunks.iter().map(|(_, b)| b.len()).sum();
    if header + total > cap {
        return Err(Error::InvalidArgument(
            "The buffer for compressed data is not large enough.".into(),
        ));
    }
    let start = out.len();
    out.put_i32(nt as i32);
    for (c, _) in &chunks {
        c.save(out)?;
    }
    for (_, b) in &chunks {
        out.put_u64(b.len() as u64);
    }
    for (_, b) in &chunks {
        out.extend_from_slice(b);
    }
    Ok(out.len() - start)
}

/// `SZ_compress_dispatcher`: picks the algorithm, falls back to lossless when the bound is not
/// positive and finite, the output does not fit, or lossless compresses better.
pub fn dispatch_compress<F: SzFloat>(conf: &mut Config, data: &[F], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
    if conf.cmpr_algo == A::BIOMDXTC {
        return Err(Error::InvalidArgument(
            "ALGO_BIOMDXTC was removed in 3.4.0; use ALGO_BIOMD".into(),
        ));
    }
    cal_abs_error_bound(conf, data, None)?;
    if !(conf.abs_error_bound > 0.0) || !conf.abs_error_bound.is_finite() {
        conf.cmpr_algo = A::LOSSLESS;
    }
    let start = out.len();
    let mut lossy: Option<usize> = None;
    if conf.cmpr_algo != A::LOSSLESS {
        let n = conf.n as usize;
        let mut copy = data.to_vec();
        let r = match conf.cmpr_algo {
            A::BIOMD => Err(Error::Unsupported("ALGO_BIOMD is not ported yet".into())),
            A::LORENZO_REG => with_n!(n, N => algo::compress_lorenzo_reg::<F, N>(conf, &mut copy, out, cap)),
            A::INTERP => with_n!(n, N => algo::compress_interp::<F, N>(conf, &mut copy, out, cap)),
            A::INTERP_LORENZO => with_n!(n, N => algo::compress_interp_lorenzo::<F, N>(conf, &mut copy, out, cap)),
            A::NOPRED => algo::compress_nopred::<F>(conf, &mut copy, out, cap),
            _ => Err(Error::InvalidArgument("Unknown compression algorithm".into())),
        };
        match r {
            Ok(s) => lossy = Some(s),
            Err(Error::BufferTooSmall) => out.truncate(start),
            Err(e) => return Err(e),
        }
    }
    let Some(size) = lossy else {
        conf.cmpr_algo = A::LOSSLESS;
        return Zstd::default().compress(&to_bytes(data), out, cap);
    };
    // A lossy ratio under 3: keep lossless if it is smaller.
    if (conf.num * F::SIZE) as f64 / size as f64 / 1.0 < 3.0 {
        let mut z = Vec::new();
        if let Ok(zs) = Zstd::default().compress(&to_bytes(data), &mut z, size)
            && zs < size
        {
            conf.cmpr_algo = A::LOSSLESS;
            out.truncate(start);
            out.extend_from_slice(&z);
            return Ok(zs);
        }
    }
    Ok(size)
}

/// Decompresses an SZ3 stream; returns its configuration and the values.
pub fn decompress<T: SzData>(cmp: &[u8]) -> Result<(Config, Vec<T>)> {
    let mut conf = Config::default();
    let mut r = Reader::new(cmp);
    if cmp.len() < 16 {
        return corrupt("compressed data is smaller than the header");
    }
    conf.sz3_magic_number = r.u32()?;
    if conf.sz3_magic_number != SZ3_MAGIC_NUMBER {
        return Err(Error::InvalidArgument(
            "magic number mismatch, the input data is not compressed by SZ3".into(),
        ));
    }
    conf.sz3_data_ver = r.u32()?;
    if conf.sz3_data_ver < version_int(SZ3_DATA_VER_OLDEST) || conf.sz3_data_ver > version_int(SZ3_DATA_VER) {
        let mut m = format!(
            "SZ3 {SZ3_VERSION} reads data versions {SZ3_DATA_VER_OLDEST} to {SZ3_DATA_VER}, but this data is version {}. ",
            version_str(conf.sz3_data_ver)
        );
        m += if conf.sz3_data_ver > version_int(SZ3_DATA_VER) {
            "Upgrade SZ3 to decompress it.".to_string()
        } else {
            format!("Use SZ3 v{} to decompress it.", version_str(conf.sz3_data_ver))
        }
        .as_str();
        return Err(Error::Unsupported(m));
    }
    let size = r.u64()?;
    if size > (cmp.len() - 16) as u64 {
        return corrupt("compressed payload size exceeds the buffer");
    }
    let payload = r.take(size as usize)?;
    conf.load(&mut r)?;
    let out = decompress_payload::<T>(&mut conf, payload)?;
    Ok((conf, out))
}

fn alloc<F: Clone + Default>(n: usize) -> Result<Vec<F>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Corrupt(format!("cannot allocate {n} values")))?;
    v.resize(n, F::default());
    Ok(v)
}

fn decompress_payload<T: SzData>(conf: &mut Config, payload: &[u8]) -> Result<Vec<T>> {
    if T::IS_FLOAT {
        return if T::FLOAT_SIZE == 4 {
            let mut v = alloc::<f32>(conf.num)?;
            decompress_impl(conf, payload, &mut v)?;
            Ok(T::from_f32_vec(v).unwrap())
        } else {
            let mut v = alloc::<f64>(conf.num)?;
            decompress_impl(conf, payload, &mut v)?;
            Ok(T::from_f64_vec(v).unwrap())
        };
    }
    if T::FLOAT_SIZE == 4 {
        let mut v = alloc::<f32>(conf.num)?;
        decompress_impl(conf, payload, &mut v)?;
        Ok(v.into_iter().map(|x| T::from_f64_rounded(x as f64)).collect())
    } else {
        let mut v = alloc::<f64>(conf.num)?;
        decompress_impl(conf, payload, &mut v)?;
        Ok(v.into_iter().map(T::from_f64_rounded).collect())
    }
}

fn decompress_impl<F: SzFloat>(conf: &Config, payload: &[u8], out: &mut [F]) -> Result<()> {
    if conf.openmp {
        decompress_omp(conf, payload, out)
    } else {
        dispatch_decompress(conf, payload, out)
    }
}

/// `SZ_decompress_dispatcher`.
pub fn dispatch_decompress<F: SzFloat>(conf: &Config, cmp: &[u8], out: &mut [F]) -> Result<()> {
    let n = conf.n as usize;
    // Data version 3.3.2: its Huffman streams are HuffmanEncoderV1's.
    if conf.sz3_data_ver < version_int("3.4.0") {
        return match conf.cmpr_algo {
            A::LORENZO_REG => {
                with_n!(n, N => algo::decompress_lorenzo_reg::<F, N, HuffmanEncoderV1>(conf, cmp, out))
            }
            A::INTERP => with_n!(n, N => algo::decompress_interp::<F, N, HuffmanEncoderV1>(conf, cmp, out)),
            A::LOSSLESS => decompress_lossless(conf, cmp, out),
            a => Err(Error::InvalidArgument(format!(
                "{} data of version {} is not supported; use SZ3 v{} to decompress it",
                enum_to_string(a, ALGO_MAP),
                version_str(conf.sz3_data_ver),
                version_str(conf.sz3_data_ver)
            ))),
        };
    }
    let n = conf.n as usize;
    match conf.cmpr_algo {
        A::LOSSLESS => decompress_lossless(conf, cmp, out),
        A::LORENZO_REG => with_n!(n, N => algo::decompress_lorenzo_reg::<F, N, HuffmanEncoder>(conf, cmp, out)),
        A::INTERP => with_n!(n, N => algo::decompress_interp::<F, N, HuffmanEncoder>(conf, cmp, out)),
        A::NOPRED => algo::decompress_nopred::<F>(conf, cmp, out),
        A::BIOMD => Err(Error::Unsupported("ALGO_BIOMD is not ported yet".into())),
        _ => Err(Error::InvalidArgument("Unknown compression algorithm".into())),
    }
}

fn decompress_lossless<F: SzFloat>(conf: &Config, cmp: &[u8], out: &mut [F]) -> Result<()> {
    let bytes = Zstd::default().decompress(cmp, conf.num * F::SIZE)?;
    if bytes.len() != conf.num * F::SIZE {
        return corrupt("Decompressed data size does not match the original data size");
    }
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(F::SIZE)) {
        *o = F::read_le(c);
    }
    Ok(())
}

/// Reads the chunked layout SZ3 writes with OpenMP: a chunk count, each chunk's config, each
/// chunk's size, then the chunks, split along the first dimension (the atoms for BioMD 3-D data).
fn decompress_omp<F: SzFloat>(conf: &Config, cmp: &[u8], out: &mut [F]) -> Result<()> {
    let mut r = Reader::new(cmp);
    let nthreads = r.i32()?;
    if nthreads <= 0 || nthreads as usize > cmp.len() {
        return corrupt("OMP: invalid thread count");
    }
    let nt = nthreads as usize;
    let split = (conf.cmpr_algo == A::BIOMD && conf.n == 3) as usize;
    let ext = conf.dims[split];
    let outer = if split == 1 { conf.dims[0] } else { 1 };
    let inner = conf.num / (outer * ext);
    let mut confs = Vec::with_capacity(nt);
    for _ in 0..nt {
        let mut c = Config::default();
        c.load(&mut r)?;
        c.sz3_data_ver = conf.sz3_data_ver;
        confs.push(c);
    }
    let sizes: Vec<u64> = r.vec(nt as u64)?;
    for (t, c) in confs.iter().enumerate() {
        let lo = t * ext / nt;
        let hi = (t + 1) * ext / nt;
        let run = (hi - lo) * inner;
        if c.num != outer * run {
            return Err(Error::Corrupt(
                "a chunk of the OpenMP stream does not match its share of the data".into(),
            ));
        }
        let chunk = r.take(usize::try_from(sizes[t]).or_else(|_| corrupt("chunk size overflows"))?)?;
        if outer > 1 {
            let mut g = alloc::<F>(c.num)?;
            dispatch_decompress(c, chunk, &mut g)?;
            for o in 0..outer {
                out[(o * ext + lo) * inner..][..run].copy_from_slice(&g[o * run..][..run]);
            }
        } else {
            dispatch_decompress(c, chunk, &mut out[lo * inner..lo * inner + run])?;
        }
    }
    Ok(())
}
