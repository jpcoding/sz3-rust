//! MDZ: compression of molecular-dynamics trajectories `(frames, atoms)` per coordinate, choosing
//! per batch among level quantization (VQ, VQT), time-series prediction (MT) and SZ2-style
//! Lorenzo/regression (LR) (ICDE'22; SZ3's tools/mdz and `SZExaaltCompressor`).

use std::time::Instant;

use crate::algo::{BoxedPredictor, lorenzo_regression};
use crate::bytes::Reader;
use crate::compressor::GenericCompressor;
use crate::config::{Config, eb};
use crate::decomposition::TimeSeriesDecomposition;
use crate::encoder::{Encoder, HuffmanEncoder};
use crate::error::{Error, Result};
use crate::kmeans::get_cluster;
use crate::lossless::{Lossless, Zstd};
use crate::predictor::{ComposedPredictor, LorenzoPredictor, Predictor, RegressionPredictor};
use crate::quantizer::{LinearQuantizer, Quantizer};
use crate::types::SzFloat;

pub const COMPRESSOR_NAMES: [&str; 5] = ["VQ", "VQT", "MT", "LR", "TS"];

/// Level-based compressor for solid-state MD (`SZExaaltCompressor`): each value is predicted by
/// its nearest level, and the level by the previous one. `timestep_op` 0: along the data; 1: the
/// first frame by levels, later frames from the previous frame; 2: levels along time.
pub struct ExaaltCompressor<T: SzFloat> {
    quantizer: LinearQuantizer<T>,
    encoder: HuffmanEncoder,
    lossless: Zstd,
    level_start: f32,
    level_offset: f32,
    level_num: i32,
    timestep_op: i32,
}

impl<T: SzFloat> ExaaltCompressor<T> {
    pub fn new(quantizer: LinearQuantizer<T>, timestep_op: i32) -> Self {
        ExaaltCompressor {
            quantizer,
            encoder: HuffmanEncoder::new(),
            lossless: Zstd::default(),
            level_start: 0.0,
            level_offset: 0.0,
            level_num: 0,
            timestep_op,
        }
    }

    pub fn set_level(&mut self, start: f32, offset: f32, num: i32) {
        self.level_start = start;
        self.level_offset = offset;
        self.level_num = num + 200;
    }

    #[inline]
    fn quantize_to_level(&self, v: T) -> i32 {
        ((v - T::from_f64(self.level_start as f64)) / T::from_f64(self.level_offset as f64))
            .round()
            .to_f64() as i32
    }

    #[inline]
    fn level(&self, l: i32) -> T {
        T::from_f64((self.level_start + l as f32 * self.level_offset) as f64)
    }

    fn dims2(conf: &Config) -> Result<(usize, usize)> {
        if conf.dims.len() != 2 {
            return Err(Error::InvalidArgument("timestep prediction requires 2d dataset".into()));
        }
        Ok((conf.dims[0], conf.dims[1]))
    }

    pub fn compress(&mut self, conf: &Config, data: &mut [T], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
        let num = conf.num;
        let mut quant = vec![0i32; num];
        let mut pred = vec![0i32; num];
        let ln = self.level_num;
        let mut l0 = self.quantize_to_level(data[0]);
        pred[0] = l0.wrapping_add(ln);
        quant[0] = self.quantizer.quantize_and_overwrite(&mut data[0], self.level(l0));
        if self.timestep_op == 0 {
            for i in 1..num {
                let l = self.quantize_to_level(data[i]);
                pred[i] = l.wrapping_sub(l0).wrapping_add(ln);
                quant[i] = self.quantizer.quantize_and_overwrite(&mut data[i], self.level(l));
                l0 = l;
            }
        } else {
            let (d0, d1) = Self::dims2(conf)?;
            let mut levels = vec![0i32; d1];
            levels[0] = l0;
            for i in 1..d1 {
                levels[i] = self.quantize_to_level(data[i]);
                pred[i] = levels[i].wrapping_sub(levels[i - 1]).wrapping_add(ln);
                quant[i] = self
                    .quantizer
                    .quantize_and_overwrite(&mut data[i], self.level(levels[i]));
            }
            let mut k = d1;
            if self.timestep_op == 1 {
                for i in 0..d1 {
                    for t in 1..d0 {
                        let p = data[(t - 1) * d1 + i];
                        quant[k] = self.quantizer.quantize_and_overwrite(&mut data[t * d1 + i], p);
                        k += 1;
                    }
                }
                pred.truncate(d1);
            } else {
                for i in 0..d1 {
                    l0 = levels[i];
                    for t in 1..d0 {
                        let idx = t * d1 + i;
                        let l = self.quantize_to_level(data[idx]);
                        pred[k] = l.wrapping_sub(l0).wrapping_add(ln);
                        quant[k] = self.quantizer.quantize_and_overwrite(&mut data[idx], self.level(l));
                        k += 1;
                        l0 = l;
                    }
                }
            }
        }
        let mut buf = Vec::new();
        self.quantizer.save(&mut buf);
        self.encoder.preprocess_encode(&quant, self.quantizer.out_range().1);
        self.encoder.save(&mut buf);
        self.encoder.encode(&quant, &mut buf)?;
        self.encoder
            .preprocess_encode(&pred, ln.wrapping_mul(2).wrapping_add(1));
        self.encoder.save(&mut buf);
        self.encoder.encode(&pred, &mut buf)?;
        self.lossless.compress(&buf, out, cap)
    }

    pub fn decompress(&mut self, conf: &Config, cmp: &[u8], out: &mut [T]) -> Result<()> {
        let buf = self
            .lossless
            .decompress(cmp, conf.num.saturating_mul(16).saturating_add(1 << 20))?;
        let mut r = Reader::new(&buf);
        self.quantizer.load(&mut r)?;
        self.encoder.load(&mut r)?;
        let quant = self.encoder.decode(&mut r, conf.num)?;
        self.encoder.load(&mut r)?;
        let n_pred = if self.timestep_op == 1 {
            Self::dims2(conf)?.1
        } else {
            conf.num
        };
        let pred = self.encoder.decode(&mut r, n_pred)?;
        let ln = self.level_num;
        let mut l = pred[0].wrapping_sub(ln);
        out[0] = self.quantizer.recover(self.level(l), quant[0])?;
        if self.timestep_op == 0 {
            for i in 1..conf.num {
                l = l.wrapping_add(pred[i].wrapping_sub(ln));
                out[i] = self.quantizer.recover(self.level(l), quant[i])?;
            }
        } else {
            let (d0, d1) = Self::dims2(conf)?;
            let mut levels = vec![0i32; d1];
            levels[0] = l;
            for i in 1..d1 {
                l = l.wrapping_add(pred[i].wrapping_sub(ln));
                out[i] = self.quantizer.recover(self.level(l), quant[i])?;
                levels[i] = l;
            }
            let mut k = d1;
            if self.timestep_op == 1 {
                for i in 0..d1 {
                    for t in 1..d0 {
                        out[t * d1 + i] = self.quantizer.recover(out[(t - 1) * d1 + i], quant[k])?;
                        k += 1;
                    }
                }
            } else {
                for i in 0..d1 {
                    l = levels[i];
                    for t in 1..d0 {
                        l = l.wrapping_add(pred[k].wrapping_sub(ln));
                        out[t * d1 + i] = self.quantizer.recover(self.level(l), quant[k])?;
                        k += 1;
                    }
                }
            }
        }
        Ok(())
    }
}

fn quantizer<T: SzFloat>(conf: &Config) -> LinearQuantizer<T> {
    LinearQuantizer::new(conf.abs_error_bound, conf.quantbin_cnt / 2)
}

/// Time-series compressor (MT): timestep 0 from `ts0` (or spatially), then each point in time.
fn compress_mt<T: SzFloat>(
    conf: &Config,
    data: &mut [T],
    ts0: Option<&[T]>,
    dec: bool,
) -> Result<(usize, Option<Vec<T>>, f64, f64)> {
    let eb = conf.abs_error_bound;
    let mut preds: Vec<Box<dyn Predictor<T, 1>>> = Vec::new();
    if conf.lorenzo {
        preds.push(Box::new(LorenzoPredictor::<T, 1, 1>::new(eb)));
    }
    if conf.lorenzo2 {
        preds.push(Box::new(LorenzoPredictor::<T, 1, 2>::new(eb)));
    }
    if conf.regression {
        preds.push(Box::new(RegressionPredictor::<T, 1, HuffmanEncoder>::new(
            conf.block_size as usize,
            eb,
        )));
    }
    let make = |preds: Vec<Box<dyn Predictor<T, 1>>>| -> Box<dyn Predictor<T, 1>> {
        if preds.len() == 1 {
            preds.into_iter().next().unwrap()
        } else {
            Box::new(ComposedPredictor::<T, 1, HuffmanEncoder>::new(preds))
        }
    };
    if preds.is_empty() {
        return Err(Error::InvalidArgument(
            "All lorenzo and regression methods are disabled.".into(),
        ));
    }
    let mut c = GenericCompressor::new(
        TimeSeriesDecomposition::new(
            conf,
            BoxedPredictor(make(preds)),
            quantizer::<T>(conf),
            ts0.map(|s| s.to_vec()),
        ),
        HuffmanEncoder::new(),
        Zstd::default(),
    );
    run_generic(conf, data, dec, &mut c, |c, cmp, out| c.decompress(conf, cmp, out))
}

fn run_generic<T: SzFloat, C>(
    conf: &Config,
    data: &mut [T],
    dec: bool,
    c: &mut C,
    decompress: impl Fn(&mut C, &[u8], &mut [T]) -> Result<()>,
) -> Result<(usize, Option<Vec<T>>, f64, f64)>
where
    C: GenericCompress<T>,
{
    let t = Instant::now();
    let mut out = Vec::new();
    let size = c.compress_into(conf, data, &mut out, 2 * conf.num * T::SIZE)?;
    let ct = t.elapsed().as_secs_f64();
    if !dec {
        return Ok((size, None, ct, 0.0));
    }
    let t = Instant::now();
    let mut v = vec![T::ZERO; conf.num];
    decompress(c, &out, &mut v)?;
    Ok((size, Some(v), ct, t.elapsed().as_secs_f64()))
}

/// The compressors MDZ drives.
trait GenericCompress<T> {
    fn compress_into(&mut self, conf: &Config, data: &mut [T], out: &mut Vec<u8>, cap: usize) -> Result<usize>;
}

impl<T: SzFloat, D: crate::decomposition::Decomposition<T>, E: Encoder, L: Lossless> GenericCompress<T>
    for GenericCompressor<T, D, E, L>
{
    fn compress_into(&mut self, conf: &Config, data: &mut [T], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
        self.compress(conf, data, out, cap)
    }
}

impl<T: SzFloat> GenericCompress<T> for ExaaltCompressor<T> {
    fn compress_into(&mut self, conf: &Config, data: &mut [T], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
        self.compress(conf, data, out, cap)
    }
}

fn compress_vq<T: SzFloat>(
    conf: &Config,
    data: &mut [T],
    dec: bool,
    method: i32,
    levels: (f32, f32, i32),
) -> Result<(usize, Option<Vec<T>>, f64, f64)> {
    if levels.2 == 0 {
        return Err(Error::InvalidArgument(
            "VQ/VQT not availble on current dataset, please use ADP or MT".into(),
        ));
    }
    let mut c = ExaaltCompressor::new(quantizer::<T>(conf), method);
    c.set_level(levels.0, levels.1, levels.2);
    run_generic(conf, data, dec, &mut c, |c, cmp, out| c.decompress(conf, cmp, out))
}

fn compress_lr<T: SzFloat>(conf: &Config, data: &mut [T], dec: bool) -> Result<(usize, Option<Vec<T>>, f64, f64)> {
    let d = lorenzo_regression::<T, 2, HuffmanEncoder>(conf, quantizer(conf))?;
    let mut c = GenericCompressor::new(d, HuffmanEncoder::new(), Zstd::default());
    run_generic(conf, data, dec, &mut c, |c, cmp, out| c.decompress(conf, cmp, out))
}

/// Result of [`mdz_compress`].
#[derive(Debug, Default)]
pub struct MdzReport {
    pub compressed_size: usize,
    pub compress_time: f64,
    pub decompress_time: f64,
    /// (first frame, last frame, compressor) per batch.
    pub batches: Vec<(usize, usize, usize)>,
}

/// Picks the compressor for a batch by compressing a sample of up to 10 frames with each.
#[allow(clippy::too_many_arguments)]
fn select<T: SzFloat>(
    mut conf: Config,
    ts: usize,
    data_all: &[T],
    levels: (f32, f32, i32),
    ts0: &[T],
    batch_size: usize,
    report: &mut MdzReport,
) -> Result<usize> {
    let mut sizes = [usize::MAX; 10];
    let mut t = ts;
    if ts == 0 {
        if conf.dims[0] == 1 {
            return Ok(if levels.2 > 0 { 0 } else { 3 });
        }
        t = conf.dims[0] / 2;
        conf.dims[0] /= 2;
    }
    if batch_size > 10 {
        conf.dims[0] = 10;
    }
    conf.num = conf.dims[0] * conf.dims[1];
    let start = t * conf.dims[1];
    let Some(window) = data_all.get(start..start + conf.num) else {
        return Err(Error::InvalidArgument(
            "MDZ: too few frames to sample for selecting a compressor".into(),
        ));
    };
    let mut time = |r: (usize, Option<Vec<T>>, f64, f64)| {
        report.compress_time += r.2;
        r.0
    };
    if levels.2 > 0 {
        sizes[0] = time(compress_vq(&conf, &mut window.to_vec(), false, 0, levels)?);
        sizes[1] = time(compress_vq(&conf, &mut window.to_vec(), false, 1, levels)?);
    } else {
        sizes[3] = time(compress_lr(&conf, &mut window.to_vec(), false)?);
    }
    sizes[2] = time(compress_mt(&conf, &mut window.to_vec(), Some(ts0), false)?);
    // first minimum
    let mut best = 0;
    for i in 1..sizes.len() {
        if sizes[i] < sizes[best] {
            best = i;
        }
    }
    Ok(best)
}

/// Compresses (and decompresses into `dec`) a `(frames, atoms)` trajectory of one coordinate in
/// batches of `batch_size` frames (0: all); `method` -1 selects per 50 batches (`MDZ_Compress`).
pub fn mdz_compress_2d(
    mut conf: Config,
    input: &[f32],
    dec: &mut [f32],
    batch_size: usize,
    method: i32,
    report: &mut MdzReport,
) -> Result<()> {
    let dims = conf.dims.clone();
    let batch_size = if batch_size == 0 { dims[0] } else { batch_size };
    let method_batch = if method == -1 { 50 } else { 0 };
    let ts0: Vec<f32> = input[..dims[1]].to_vec();
    let mut levels = (0f32, 0f32, 0i32);
    if method != 2 && method != 3 && method != 4 {
        let mut sample_num = (0.1 * dims[1] as f64) as usize;
        sample_num = sample_num.min(20000);
        sample_num = sample_num.max(5000.min(dims[1]));
        levels = get_cluster(&input[..dims[1]], sample_num);
        if levels.2 as f64 > dims[1] as f64 * 0.25 {
            levels.2 = 0;
        }
    }
    let mut current = method;
    let mut lossless_first_frame = false;
    let mut ts = 0;
    while ts < dims[0] {
        conf.dims[0] = if ts + batch_size > dims[0] {
            dims[0] - ts
        } else {
            batch_size
        };
        conf.num = conf.dims[0] * conf.dims[1];
        let range = ts * dims[1]..ts * dims[1] + conf.num;
        let data = &input[range.clone()];
        let max = data.iter().copied().fold(data[0], |a, b| if a < b { b } else { a });
        let min = data.iter().copied().fold(data[0], |a, b| if b < a { b } else { a });
        if conf.error_bound_mode == eb::ABS {
            conf.rel_error_bound = conf.abs_error_bound / (max - min) as f64;
        } else if conf.error_bound_mode == eb::REL {
            conf.abs_error_bound = conf.rel_error_bound * (max - min) as f64;
        }
        if method_batch > 0 && (ts / batch_size) % method_batch == 0 {
            current = select(conf.clone(), ts, input, levels, &ts0, batch_size, report)? as i32;
        }
        report.batches.push((ts, ts + conf.dims[0] - 1, current as usize));
        let mut batch = data.to_vec();
        let r = match current {
            0 | 1 => compress_vq(&conf, &mut batch, true, current, levels)?,
            2 => {
                lossless_first_frame = true;
                compress_mt(&conf, &mut batch, Some(&ts0), true)?
            }
            4 => compress_mt(&conf, &mut batch, None, true)?,
            _ => compress_lr(&conf, &mut batch, true)?,
        };
        report.compressed_size += r.0;
        report.compress_time += r.2;
        report.decompress_time += r.3;
        dec[range].copy_from_slice(&r.1.unwrap());
        ts += batch_size;
    }
    if lossless_first_frame {
        let mut raw = Vec::new();
        crate::bytes::ByteWriter::put_slice(&mut raw, &ts0);
        let mut out = Vec::new();
        report.compressed_size += Zstd::default().compress(&raw, &mut out, raw.len() * 2)?;
    }
    Ok(())
}

/// `(frames, atoms, 3)`: each coordinate compressed as its own `(frames, atoms)` trajectory.
pub fn mdz_compress_3d(
    conf: &Config,
    input: &[f32],
    dec: &mut [f32],
    batch_size: usize,
    method: i32,
    report: &mut MdzReport,
) -> Result<()> {
    let (d0, d1, d2) = (conf.dims[0], conf.dims[1], conf.dims[2]);
    let mut planar = vec![0f32; conf.num];
    for f in 0..d0 {
        for a in 0..d1 {
            for c in 0..d2 {
                planar[c * d0 * d1 + f * d1 + a] = input[f * d1 * d2 + a * d2 + c];
            }
        }
    }
    let mut out = vec![0f32; conf.num];
    for c in 0..d2 {
        let mut c2 = conf.clone();
        c2.dims = vec![d0, d1];
        c2.num = d0 * d1;
        let r = c * d0 * d1..(c + 1) * d0 * d1;
        mdz_compress_2d(c2, &planar[r.clone()], &mut out[r], batch_size, method, report)?;
    }
    for f in 0..d0 {
        for a in 0..d1 {
            for c in 0..d2 {
                dec[f * d1 * d2 + a * d2 + c] = out[c * d0 * d1 + f * d1 + a];
            }
        }
    }
    Ok(())
}
