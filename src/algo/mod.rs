//! Assembles the modules into SZ3's algorithms (SZAlgo*.hpp).

pub mod sample;

use crate::compressor::GenericCompressor;
use crate::config::{Config, algo, interp};
use crate::decomposition::{
    BlockwiseDecomposition, Decomposition, InterpolationDecomposition, NoPredictionDecomposition,
};
use crate::encoder::{Encoder, HuffmanEncoder};
use crate::error::{Error, Result};
use crate::lossless::{Lossless, Zstd};
use crate::predictor::{ComposedPredictor, LorenzoPredictor, Predictor, RegressionPredictor};
use crate::quantizer::LinearQuantizer;
use crate::stats::{cal_abs_error_bound, factorial};
use crate::types::SzFloat;

type Q<T> = LinearQuantizer<T>;

fn quantizer<T: SzFloat>(conf: &Config) -> Q<T> {
    LinearQuantizer::new(conf.abs_error_bound, conf.quantbin_cnt / 2)
}

// ---------------------------------------------------------------- no prediction

pub fn compress_nopred<T: SzFloat>(conf: &mut Config, data: &mut [T], out: &mut Vec<u8>, cap: usize) -> Result<usize> {
    cal_abs_error_bound(conf, data, None)?;
    let mut c = GenericCompressor::new(
        NoPredictionDecomposition::new(quantizer::<T>(conf)),
        HuffmanEncoder::new(),
        Zstd::default(),
    );
    c.compress(conf, data, out, cap)
}

pub fn decompress_nopred<T: SzFloat>(conf: &Config, cmp: &[u8], out: &mut [T]) -> Result<()> {
    let mut c = GenericCompressor::new(
        NoPredictionDecomposition::new(quantizer::<T>(conf)),
        HuffmanEncoder::new(),
        Zstd::default(),
    );
    c.decompress(conf, cmp, out)
}

// ---------------------------------------------------------------- Lorenzo + regression

/// The blockwise decomposition for `ALGO_LORENZO_REG`: one predictor, or a composed one choosing
/// per block among the enabled predictors.
pub fn lorenzo_regression<T: SzFloat, const N: usize, E: Encoder + Default + 'static>(
    conf: &Config,
    q: Q<T>,
) -> Result<Box<dyn Decomposition<T>>> {
    let eb = conf.abs_error_bound;
    let mut preds: Vec<Box<dyn Predictor<T, N>>> = Vec::new();
    if conf.lorenzo {
        preds.push(Box::new(LorenzoPredictor::<T, N, 1>::new(eb)));
    }
    if conf.lorenzo2 {
        preds.push(Box::new(LorenzoPredictor::<T, N, 2>::new(eb)));
    }
    if conf.regression {
        if conf.block_size <= 0 {
            return Err(Error::InvalidArgument("block size must be positive".into()));
        }
        preds.push(Box::new(RegressionPredictor::<T, N, E>::new(
            conf.block_size as usize,
            eb,
        )));
    }
    Ok(match preds.len() {
        0 => {
            return Err(Error::InvalidArgument(
                "All lorenzo and regression methods are disabled.".into(),
            ));
        }
        1 => {
            // A lone predictor is used directly (statically dispatched in SZ3), not composed.
            let p = preds.pop().unwrap();
            Box::new(BlockwiseDecomposition::<T, N, _, _>::new(conf, BoxedPredictor(p), q))
        }
        _ => Box::new(BlockwiseDecomposition::<T, N, _, _>::new(
            conf,
            ComposedPredictor::<T, N, E>::new(preds),
            q,
        )),
    })
}

/// A boxed predictor used as a predictor.
pub struct BoxedPredictor<T: SzFloat, const N: usize>(pub Box<dyn Predictor<T, N>>);

impl<T: SzFloat, const N: usize> Predictor<T, N> for BoxedPredictor<T, N> {
    fn precompress(&mut self, b: &crate::blockwise::Block<N>, d: &[T]) -> bool {
        self.0.precompress(b, d)
    }
    fn precompress_block_commit(&mut self) {
        self.0.precompress_block_commit()
    }
    fn predecompress(&mut self, b: &crate::blockwise::Block<N>) -> Result<bool> {
        self.0.predecompress(b)
    }
    #[inline(always)]
    fn predict(&self, d: &[T], p: usize, i: &[usize; N], s: &[usize; N]) -> T {
        self.0.predict(d, p, i, s)
    }
    fn estimate_error(&self, d: &[T], p: usize, i: &[usize; N], s: &[usize; N]) -> T {
        self.0.estimate_error(d, p, i, s)
    }
    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        self.0.save(out)
    }
    fn load(&mut self, r: &mut crate::bytes::Reader, max_blocks: usize) -> Result<()> {
        self.0.load(r, max_blocks)
    }
    fn padding(&self) -> usize {
        self.0.padding()
    }
    fn size_est(&self) -> usize {
        self.0.size_est()
    }
}

pub fn compress_lorenzo_reg<T: SzFloat, const N: usize>(
    conf: &mut Config,
    data: &mut [T],
    out: &mut Vec<u8>,
    cap: usize,
) -> Result<usize> {
    cal_abs_error_bound(conf, data, None)?;
    let d = lorenzo_regression::<T, N, HuffmanEncoder>(conf, quantizer(conf))?;
    GenericCompressor::new(d, HuffmanEncoder::new(), Zstd::default()).compress(conf, data, out, cap)
}

/// `E` is the Huffman coder of the stream's data version.
pub fn decompress_lorenzo_reg<T: SzFloat, const N: usize, E: Encoder + Default + 'static>(
    conf: &Config,
    cmp: &[u8],
    out: &mut [T],
) -> Result<()> {
    let d = lorenzo_regression::<T, N, E>(conf, LinearQuantizer::default())?;
    GenericCompressor::new(d, E::default(), Zstd::default()).decompress(conf, cmp, out)
}

// ---------------------------------------------------------------- interpolation

const ANCHOR_STRIDES: [i32; 4] = [4096, 128, 32, 16];

pub fn compress_interp<T: SzFloat, const N: usize>(
    conf: &mut Config,
    data: &mut [T],
    out: &mut Vec<u8>,
    cap: usize,
) -> Result<usize> {
    cal_abs_error_bound(conf, data, None)?;
    if conf.interp_anchor_stride < 0 {
        conf.interp_anchor_stride = ANCHOR_STRIDES[N - 1];
    }
    let d = InterpolationDecomposition::<T, N, _>::new(quantizer::<T>(conf));
    GenericCompressor::new(d, HuffmanEncoder::new(), Zstd::default()).compress(conf, data, out, cap)
}

pub fn decompress_interp<T: SzFloat, const N: usize, E: Encoder + Default>(
    conf: &Config,
    cmp: &[u8],
    out: &mut [T],
) -> Result<()> {
    let d = InterpolationDecomposition::<T, N, _>::new(quantizer::<T>(conf));
    GenericCompressor::new(d, E::default(), Zstd::default()).decompress(conf, cmp, out)
}

/// Compression ratio of a decomposition over the sampled blocks, one decomposition reused for all
/// blocks as SZ3 does (its unpredictable values accumulate).
fn sampled_ratio<T: SzFloat, D: Decomposition<T>>(
    d: &mut D,
    blocks: &[Vec<T>],
    conf: &Config,
    state_num: Option<i32>,
    cap: usize,
) -> Result<f64> {
    let mut bins = Vec::new();
    for b in blocks {
        let mut cur = b.clone();
        bins.extend(d.compress(conf, &mut cur)?);
    }
    let mut enc = HuffmanEncoder::new();
    enc.preprocess_encode(&bins, state_num.unwrap_or(d.out_range().1));
    let mut buf = Vec::new();
    d.save(&mut buf)?;
    enc.save(&mut buf);
    crate::bytes::ByteWriter::put_u64(&mut buf, bins.len() as u64);
    enc.encode(&bins, &mut buf)?;
    let mut out = Vec::new();
    let n = Zstd::default().compress(&buf, &mut out, cap)?;
    Ok((conf.num * blocks.len() * T::SIZE) as f64 / n as f64)
}

fn interp_test<T: SzFloat, const N: usize>(blocks: &[Vec<T>], conf: &Config, cap: usize) -> Result<f64> {
    let mut d = InterpolationDecomposition::<T, N, _>::new(quantizer::<T>(conf));
    sampled_ratio(&mut d, blocks, conf, None, cap)
}

fn lorenzo_test<T: SzFloat, const N: usize>(blocks: &[Vec<T>], conf: &Config, cap: usize) -> Result<f64> {
    let preds: Vec<Box<dyn Predictor<T, N>>> = vec![
        Box::new(LorenzoPredictor::<T, N, 1>::new(conf.abs_error_bound)),
        Box::new(LorenzoPredictor::<T, N, 2>::new(conf.abs_error_bound)),
    ];
    let mut d = BlockwiseDecomposition::<T, N, _, _>::new(
        conf,
        ComposedPredictor::<T, N, HuffmanEncoder>::new(preds),
        quantizer::<T>(conf),
    );
    sampled_ratio(&mut d, blocks, conf, Some(conf.quantbin_cnt), cap)
}

/// `ALGO_INTERP_LORENZO`: tunes interpolation (kernel, direction, level error bounds) on sampled
/// blocks, and for 1-D data tries Lorenzo, then compresses with the winner.
pub fn compress_interp_lorenzo<T: SzFloat, const N: usize>(
    conf: &mut Config,
    data: &mut [T],
    out: &mut Vec<u8>,
    cap: usize,
) -> Result<usize> {
    cal_abs_error_bound(conf, data, None)?;
    if conf.interp_anchor_stride < 0 {
        conf.interp_anchor_stride = ANCHOR_STRIDES[N - 1];
    }
    let sample_rate = 0.005;
    let mut sbs: usize = [4096, 128, 32, 16][N - 1];
    let shortest = conf.dims.iter().copied().min().unwrap();
    while sbs >= shortest {
        sbs /= 2;
    }
    while sbs >= 16 && ((sbs + 1) as f64).powf(N as f64) / conf.num as f64 > 1.5 * sample_rate {
        sbs /= 2;
    }
    if sbs < 8 {
        sbs = 8;
    }
    let mut to_tune = ((sbs + 1) as f64).powf(N as f64) <= 0.05 * conf.num as f64;
    if conf.dims.iter().any(|&d| d < sbs) {
        to_tune = false;
    }
    if !to_tune {
        conf.cmpr_algo = algo::INTERP;
        return compress_interp::<T, N>(conf, data, out, cap);
    }
    let per_block = ((sbs + 1) as f64).powf(N as f64) as usize;
    let starts = sample::profiling_block::<T, N>(data, &conf.dims, sbs, conf.abs_error_bound, sbs / 4);
    let profiling = (starts.len() * per_block) as f64 >= 0.5 * sample_rate * conf.num as f64;
    let blocks = sample::sample_blocks::<T, N>(data, &conf.dims, sbs, sample_rate, profiling, &starts);
    let sampling_num = blocks.len() * per_block;
    if sampling_num == 0 || sampling_num as f64 >= conf.num as f64 * 0.2 {
        conf.cmpr_algo = algo::INTERP;
        return compress_interp::<T, N>(conf, data, out, cap);
    }

    let cap_test = conf.num * T::SIZE;
    let mut best_lorenzo = 0f64;
    let mut best_interp = 0f64;
    let mut lorenzo_conf = conf.clone();
    {
        conf.interp_direction = 0;
        conf.interp_alpha = 1.25;
        conf.interp_beta = 2.0;
        let mut test = conf.clone();
        test.set_dims(&vec![sbs + 1; N]);
        for op in [interp::LINEAR, interp::CUBIC] {
            test.interp_algo = op;
            let r = interp_test::<T, N>(&blocks, &test, cap_test)?;
            if r > best_interp {
                best_interp = r;
                conf.interp_algo = op;
            }
        }
        test.interp_algo = conf.interp_algo;
        test.interp_direction = (factorial(N) - 1) as u8;
        let r = interp_test::<T, N>(&blocks, &test, cap_test)?;
        if r > best_interp * 1.02 {
            best_interp = r;
            conf.interp_direction = test.interp_direction;
        }
        test.interp_direction = conf.interp_direction;
        for (alpha, beta) in [(1.0, 1.0), (1.5, 2.5), (2.0, 3.0)] {
            test.interp_alpha = alpha;
            test.interp_beta = beta;
            let r = interp_test::<T, N>(&blocks, &test, cap_test)?;
            if r > best_interp * 1.02 {
                best_interp = r;
                conf.interp_alpha = alpha;
                conf.interp_beta = beta;
            }
        }
    }
    if N == 1 && best_interp < 50.0 {
        lorenzo_conf.cmpr_algo = algo::LORENZO_REG;
        lorenzo_conf.set_dims(&vec![sbs + 1; N]);
        lorenzo_conf.lorenzo = true;
        lorenzo_conf.lorenzo2 = true;
        lorenzo_conf.regression = false;
        lorenzo_conf.regression2 = false;
        lorenzo_conf.openmp = false;
        lorenzo_conf.block_size = 5;
        best_lorenzo = lorenzo_test::<T, N>(&blocks, &lorenzo_conf, cap_test)?;
    }
    let use_interp = !(best_lorenzo >= best_interp * 1.1 && best_lorenzo < 50.0 && best_interp < 50.0);
    if use_interp {
        conf.cmpr_algo = algo::INTERP;
        compress_interp::<T, N>(conf, data, out, cap)
    } else {
        if conf.rel_error_bound < 1.01e-6 && best_lorenzo > 5.0 && lorenzo_conf.quantbin_cnt != 16384 {
            let quant_num = lorenzo_conf.quantbin_cnt;
            lorenzo_conf.quantbin_cnt = 16384;
            let r = lorenzo_test::<T, N>(&blocks, &lorenzo_conf, cap_test)?;
            if r > best_lorenzo * 1.02 {
                best_lorenzo = r;
            } else {
                lorenzo_conf.quantbin_cnt = quant_num;
            }
        }
        let _ = best_lorenzo;
        let dims = conf.dims.clone();
        lorenzo_conf.set_dims(&dims);
        *conf = lorenzo_conf;
        compress_lorenzo_reg::<T, N>(conf, data, out, cap)
    }
}
