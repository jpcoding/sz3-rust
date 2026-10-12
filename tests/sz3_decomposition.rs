//! SZ3's test_integer_data.cpp, test_reconstruction.cpp and test_decomposition_save_load.cpp.

mod common;

use common::{Mt32, Mt64};
use sz3::algo;
use sz3::bytes::Reader;
use sz3::config::{algo as A, eb, interp};
use sz3::decomposition::{BioMdDecomposition, BlockwiseDecomposition, Decomposition};
use sz3::encoder::HuffmanEncoder;
use sz3::predictor::{ComposedPredictor, LorenzoPredictor, Predictor, RegressionPredictor};
use sz3::quantizer::LinearQuantizer;
use sz3::types::SzFloat;
use sz3::{Config, Error, SzData, compress, compress_size_bound, decompress};

// ---------------------------------------------------------------- test_integer_data.cpp

/// Every decompressed integer lies within floor(bound) of the input, including values at the
/// type's limits and data whose range the type cannot hold.
macro_rules! run_integers {
    ($t:ty, $shift:expr) => {{
        let (lo, hi) = (<$t>::MIN, <$t>::MAX);
        let mut g = Mt64::new(7);
        let mut inputs = vec![Vec::<$t>::with_capacity(20000); 3];
        for i in 0..20000u64 {
            inputs[0].push(g.next_u64() as $t); // full range
            inputs[1].push(if g.next_u64() & 1 != 0 {
                // both ends
                hi - (g.next_u64() % 4) as $t
            } else {
                lo + (g.next_u64() % 4) as $t
            });
            inputs[2].push(hi - ((i * 7) % 50) as $t); // smooth, at the top
        }
        // 8-byte values shifted to within +-2^53
        for v in inputs.iter_mut().flatten() {
            *v >>= $shift;
        }
        for input in &inputs {
            check_integers(input);
        }
    }};
}

fn check_integers<T: SzData>(input: &[T]) {
    let (mn, mx) = input
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x.to_f64()), b.max(x.to_f64())));
    for a in [A::LORENZO_REG, A::INTERP_LORENZO, A::INTERP, A::NOPRED] {
        for (mode, e) in [(eb::ABS, 0.0), (eb::ABS, 1.0), (eb::ABS, 3.7), (eb::REL, 1e-2)] {
            let mut conf = Config::new(&[input.len()]);
            conf.cmpr_algo = a;
            conf.error_bound_mode = mode;
            conf.abs_error_bound = e;
            conf.rel_error_bound = e;
            let (_, out) = decompress::<T>(&compress(&conf, input).unwrap()).unwrap();
            let bound = if mode == eb::ABS { e } else { e * (mx - mn) }.floor();
            for (i, (x, y)) in input.iter().zip(&out).enumerate() {
                let err = (y.to_f64() - x.to_f64()).abs();
                assert!(err <= bound, "algo {a} mode {mode} bound {e} at {i}: {err}");
            }
        }
    }
}

// SZ3_IntegerData.WithinTheBound
#[test]
fn integer_data_within_the_bound() {
    run_integers!(i8, 0);
    run_integers!(u8, 0);
    run_integers!(i16, 0);
    run_integers!(u16, 0);
    run_integers!(i32, 0);
    run_integers!(u32, 0);
    run_integers!(i64, 12);
    run_integers!(u64, 12);
}

// SZ3_IntegerData.EightByteValuesPast2To53AreRefused
#[test]
fn integer_data_eight_byte_values_past_2_to_53_are_refused() {
    let input = vec![1i64 << 53; 100];
    let conf = Config::new(&[input.len()]);
    assert!(matches!(compress(&conf, &input), Err(Error::InvalidArgument(_))));
}

// ---------------------------------------------------------------- test_reconstruction.cpp
//
// The compressor predicts from its own reconstructed values, so the decompressor must reconstruct
// exactly the same values. ALGO_BIOMD leaves the input as it was and predicts on its integer
// lattice: the values it decodes, compressed again, must decode to themselves.

fn differing_values<T: SzFloat + SzData>(a: u8, interp_algo: u8, bound: f64) -> usize {
    let mut conf = Config::new(if a == A::BIOMD { &[32, 341, 3] } else { &[32, 32, 32] });
    conf.cmpr_algo = a;
    conf.interp_algo = interp_algo;
    conf.error_bound_mode = eb::ABS;
    conf.abs_error_bound = bound;
    let mut data: Vec<T> = (0..conf.num)
        .map(|i| {
            let (x, y, z) = ((i / 1024) as f64, (i / 32 % 32) as f64, (i % 32) as f64);
            T::from_f64((0.3 * x).sin() * (0.2 * y).cos() + 0.5 * (0.1 * z + 0.05 * x).sin())
        })
        .collect();
    let mut out = vec![T::ZERO; conf.num];
    let cap = compress_size_bound::<T>(&conf);
    let mut cmp = Vec::new();
    if a == A::INTERP {
        algo::compress_interp::<T, 3>(&mut conf, &mut data, &mut cmp, cap).unwrap();
        algo::decompress_interp::<T, 3, HuffmanEncoder>(&conf, &cmp, &mut out).unwrap();
    } else {
        algo::compress_biomd::<T>(&mut conf, &data, &mut cmp, cap).unwrap();
        if conf.cmpr_algo != A::BIOMD {
            return 0; // beyond its lattice: stored losslessly
        }
        algo::decompress_biomd::<T>(&conf, &cmp, &mut data).unwrap();
        cmp.clear();
        algo::compress_biomd::<T>(&mut conf, &data, &mut cmp, cap).unwrap();
        algo::decompress_biomd::<T>(&conf, &cmp, &mut out).unwrap();
    }
    data.iter().zip(&out).filter(|(x, y)| x.bits() != y.bits()).count()
}

fn expect_identical<T: SzFloat + SzData>(a: u8, interp_algo: u8) {
    for bound in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] {
        assert_eq!(differing_values::<T>(a, interp_algo, bound), 0, "eb = {bound}");
    }
}

// SZ3_Reconstruction.InterpLinear
#[test]
fn reconstruction_interp_linear() {
    expect_identical::<f32>(A::INTERP, interp::LINEAR);
    expect_identical::<f64>(A::INTERP, interp::LINEAR);
}

// SZ3_Reconstruction.InterpCubic
#[test]
fn reconstruction_interp_cubic() {
    expect_identical::<f32>(A::INTERP, interp::CUBIC);
    expect_identical::<f64>(A::INTERP, interp::CUBIC);
}

// SZ3_Reconstruction.BioMD
#[test]
fn reconstruction_biomd() {
    expect_identical::<f32>(A::BIOMD, interp::CUBIC);
    expect_identical::<f64>(A::BIOMD, interp::CUBIC);
}

// SZ3_Reconstruction.SZCompressLeavesTheInputAsItWas (compress takes the input by shared
// reference, so this holds by construction; kept for each predictor combination as in SZ3)
#[test]
fn reconstruction_compress_leaves_the_input_as_it_was() {
    let cases = [
        (A::LORENZO_REG, true, false, true),
        (A::LORENZO_REG, true, false, false),
        (A::LORENZO_REG, false, true, false),
        (A::LORENZO_REG, false, false, true),
        (A::LORENZO_REG, true, true, true),
        (A::INTERP_LORENZO, true, false, true),
        (A::INTERP, true, false, true),
        (A::NOPRED, true, false, true),
        (A::BIOMD, true, false, true),
    ];
    for (a, lorenzo, lorenzo2, regression) in cases {
        for dims in [&[32, 341, 3][..], &[5000], &[70, 73], &[5, 6, 7, 9]] {
            if a == A::BIOMD && *dims.last().unwrap() != 3 {
                continue; // coordinates only
            }
            let mut conf = Config::new(dims);
            conf.cmpr_algo = a;
            (conf.lorenzo, conf.lorenzo2, conf.regression) = (lorenzo, lorenzo2, regression);
            conf.error_bound_mode = eb::ABS;
            conf.abs_error_bound = 1e-3;
            let data: Vec<f32> = (0..conf.num)
                .map(|i| ((0.01 * i as f64).sin() + 0.1 * (i % 3) as f64) as f32)
                .collect();
            let before = data.clone();
            compress(&conf, &data).unwrap();
            assert!(
                before.iter().zip(&data).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{a} {lorenzo}{lorenzo2}{regression} {}D",
                dims.len()
            );
        }
    }
}

// SZCompress.BoundsThatAreNotPositiveAndFiniteGoLossless: an error bound that is negative or not
// finite leaves nothing to bound; every algorithm compresses losslessly, as with a bound of 0.
#[test]
fn bounds_that_are_not_positive_and_finite_go_lossless() {
    for a in [A::LORENZO_REG, A::INTERP_LORENZO, A::INTERP, A::NOPRED, A::BIOMD] {
        for bound in [0.0, -1e-3, f64::INFINITY, f64::NAN] {
            let mut conf = Config::new(&[40, 30, 3]);
            conf.cmpr_algo = a;
            conf.error_bound_mode = eb::ABS;
            conf.abs_error_bound = bound;
            let data: Vec<f32> = (0..conf.num).map(|i| (0.01 * i as f64).sin() as f32).collect();
            let (dec, out) = decompress::<f32>(&compress(&conf, &data).unwrap()).unwrap();
            assert_eq!(dec.cmpr_algo, A::LOSSLESS, "{a} {bound}");
            assert!(
                out.iter().zip(&data).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{a} {bound}"
            );
        }
    }
}

// ---------------------------------------------------------------- test_decomposition_save_load.cpp
//
// save() has to write the same bytes for the same input, and load() has to consume exactly what
// save() wrote. SZ3 builds the decomposition over poisoned memory to catch members save() writes
// but compress() never assigns; Rust has no uninitialized members, so the closest observable check
// is that a decomposition reused after compressing other data writes what a fresh one writes.

fn noise(count: usize, seed: u32) -> Vec<f32> {
    let mut g = Mt32::new(seed);
    (0..count).map(|_| g.uniform_f32(-5.0, 5.0)).collect()
}

/// A ramp is what a regression predictor fits best, so it is also what makes it store the most.
fn ramp(count: usize) -> Vec<f32> {
    let mut g = Mt32::new(3);
    (0..count).map(|i| i as f32 * 0.5 + g.uniform_f32(-1.0, 1.0)).collect()
}

fn abs_conf(dims: &[usize]) -> Config {
    let mut conf = Config::new(dims);
    conf.error_bound_mode = eb::ABS;
    conf.abs_error_bound = 1e-3;
    conf
}

fn biomd_header(d: &mut BioMdDecomposition<f32>, conf: &Config) -> Vec<u8> {
    let mut data = noise(conf.num, 11);
    d.compress(conf, &mut data).unwrap();
    let mut header = Vec::new();
    d.save(&mut header).unwrap();
    header
}

fn expect_header_depends_only_on_input(dims: &[usize]) {
    let conf = abs_conf(dims);
    let fresh = biomd_header(&mut BioMdDecomposition::new(&conf).unwrap(), &conf);
    let again = biomd_header(&mut BioMdDecomposition::new(&conf).unwrap(), &conf);
    assert_eq!(fresh, again);
    let mut used = BioMdDecomposition::new(&conf).unwrap();
    let mut other: Vec<f32> = ramp(conf.num).iter().map(|v| v * 0.002).collect();
    used.compress(&conf, &mut other).unwrap();
    assert_eq!(
        biomd_header(&mut used, &conf),
        fresh,
        "save() wrote bytes left from an earlier compress()"
    );
}

// SZ3_DecompositionSaveLoad.BioMDTwoDimensions
#[test]
fn biomd_two_dimensions() {
    expect_header_depends_only_on_input(&[777, 3]);
}

// SZ3_DecompositionSaveLoad.BioMDThreeDimensions
#[test]
fn biomd_three_dimensions() {
    expect_header_depends_only_on_input(&[5, 777, 3]);
}

// SZ3_DecompositionSaveLoad.BioMDChargesWhatItReads
#[test]
fn biomd_charges_what_it_reads() {
    for dims in [&[777, 3][..], &[5, 777, 3]] {
        let conf = abs_conf(dims);
        let mut stream = biomd_header(&mut BioMdDecomposition::new(&conf).unwrap(), &conf);
        let written = stream.len();
        stream.resize(written + 4096, 0);
        let mut reader = BioMdDecomposition::<f32>::new(&conf).unwrap();
        let mut r = Reader::new(&stream);
        reader.load(&mut r, &conf).unwrap();
        assert_eq!(r.position(), written, "load() did not consume what save() wrote");
        assert_eq!(stream.len() - r.remaining(), written);
    }
}

// SZ3_DecompositionSaveLoad.BioMDRefusesUnknownModeAndBoxFlag: the mode bytes and the box flag
// hold 0 or 1; any other value is a corrupt stream.
#[test]
fn biomd_refuses_unknown_mode_and_box_flag() {
    let conf = abs_conf(&[5, 777, 3]);
    let header = biomd_header(&mut BioMdDecomposition::new(&conf).unwrap(), &conf);
    // coded frames (8), fill value (4), step (8), water distances (8 + 8), water count (8), then 4
    // modes and the flag
    let (modes, boxflag) = (44, 44 + 4);
    for at in [modes, modes + 3, boxflag] {
        let mut corrupt = header.clone();
        assert!(
            corrupt[at] <= 1,
            "byte {at} is not where the mode bytes and the box flag are"
        );
        corrupt[at] = 2;
        let mut reader = BioMdDecomposition::<f32>::new(&conf).unwrap();
        assert!(
            matches!(reader.load(&mut Reader::new(&corrupt), &conf), Err(Error::Corrupt(_))),
            "byte {at}"
        );
    }
}

/// The compressor allocates from size_est(): save() must stay within it.
fn expect_save_stays_within_size_est<D: Decomposition<f32>>(d: &mut D, conf: &Config, mut data: Vec<f32>) {
    d.compress(conf, &mut data).unwrap();
    let declared = d.size_est();
    let mut buf = Vec::new();
    d.save(&mut buf).unwrap();
    assert!(
        buf.len() <= declared,
        "save() wrote {} bytes where size_est() promised {declared}",
        buf.len()
    );
}

// SZ3_DecompositionSaveLoad.BlockwiseSizeEstBoundsSaveAtSmallBlockSizes: the regression
// coefficients grow as n / blockSize.
#[test]
fn blockwise_size_est_bounds_save_at_small_block_sizes() {
    for block_size in [2, 3, 4, 6, 8, 128] {
        for bound in [1e-1, 1e-2, 1e-3, 1e-4] {
            let mut conf = Config::new(&[16384]);
            conf.error_bound_mode = eb::ABS;
            conf.abs_error_bound = bound;
            conf.block_size = block_size;
            let preds: Vec<Box<dyn Predictor<f32, 1>>> = vec![
                Box::new(LorenzoPredictor::<f32, 1, 1>::new(bound)),
                Box::new(RegressionPredictor::<f32, 1>::new(block_size as usize, bound)),
            ];
            let mut d = BlockwiseDecomposition::<f32, 1, _, _>::new(
                &conf,
                ComposedPredictor::<f32, 1>::new(preds),
                LinearQuantizer::<f32>::new(bound, conf.quantbin_cnt / 2),
            );
            expect_save_stays_within_size_est(&mut d, &conf, ramp(conf.num));
        }
    }
}

/// A block RegressionPredictor declines falls back to Lorenzo, which reads a neighbour outside the
/// block.
fn expect_regression_only_round_trip<const N: usize>(dims: &[usize]) {
    let conf = Config::new(dims);
    let bound = 1e-3;
    let make = || {
        BlockwiseDecomposition::<f32, N, _, _>::new(
            &conf,
            RegressionPredictor::<f32, N>::new(conf.block_size as usize, bound),
            LinearQuantizer::<f32>::new(bound, conf.quantbin_cnt / 2),
        )
    };
    let input = noise(conf.num, 11);
    let mut data = input.clone();
    let mut writer = make();
    let bins = writer.compress(&conf, &mut data).unwrap();
    let mut stream = Vec::new();
    writer.save(&mut stream).unwrap();
    let mut reader = make();
    reader.load(&mut Reader::new(&stream), &conf).unwrap();
    let mut output = vec![0f32; conf.num];
    reader.decompress(&conf, &bins, &mut output).unwrap();
    for (i, (x, y)) in input.iter().zip(&output).enumerate() {
        assert!(((x - y).abs() as f64) <= bound, "{N}D, value {i}");
    }
}

// SZ3_DecompositionSaveLoad.BlockwiseRegressionOnlyFallback
#[test]
fn blockwise_regression_only_fallback() {
    expect_regression_only_round_trip::<3>(&[40, 41, 43]);
    expect_regression_only_round_trip::<4>(&[12, 13, 14, 15]);
}

// SZ3_DecompositionSaveLoad.BioMDSizeEstBoundsSave
#[test]
fn biomd_size_est_bounds_save() {
    for bound in [1e-1, 1e-2, 1e-3, 1e-4] {
        let mut conf = Config::new(&[5, 777, 3]);
        conf.error_bound_mode = eb::ABS;
        conf.abs_error_bound = bound;
        let mut d = BioMdDecomposition::<f32>::new(&conf).unwrap();
        // up to 12 nm, within BIOMD's lattice (64 nm at 1e-4)
        let coordinates = ramp(conf.num).iter().map(|v| v * 0.002).collect();
        expect_save_stays_within_size_est(&mut d, &conf, coordinates);
    }
}
