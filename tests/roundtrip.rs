use sz3::config::{algo, eb};
use sz3::{Config, compress, compress_with, decompress};

fn field(dims: &[usize]) -> Vec<f64> {
    let n: usize = dims.iter().product();
    let mut seed: u64 = 42;
    (0..n)
        .map(|i| {
            let mut rem = i;
            let mut v = 0.0;
            for (k, &d) in dims.iter().enumerate().rev() {
                let x = (rem % d) as f64 / d as f64;
                rem /= d;
                v += ((3 + 2 * k) as f64 * std::f64::consts::PI * x + k as f64).sin();
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v + ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.02
        })
        .collect()
}

const SHAPES: &[&[usize]] = &[&[1000], &[5000], &[60, 70], &[17, 33, 40], &[6, 9, 11, 13], &[3]];
const ALGOS: &[u8] = &[
    algo::INTERP_LORENZO,
    algo::INTERP,
    algo::LORENZO_REG,
    algo::NOPRED,
    algo::LOSSLESS,
];

fn check<T>(dims: &[usize], a: u8, data: &[T], bound: f64)
where
    T: sz3::SzData + PartialEq,
{
    let mut conf = Config::new(dims);
    conf.cmpr_algo = a;
    conf.error_bound_mode = eb::ABS;
    conf.abs_error_bound = bound;
    let bytes = compress(&conf, data).unwrap();
    let (c, out) = decompress::<T>(&bytes).unwrap();
    assert_eq!(c.num, data.len());
    for (x, y) in data.iter().zip(&out) {
        let err = (x.to_f64() - y.to_f64()).abs();
        assert!(err <= bound, "{dims:?} algo {a}: error {err} > {bound}");
    }
}

#[test]
fn error_bound_holds_for_floats() {
    for dims in SHAPES {
        let d = field(dims);
        let f: Vec<f32> = d.iter().map(|&x| x as f32).collect();
        for &a in ALGOS {
            check(dims, a, &f, 1e-3);
            check(dims, a, &d, 1e-6);
        }
    }
}

#[test]
fn error_bound_holds_for_integers() {
    for dims in SHAPES {
        let d = field(dims);
        let i: Vec<i32> = d.iter().map(|&x| (x * 1000.0).round() as i32).collect();
        let l: Vec<i64> = d.iter().map(|&x| (x * 1e6).round() as i64).collect();
        let u: Vec<u16> = d.iter().map(|&x| ((x + 5.0) * 1000.0).round() as u16).collect();
        for &a in ALGOS {
            check(dims, a, &i, 3.0);
            check(dims, a, &l, 100.0);
            check(dims, a, &u, 0.0);
        }
    }
}

#[test]
fn relative_and_combined_bounds() {
    let dims = [40, 50, 60];
    let d: Vec<f32> = field(&dims).iter().map(|&x| x as f32).collect();
    let (mn, mx) = d.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let range = (mx - mn) as f64;
    for (mode, rel, abs, want) in [
        (eb::REL, 1e-4, 0.0, 1e-4 * range),
        (eb::ABS_AND_REL, 1e-4, 1e-6, 1e-6),
        (eb::ABS_OR_REL, 1e-4, 1e-6, 1e-4 * range),
    ] {
        let mut conf = Config::new(&dims);
        conf.error_bound_mode = mode;
        conf.rel_error_bound = rel;
        conf.abs_error_bound = abs;
        let bytes = compress_with(&mut conf, &d).unwrap();
        assert_eq!(conf.error_bound_mode, eb::ABS);
        assert!((conf.abs_error_bound - want).abs() <= want * 1e-6);
        let (_, out) = decompress::<f32>(&bytes).unwrap();
        let max = d
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        assert!(max <= want, "mode {mode}: {max} > {want}");
    }
}

#[test]
fn special_values_round_trip() {
    let mut d: Vec<f32> = field(&[2000]).iter().map(|&x| x as f32).collect();
    d[10] = f32::NAN;
    d[20] = f32::INFINITY;
    d[30] = f32::NEG_INFINITY;
    d[40] = 1e30;
    for &a in &[algo::INTERP, algo::LORENZO_REG, algo::NOPRED] {
        let mut conf = Config::new(&[2000]);
        conf.cmpr_algo = a;
        conf.abs_error_bound = 1e-3;
        let (_, out) = decompress::<f32>(&compress(&conf, &d).unwrap()).unwrap();
        assert!(out[10].is_nan());
        assert_eq!(out[20], f32::INFINITY);
        assert_eq!(out[30], f32::NEG_INFINITY);
        for i in (0..2000).filter(|i| ![10, 20, 30].contains(i)) {
            assert!(
                (out[i] - d[i]).abs() as f64 <= 1e-3 * (1.0 + d[i].abs() as f64 * 1e-6),
                "{i}"
            );
        }
    }
}

#[test]
fn constant_and_zero_bound_go_lossless() {
    let d = vec![3.5f32; 4096];
    let conf = Config::new(&[64, 64]);
    let bytes = compress(&conf, &d).unwrap();
    assert_eq!(decompress::<f32>(&bytes).unwrap().1, d);
    let mut conf = Config::new(&[64, 64]);
    conf.abs_error_bound = 0.0;
    let v: Vec<f32> = field(&[64, 64]).iter().map(|&x| x as f32).collect();
    let mut c2 = conf.clone();
    let bytes = compress_with(&mut c2, &v).unwrap();
    assert_eq!(c2.cmpr_algo, algo::LOSSLESS);
    assert_eq!(decompress::<f32>(&bytes).unwrap().1, v);
}

/// Damaged streams must be rejected with an error, never a panic.
#[test]
fn damaged_streams_do_not_panic() {
    let dims = [30, 40];
    let d: Vec<f32> = field(&dims).iter().map(|&x| x as f32).collect();
    for &a in ALGOS {
        let mut conf = Config::new(&dims);
        conf.cmpr_algo = a;
        let bytes = compress(&conf, &d).unwrap();
        for cut in [0, 1, 15, 16, 17, bytes.len() / 2, bytes.len() - 1] {
            let _ = decompress::<f32>(&bytes[..cut]);
        }
        let mut seed: u64 = 7;
        for _ in 0..300 {
            let mut b = bytes.clone();
            for _ in 0..3 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let i = (seed >> 33) as usize % b.len();
                b[i] ^= 1 << ((seed >> 20) % 8);
            }
            let _ = decompress::<f32>(&b);
        }
    }
}

#[test]
fn openmp_layout_round_trips() {
    for dims in [&[5000usize][..], &[60, 70], &[17, 33, 40], &[3, 5]] {
        let d: Vec<f32> = field(dims).iter().map(|&x| x as f32).collect();
        for &a in ALGOS {
            let mut conf = Config::new(dims);
            conf.cmpr_algo = a;
            conf.openmp = true;
            conf.error_bound_mode = eb::REL;
            conf.rel_error_bound = 1e-3;
            let mut c = conf.clone();
            let bytes = compress_with(&mut c, &d).unwrap();
            let (back, out) = decompress::<f32>(&bytes).unwrap();
            assert!(back.openmp);
            for (x, y) in d.iter().zip(&out) {
                assert!(((x - y).abs() as f64) <= c.abs_error_bound, "{dims:?} algo {a}");
            }
        }
    }
}
