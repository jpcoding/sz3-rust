//! SZ3's test_biomd.cpp and test_biomd_codec.cpp: ALGO_BIOMD through the public API, on water-box
//! trajectories and synthetic molecular-dynamics systems (rigid 3- and 4-site water, bonded chains
//! and ions, as GROMACS lays them out): the bound, fill frames, the input it refuses,
//! recompression, and corrupt or truncated streams, which are refused without panicking.

// Literals are written as SZ3's tests write them.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

mod common;

use common::{Mt32, Normal, max_abs_error};
use sz3::config::{algo as A, eb};
use sz3::{Config, Error, SzData, compress, decompress};

// ---------------------------------------------------------------- test_biomd.cpp

/// A water-box trajectory: atoms placed on a jittered lattice, then walking a little each frame.
fn make_trajectory(frames: usize, atoms: usize) -> Vec<f32> {
    let mut g = Mt32::new(7);
    let side = ((atoms as f64).cbrt() + 1.0) as usize;
    let mut pos = vec![0f32; atoms * 3];
    for a in 0..atoms {
        pos[a * 3] = (a % side) as f32 * 0.31;
        pos[a * 3 + 1] = ((a / side) % side) as f32 * 0.31;
        pos[a * 3 + 2] = (a / (side * side)) as f32 * 0.31;
    }
    let mut traj = Vec::with_capacity(frames * atoms * 3);
    for _ in 0..frames {
        for p in pos.iter_mut() {
            *p += g.uniform_f32(-0.02, 0.02);
        }
        traj.extend_from_slice(&pos);
    }
    traj
}

/// Compresses and decompresses through the public API; returns the compressed bytes and the
/// decompressed values.
fn round_trip(bound: f64, dims: &[usize], input: &[f32]) -> sz3::Result<(Vec<u8>, Vec<f32>)> {
    let mut conf = Config::new(dims);
    conf.cmpr_algo = A::BIOMD;
    conf.error_bound_mode = eb::ABS;
    conf.abs_error_bound = bound;
    let bytes = compress(&conf, input)?;
    let (_, out) = decompress::<f32>(&bytes)?;
    Ok((bytes, out))
}

// SZ3_BioMD.TrajectoryRoundTripHonoursTheBound: dims {frames, atoms, xyz} is the layout a
// trajectory arrives in.
#[test]
fn trajectory_round_trip_honours_the_bound() {
    let dims = [21, 2652, 3];
    let input = make_trajectory(dims[0], dims[1]);
    for bound in [1e-1, 1e-2, 1e-3, 1e-4] {
        let (bytes, out) = round_trip(bound, &dims, &input).unwrap();
        assert_eq!(out.len(), input.len(), "eb={bound}");
        assert!(max_abs_error(&input, &out) <= bound, "eb={bound}");
        assert!(bytes.len() < input.len() * 4, "eb={bound} did not compress");
    }
}

// SZ3_BioMD.TrailingFilledFramesAreRestored: trailing frames that hold one value are skipped on
// compression and written back on decompression.
#[test]
fn trailing_filled_frames_are_restored() {
    let dims = [16, 500, 3];
    let frame = dims[1] * dims[2];
    let written_frames = 9;
    let fill = 0.12345f32;
    let mut input = make_trajectory(dims[0], dims[1]);
    input[written_frames * frame..].fill(fill);

    let (bytes, out) = round_trip(1e-3, &dims, &input).unwrap();
    assert_eq!(out.len(), input.len());
    assert!(max_abs_error(&input, &out) <= 1e-3);
    // The filled tail is written back verbatim, not quantized.
    for (i, &v) in out.iter().enumerate().skip(written_frames * frame) {
        assert_eq!(v, fill, "at element {i}");
    }
    // Skipping the filled tail has to cost less than quantizing it.
    let (without_fill, _) = round_trip(1e-3, &dims, &make_trajectory(dims[0], dims[1])).unwrap();
    assert!(bytes.len() < without_fill.len());
}

// SZ3_BioMD.EveryFrameAfterTheFirstIsFill: the largest the skipped range can be.
#[test]
fn every_frame_after_the_first_is_fill() {
    let dims = [8, 300, 3];
    let frame = dims[1] * dims[2];
    let fill = -3.14159f32;
    let mut input = make_trajectory(dims[0], dims[1]);
    input[frame..].fill(fill);
    let (_, out) = round_trip(1e-3, &dims, &input).unwrap();
    assert_eq!(out.len(), input.len());
    assert!(max_abs_error(&input, &out) <= 1e-3);
    for (i, &v) in out.iter().enumerate().skip(frame) {
        assert_eq!(v, fill, "at element {i}");
    }
}

// SZ3_BioMD.OneFramePerChunk: Config drops a dimension of 1, so this reaches the two-dimensional
// path rather than the trajectory one.
#[test]
fn one_frame_per_chunk() {
    let input = make_trajectory(1, 1024);
    let (_, out) = round_trip(1e-3, &[1, 1024, 3], &input).unwrap();
    assert_eq!(out.len(), input.len());
    assert!(max_abs_error(&input, &out) <= 1e-3);
}

// SZ3_BioMD.TwoFrameTrajectory: the smallest trajectory that still reaches the multi-frame path.
#[test]
fn two_frame_trajectory() {
    let input = make_trajectory(2, 1024);
    let (_, out) = round_trip(1e-3, &[2, 1024, 3], &input).unwrap();
    assert_eq!(out.len(), input.len());
    assert!(max_abs_error(&input, &out) <= 1e-3);
}

/// ALGO_BIOMD takes coordinates, {..., 3}, and refuses other shapes.
fn refused(dims: &[usize]) -> bool {
    *dims.last().unwrap() != 3
}

// SZ3_BioMD.OneAndTwoDimensionalInput: the single-frame paths, which 1D and 2D data take.
#[test]
fn one_and_two_dimensional_input() {
    let atoms = make_trajectory(1, 4096);
    for dims in [&[atoms.len()][..], &[4096, 3]] {
        let r = round_trip(1e-3, dims, &atoms);
        if refused(dims) {
            assert!(matches!(r, Err(Error::InvalidArgument(_))), "{}D", dims.len());
            continue;
        }
        let (_, out) = r.unwrap();
        assert_eq!(out.len(), atoms.len(), "{}D", dims.len());
        assert!(max_abs_error(&atoms, &out) <= 1e-3, "{}D", dims.len());
    }
}

// SZ3_BioMD.InputsShorterThanTwoAtoms: refused or round-trip within the bound.
#[test]
fn inputs_shorter_than_two_atoms() {
    for n in [1, 2, 3, 5] {
        let input: Vec<f32> = (0..n).map(|i| 0.5 * i as f32 - 1.0).collect();
        let r = round_trip(1e-3, &[n], &input);
        if refused(&[n]) {
            assert!(matches!(r, Err(Error::InvalidArgument(_))), "n={n}");
            continue;
        }
        let (_, out) = r.unwrap();
        assert_eq!(out.len(), n, "n={n}");
        assert!(max_abs_error(&input, &out) <= 1e-3, "n={n}");
    }
}

// SZ3_BioMD.CompressionIsDeterministic: the same input has to produce the same file, or a
// trajectory cannot be checksummed.
#[test]
fn compression_is_deterministic() {
    for dims in [&[12288][..], &[64, 64], &[5, 777, 3]] {
        if refused(dims) {
            continue;
        }
        let (frames, atoms) = if dims.len() == 3 { (dims[0], dims[1]) } else { (1, 4096) };
        let input = make_trajectory(frames, atoms);
        let (first, _) = round_trip(1e-3, dims, &input).unwrap();
        let (second, _) = round_trip(1e-3, dims, &input).unwrap();
        assert_eq!(first, second, "{}D", dims.len());
    }
}

// SZ3_BioMD.RemovedXtcAlgorithmIsRefused: asking for ALGO_BIOMDXTC, also with a bound that would
// go lossless, says to use ALGO_BIOMD.
#[test]
fn removed_xtc_algorithm_is_refused() {
    let input = make_trajectory(2, 100);
    for bound in [1e-3, 0.0] {
        let mut conf = Config::new(&[2, 100, 3]);
        conf.cmpr_algo = A::BIOMDXTC;
        conf.error_bound_mode = eb::ABS;
        conf.abs_error_bound = bound;
        match compress(&conf, &input) {
            Err(Error::InvalidArgument(m)) => assert!(m.contains("ALGO_BIOMD"), "{m}"),
            r => panic!("ALGO_BIOMDXTC at eb={bound}: {r:?}"),
        }
    }
}

// ---------------------------------------------------------------- test_biomd_codec.cpp

struct SystemSpec {
    waters: usize,
    four_site: bool,
    chains: usize,
    chain_len: usize, // heavy atoms per chain, each with one hydrogen
    ions: usize,
    frames: usize,
    seed: u32,
}

impl Default for SystemSpec {
    fn default() -> Self {
        SystemSpec {
            waters: 800,
            four_site: false,
            chains: 2,
            chain_len: 60,
            ions: 10,
            frames: 1,
            seed: 1,
        }
    }
}

type Vec3 = [f64; 3];

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(s: f64, a: Vec3) -> Vec3 {
    [s * a[0], s * a[1], s * a[2]]
}
fn unit(a: Vec3) -> Vec3 {
    scale(1.0 / (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt(), a)
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// frames x atoms x 3 floats, in nm; returns them and the atom count.
fn make_system(s: &SystemSpec) -> (Vec<f32>, usize) {
    let mut g = Mt32::new(s.seed);
    let mut normal = Normal::default();
    let mut rvec = |g: &mut Mt32| unit([normal.sample(g), normal.sample(g), normal.sample(g)]);
    let per_water = if s.four_site { 4 } else { 3 };
    let n = s.chains * s.chain_len * 2 + s.waters * per_water + s.ions;
    let bx = (n as f64 * 0.01).cbrt();
    let in_box = |g: &mut Mt32| {
        let mut p = [0.0; 3];
        for c in &mut p {
            *c = bx * (g.uniform(-1.0, 1.0) + 1.0) / 2.0;
        }
        p
    };
    let (r, theta, a_vs) = (0.09572, 104.52 * std::f64::consts::PI / 180.0, 0.128); // TIP4P geometry
    let (bonds, bh) = ([0.1529, 0.1335, 0.1471], 0.109);
    let (mut o, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..s.waters {
        o.push(in_box(&mut g));
        let uw = rvec(&mut g);
        u.push(uw);
        v.push(unit(cross(uw, rvec(&mut g))));
    }
    let mut chain_dir: Vec<Vec3> = (0..s.chains * s.chain_len).map(|_| rvec(&mut g)).collect();
    let chain0: Vec<Vec3> = (0..s.chains).map(|_| in_box(&mut g)).collect();
    let mut ion: Vec<Vec3> = (0..s.ions).map(|_| in_box(&mut g)).collect();
    let mut x = Vec::with_capacity(s.frames * n * 3);
    for _ in 0..s.frames {
        let mut put = |p: Vec3| x.extend(p.iter().map(|&c| c as f32));
        for c in 0..s.chains {
            let mut p = chain0[c];
            for a in 0..s.chain_len {
                let d = chain_dir[c * s.chain_len + a];
                p = add(p, scale(bonds[a % 3], d));
                put(p);
                put(add(p, scale(bh, unit(cross(d, rvec(&mut g))))));
            }
        }
        for w in 0..s.waters {
            let h1 = add(o[w], scale(r, u[w]));
            let h2 = add(o[w], scale(r, add(scale(theta.cos(), u[w]), scale(theta.sin(), v[w]))));
            put(o[w]);
            put(h1);
            put(h2);
            if s.four_site {
                put(add(
                    o[w],
                    scale(a_vs, add(add(h1, scale(-1.0, o[w])), add(h2, scale(-1.0, o[w])))),
                ));
            }
        }
        for &p in &ion {
            put(p);
        }
        // the next frame: molecules move and turn a little
        for w in 0..s.waters {
            o[w] = add(o[w], scale(0.003, rvec(&mut g)));
            u[w] = unit(add(u[w], scale(0.05, rvec(&mut g))));
            v[w] = unit(cross(u[w], cross(v[w], u[w])));
        }
        for d in chain_dir.iter_mut() {
            *d = unit(add(*d, scale(0.03, rvec(&mut g))));
        }
        for p in ion.iter_mut() {
            *p = add(*p, scale(0.005, rvec(&mut g)));
        }
    }
    (x, n)
}

struct RoundTrip<T> {
    bytes: Vec<u8>,
    out: Vec<T>,
    algo: u8, // the algorithm the stream holds
    max_err: f64,
}

fn rt<T: SzData + Into<f64>>(x: &[T], frames: usize, atoms: usize, bound: f64) -> RoundTrip<T> {
    let mut conf = if frames == 1 {
        Config::new(&[atoms, 3])
    } else {
        Config::new(&[frames, atoms, 3])
    };
    conf.cmpr_algo = A::BIOMD;
    conf.error_bound_mode = eb::ABS;
    conf.abs_error_bound = bound;
    let bytes = compress(&conf, x).unwrap();
    let (dconf, out) = decompress::<T>(&bytes).unwrap();
    let max_err = max_abs_error(x, &out);
    RoundTrip {
        bytes,
        out,
        algo: dconf.cmpr_algo,
        max_err,
    }
}

// BioMD.WithinBoundAndBeatsFourBytesPerValue
#[test]
fn within_bound_and_beats_four_bytes_per_value() {
    for four_site in [false, true] {
        for frames in [1, 5, 20] {
            for bound in [5e-4, 1e-3, 1e-5, 0.1] {
                let (x, n) = make_system(&SystemSpec {
                    four_site,
                    frames,
                    ..Default::default()
                });
                let r = rt(&x, frames, n, bound);
                let at = format!("four_site={four_site} frames={frames} eb={bound}");
                assert_eq!(r.algo, A::BIOMD, "{at}");
                assert!(r.max_err <= bound, "{at}");
                if bound == 5e-4 {
                    assert!((x.len() * 4) as f64 / r.bytes.len() as f64 > 3.0, "{at}");
                }
            }
        }
    }
}

// BioMD.LooseBoundsStayWithinTheBound
#[test]
fn loose_bounds_stay_within_the_bound() {
    let s = SystemSpec {
        frames: 20,
        ..Default::default()
    };
    let (mut x, n) = make_system(&s);
    for t in 1..s.frames {
        // frames that do not move
        x.copy_within(0..3 * n, t * 3 * n);
    }
    for bound in [0.02, 0.1] {
        let r = rt(&x, s.frames, n, bound);
        assert_eq!(r.algo, A::BIOMD);
        assert!(r.max_err <= bound);
    }
}

// BioMD.DoubleInput
#[test]
fn double_input() {
    let s = SystemSpec {
        frames: 4,
        ..Default::default()
    };
    let (xf, n) = make_system(&s);
    let x: Vec<f64> = xf.iter().map(|&v| v as f64).collect();
    for bound in [5e-4, 1e-7] {
        assert!(rt(&x, s.frames, n, bound).max_err <= bound, "eb={bound}");
    }
}

// BioMD.EdgeCasesStayWithinBound
#[test]
fn edge_cases_stay_within_bound() {
    let mut g = Mt32::new(3);
    let mut cases: Vec<(String, Vec<f32>)> = Vec::new();
    for n in [1, 2, 3, 4, 5, 7] {
        let x = (0..3 * n).map(|_| g.uniform_f32(0.0, 5.0)).collect();
        cases.push((format!("tiny N={n}"), x));
    }
    cases.push(("constant".into(), vec![1.234; 3000]));
    cases.push(("zeros".into(), vec![0.0; 3000]));
    let s = SystemSpec::default();
    let (base, n) = make_system(&s);
    {
        let mut x = base.clone(); // waters split over a periodic boundary
        let mut i = s.chains * s.chain_len * 2;
        while i + 2 < n {
            x[3 * (i + 1)] += 3.1;
            i += 30;
        }
        cases.push(("broken waters".into(), x));
    }
    {
        let mut x = base.clone(); // atoms in no particular order
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, (g.next_u32() as usize) % (i + 1));
        }
        for i in 0..n {
            x[3 * i..3 * i + 3].copy_from_slice(&base[3 * perm[i]..3 * perm[i] + 3]);
        }
        cases.push(("shuffled".into(), x));
    }
    // within the lattice of 5e-4 (2048 nm)
    cases.push(("offset 1600 nm".into(), base.iter().map(|v| v + 1600.0).collect()));
    cases.push(("Angstrom".into(), base.iter().map(|v| v * 10.0).collect()));
    for (name, x) in &cases {
        let r = rt(x, 1, x.len() / 3, 5e-4);
        assert_eq!(r.algo, A::BIOMD, "{name}");
        assert!(r.max_err <= 5e-4, "{name}");
    }
}

/// Compresses `x` as ALGO_BIOMD with `dims`; returns the algorithm the stream holds and the values.
fn compress_decompress<T: SzData>(x: &[T], dims: &[usize], bound: f64) -> sz3::Result<(u8, Vec<T>)> {
    let mut conf = Config::new(dims);
    conf.cmpr_algo = A::BIOMD;
    conf.error_bound_mode = eb::ABS;
    conf.abs_error_bound = bound;
    let (dec, y) = decompress::<T>(&compress(&conf, x)?)?;
    Ok((dec.cmpr_algo, y))
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn expect_within<T: Copy + Into<f64>>(x: &[T], y: &[T], bound: f64) {
    for (i, (&a, &b)) in x.iter().zip(y).enumerate() {
        assert!((b.into() - a.into()).abs() <= bound, "at {i}");
    }
}

// BioMD.FallsBackOnWhatItDoesNotCode: shapes other than {frames, atoms, 3} are refused;
// coordinates beyond the lattice the bound allows, NaN or Inf outside trailing fill, and bounds
// that are not positive and finite go to lossless compression.
#[test]
fn falls_back_on_what_it_does_not_code() {
    let (x, n) = make_system(&SystemSpec::default());
    assert_eq!(compress_decompress(&x, &[n, 3], 5e-4).unwrap().0, A::BIOMD);
    for dims in [&[x.len()][..], &[3, n], &[2, 5, n / 10, 3]] {
        // n = 2650
        assert!(
            matches!(compress_decompress(&x, dims, 5e-4), Err(Error::InvalidArgument(_))),
            "{dims:?}"
        );
    }
    {
        let y: Vec<f32> = x.iter().map(|v| v + 3e3).collect(); // beyond the lattice of 5e-4 (2048 nm)
        let r = compress_decompress(&y, &[n, 3], 5e-4).unwrap();
        assert_eq!(r.0, A::LOSSLESS);
        assert!(same_bits(&r.1, &y));
    }
    for bad in [f32::NAN, f32::INFINITY] {
        let mut y = x.clone();
        y[100] = bad;
        let r = compress_decompress(&y, &[n, 3], 5e-4).unwrap();
        assert_eq!(r.0, A::LOSSLESS);
        assert!(same_bits(&r.1, &y));
    }
    // bounds that are not positive and finite, ABS ones and REL over data with Inf
    for bound in [f64::INFINITY, f64::NAN, -1e-3] {
        let r = compress_decompress(&x, &[n, 3], bound).unwrap();
        assert_eq!(r.0, A::LOSSLESS, "{bound}");
        assert!(same_bits(&r.1, &x), "{bound}");
    }
    {
        let mut y = x.clone();
        y[100] = f32::INFINITY;
        let mut conf = Config::new(&[n, 3]);
        conf.cmpr_algo = A::BIOMD;
        conf.error_bound_mode = eb::REL;
        conf.rel_error_bound = 1e-4;
        let (_, z) = decompress::<f32>(&compress(&conf, &y).unwrap()).unwrap();
        assert!(same_bits(&z, &y));
    }
    // a bound so large that the lattice would pass the largest value of the type
    for (v, bound) in [(3e38f32, 1e38), (1.0, 1e30)] {
        let y = [v, 0.0, 0.0, 0.0, v, 0.0];
        let r = compress_decompress(&y, &[2, 3], bound).unwrap();
        assert!(r.1.iter().all(|v| v.is_finite()));
        expect_within(&y, &r.1, bound);
    }
    let z = [1.5e308f64, 0.0, 0.0, 0.0, 1.5e308, 0.0];
    let r = compress_decompress(&z, &[2, 3], 5e307).unwrap();
    assert!(r.1.iter().all(|v| v.is_finite()));
    expect_within(&z, &r.1, 5e307);
}

// BioMD.NaNAppendedToACodedChunkKeepsItsFramesWithinTheBound: a chunk BIOMD coded, decompressed
// and compressed again with a NaN in a frame appended to it (HDF5 rewriting a chunk).
#[test]
fn nan_appended_to_a_coded_chunk_keeps_its_frames_within_the_bound() {
    let s = SystemSpec {
        frames: 10,
        ..Default::default()
    };
    let (x, n) = make_system(&s);
    let bound = 5e-4;
    let first = compress_decompress(&x, &[s.frames, n, 3], bound).unwrap();
    assert_eq!(first.0, A::BIOMD);
    let mut y = first.1;
    y[9 * n * 3 + 100] = f32::NAN;
    let second = compress_decompress(&y, &[s.frames, n, 3], bound).unwrap();
    assert_eq!(second.0, A::LOSSLESS);
    expect_within(&x[..9 * n * 3], &second.1[..9 * n * 3], bound);
}

// BioMD.FourSiteWaterIsFoundWhereverTheProbesLand: water detection probes every atoms / 300th
// atom; with leading atoms the probes meet 4-site waters at O, H1, H2 and M. The waters must be
// found from each.
#[test]
fn four_site_water_is_found_wherever_the_probes_land() {
    let (base, n) = make_system(&SystemSpec {
        four_site: true,
        waters: 600, // 2650 atoms, and up to 3 more: a probe every 8 atoms
        ..Default::default()
    });
    let mut ratios = Vec::new();
    for k in 0..4 {
        let mut x = Vec::new();
        for a in 0..k {
            x.extend_from_slice(&[0.5 + a as f32, 0.5, 0.5]);
        }
        x.extend_from_slice(&base);
        let r = rt(&x, 1, n + k, 5e-4);
        assert!(r.max_err <= 5e-4);
        ratios.push((x.len() * 4) as f64 / r.bytes.len() as f64);
    }
    let (mn, mx) = ratios.iter().fold((f64::MAX, 0f64), |(a, b), &r| (a.min(r), b.max(r)));
    assert!(mn / mx > 0.95, "{ratios:?}");
}

// BioMD.BeadsWithoutBondsAreStoredAsLatticePoints: a frame of beads with no bond in range
// (coarse-grained water) is stored as raw lattice points in its box: within the bound, in about the
// bits of its box, and the same values when compressed again.
#[test]
fn beads_without_bonds_are_stored_as_lattice_points() {
    let mut g = Mt32::new(5);
    let x: Vec<f32> = (0..3 * 3000).map(|_| g.uniform_f32(0.0, 10.0)).collect();
    for bound in [5e-3, 5e-4] {
        let r = rt(&x, 1, 3000, bound);
        assert_eq!(r.algo, A::BIOMD);
        assert!(r.max_err <= bound);
        let bits = 3.0 * (10.0 / (2.0 * bound)).log2().ceil(); // a 10 nm box at about 2 eb per point
        assert!((r.bytes.len() * 8) as f64 <= 1.05 * bits * 3000.0 + 2048.0, "{bound}");
        assert_eq!(rt(&r.out, 1, 3000, bound).out, r.out, "{bound}");
    }
}

// BioMD.CoordinatesJustBelowTheSpanCompressTheSameAgain: decompressed onto lattice points that may
// round past their originals, they must still be BIOMD's and compress to the same values.
#[test]
fn coordinates_just_below_the_span_compress_the_same_again() {
    let (x, n) = make_system(&SystemSpec::default());
    let (bound, span) = (5e-4, 2048.0); // B for float at 5e-4: the power of two above 2^21 eb
    let hi = x.iter().fold(0f32, |a, &v| a.max(v)) as f64;
    for gap in [0.0004, 0.0006, 0.0011] {
        let y: Vec<f32> = x.iter().map(|&v| (v as f64 + (span - gap - hi)) as f32).collect();
        let mut r = rt(&y, 1, n, bound);
        assert_eq!(r.algo, A::BIOMD, "{gap}");
        for round in 0..3 {
            let again = rt(&r.out, 1, n, bound);
            assert_eq!(again.algo, A::BIOMD, "{gap}");
            assert_eq!(again.out, r.out, "{gap} round {round}");
            r = again;
        }
        expect_within(&y, &r.out, bound);
    }
}

/// `biomd::LATTICE_SPAN_BITS<T>` (not exported by the Rust port).
const LATTICE_SPAN_BITS_F32: i32 = 21;
const LATTICE_SPAN_BITS_F64: i32 = 27;

/// The exponent `frexp` returns.
fn frexp_exp(v: f64) -> i32 {
    let e = ((v.to_bits() >> 52) & 0x7ff) as i32;
    e - 1022
}

// BioMD.ValuesAroundTheSpanStayWithinTheBound: every float from 6 steps below the lattice span B to
// 4 above, alone at the end of a frame, at bounds whose step falls in different places of a float
// ulp.
#[test]
fn values_around_the_span_stay_within_the_bound() {
    let (x, n) = make_system(&SystemSpec::default());
    for bound in [5e-4, 2f64.powi(-11), 3e-5, 7.3e-4] {
        let span = 2f64.powi(frexp_exp(bound * 2f64.powi(LATTICE_SPAN_BITS_F32))) as f32;
        for sign in [1f32, -1.0] {
            let mut v = span - (6.0 * bound) as f32;
            while v < span + (4.0 * bound) as f32 {
                let mut y = x.clone();
                *y.last_mut().unwrap() = sign * v;
                assert!(rt(&y, 1, n, bound).max_err <= bound, "{bound} {}", sign * v);
                v = f32::from_bits(v.to_bits() + 1);
            }
        }
    }
}

// BioMD.ChunkRewrittenAcrossTheSpanStaysWithinTheBound: a chunk's last frame goes past the lattice
// span and back, so the chunk goes from BIOMD to lossless storage and back, again and again. The
// frames before stay within the bound of the original data.
#[test]
fn chunk_rewritten_across_the_span_stays_within_the_bound() {
    let s = SystemSpec {
        frames: 4,
        ..Default::default()
    };
    let (x, n) = make_system(&s);
    let bound = 5e-4;
    let dims = [s.frames, n, 3];
    let mut y = compress_decompress(&x, &dims, bound).unwrap().1;
    for _ in 0..3 {
        let mut out = y.clone();
        for i in 3 * n * 3..4 * n * 3 {
            out[i] = x[i] + 2400.0; // past B = 2048 nm
        }
        let past = compress_decompress(&out, &dims, bound).unwrap();
        assert_eq!(past.0, A::LOSSLESS);
        y = past.1;
        y[3 * n * 3..].copy_from_slice(&x[3 * n * 3..]); // the frame written again, within B
        let back = compress_decompress(&y, &dims, bound).unwrap();
        assert_eq!(back.0, A::BIOMD);
        y = back.1;
        expect_within(&x[..3 * n * 3], &y[..3 * n * 3], bound);
    }
}

// BioMD.OneAtom: Config drops dimensions of 1: one atom arrives as {frames, 3}, one atom of one
// frame as {3}.
#[test]
fn one_atom() {
    for frames in [1, 5] {
        let x: Vec<f32> = (0..frames * 3).map(|i| 1.0 + 0.01 * i as f32).collect();
        let mut conf = Config::new(&[frames, 1, 3]);
        conf.cmpr_algo = A::BIOMD;
        conf.error_bound_mode = eb::ABS;
        conf.abs_error_bound = 5e-4;
        let (dconf, y) = decompress::<f32>(&compress(&conf, &x).unwrap()).unwrap();
        assert_eq!(dconf.cmpr_algo, A::BIOMD);
        assert!(max_abs_error(&x, &y) <= 5e-4, "{frames} frames");
    }
}

// BioMD.TrailingFillFramesComeBackExactly
#[test]
fn trailing_fill_frames_come_back_exactly() {
    let s = SystemSpec {
        frames: 8,
        ..Default::default()
    };
    let (mut x, n) = make_system(&s);
    x[5 * n * 3..].fill(0.0); // a chunk with 5 frames written
    let r = rt(&x, s.frames, n, 5e-4);
    assert!(r.max_err <= 5e-4);
    assert!(r.out[5 * n * 3..].iter().all(|&v| v == 0.0));
}

// BioMD.RecompressionChangesNothing: GROMACS rewrites a chunk after restarting from a checkpoint:
// decompress, append, compress again.
#[test]
fn recompression_changes_nothing() {
    for frames in [1, 6] {
        for bound in [5e-4, 1e-3] {
            let (x, n) = make_system(&SystemSpec {
                frames,
                ..Default::default()
            });
            let a = rt(&x, frames, n, bound);
            let b = rt(&a.out, frames, n, bound);
            assert!(same_bits(&a.out, &b.out), "{frames} {bound}");
            assert!(b.max_err <= bound);
        }
    }
}

// BioMD.RecompressionAcrossAPowerOfTwoStaysWithinTheBound: the lattice depends on the bound alone:
// a decoded value that crosses a power of two, or a frame appended to a chunk that raises its
// largest |x|, lands on the same lattice points when compressed again.
#[test]
fn recompression_across_a_power_of_two_stays_within_the_bound() {
    let bound = 5e-4;
    let s = SystemSpec {
        frames: 4,
        ..Default::default()
    };
    let (mut x, n) = make_system(&s);
    let mut k = 0;
    for i in 0..x.len() {
        if x[i].abs() > x[k].abs() {
            k = i;
        }
    }
    let p2 = (x[k].abs() as f64).log2().ceil().exp2();
    x[k] = (p2 - 1e-4).copysign(x[k] as f64) as f32;
    let mut cur = x.clone();
    for _ in 0..3 {
        cur = rt(&cur, s.frames, n, bound).out;
        expect_within(&x, &cur, bound);
    }
    // two frames and two of fill, then a third frame whose largest |x| is past the power of two
    let mut chunk = x[..2 * n * 3].to_vec();
    chunk.resize(4 * n * 3, 0.0);
    let mut dec = rt(&chunk, 4, n, bound).out;
    dec[2 * n * 3..3 * n * 3].copy_from_slice(&x[2 * n * 3..3 * n * 3]);
    dec[2 * n * 3] = (p2 + 0.5) as f32;
    let again = rt(&dec, 4, n, bound).out;
    expect_within(&x[..2 * n * 3], &again[..2 * n * 3], bound);
}

// BioMD.CorruptStreamsAreRefusedOrDecodeInBounds: truncated and altered payloads return an error
// or decode; they must not panic.
#[test]
fn corrupt_streams_are_refused_or_decode() {
    let s = SystemSpec {
        frames: 3,
        four_site: true,
        ..Default::default()
    };
    let (x, n) = make_system(&s);
    let r = rt(&x, s.frames, n, 5e-4);
    // SZ3's stream: 16-byte header (magic, version, payload size), the payload, then the config
    let payload = u64::from_le_bytes(r.bytes[8..16].try_into().unwrap()) as usize;
    assert!(16 + payload < r.bytes.len());
    let mut g = Mt32::new(9);
    let mut cut = 16;
    while cut < 16 + payload {
        // payload truncated
        let mut b = r.bytes[..cut].to_vec();
        b.extend_from_slice(&r.bytes[16 + payload..]);
        b[8..16].copy_from_slice(&((cut - 16) as u64).to_le_bytes());
        let _ = decompress::<f32>(&b);
        cut += (payload / 300).max(1);
    }
    for _ in 0..300 {
        // payload bytes changed
        let mut b = r.bytes.clone();
        for _ in 0..4 {
            b[16 + g.next_u32() as usize % payload] ^= (1 + g.next_u32() % 255) as u8;
        }
        let _ = decompress::<f32>(&b);
    }
}

// BioMD.DoubleNearHalfLatticePointsStaysWithinBound: double values near the midpoint between two
// lattice points, where the double rounding of x / step and of q step back takes the most of the
// step's margin.
#[test]
fn double_near_half_lattice_points_stays_within_bound() {
    let x = [510.39700369221237, 0.0, 0.0, 467.45074901980968, 0.0, 0.0];
    let bound = 5.8345702187110699e-06;
    let r = rt(&x, 1, 2, bound);
    assert_eq!(r.algo, A::BIOMD);
    assert!(r.max_err <= bound);
}

/// A water whose O and H2 swap ends of the lattice between frames: the H2 symbol against the
/// previous frame must still fit 32 bits.
fn water_jumps_across_the_range<T: SzData + Into<f64>>(span_bits: i32, cast: fn(f64) -> T) {
    // the lattice spans |x| < 2^(LATTICE_SPAN_BITS + 1) eb when 2^LATTICE_SPAN_BITS eb is a power of two
    let bound = 2f64.powi(-10);
    let m = 0.99 * bound * 2f64.powi(span_bits + 1);
    let (waters, frames) = (40, 3);
    let atoms = 3 * waters + 2;
    let mut x = vec![cast(0.0); frames * atoms * 3];
    let mut g = Mt32::new(1);
    let (r, th) = (0.09572, 104.52 * (-1f64).acos() / 180.0);
    let o0: Vec<f64> = (0..3 * waters).map(|_| g.uniform(0.0, 3.0)).collect();
    for f in 0..frames {
        let xf = &mut x[f * atoms * 3..(f + 1) * atoms * 3];
        for w in 0..waters {
            let mut o = [o0[3 * w] + 0.001 * f as f64, o0[3 * w + 1], o0[3 * w + 2]];
            let mut h1 = [o[0], o[1] + r, o[2]];
            let mut h2 = [o[0], o[1] + r * th.cos(), o[2] + r * th.sin()];
            if w == 7 && f > 0 {
                o[0] = if f == 1 { m } else { -m };
                h1[0] = o[0];
                h2[0] = if f == 1 { -m } else { m };
            }
            for c in 0..3 {
                xf[9 * w + c] = cast(o[c]);
                xf[9 * w + 3 + c] = cast(h1[c]);
                xf[9 * w + 6 + c] = cast(h2[c]);
            }
        }
        xf[9 * waters] = cast(m); // two ions fix max|x|
        xf[9 * waters + 3] = cast(-m);
    }
    let r = rt(&x, frames, atoms, bound);
    assert_eq!(r.algo, A::BIOMD);
    assert!(r.max_err <= bound);
}

// BioMD.WaterJumpsAcrossTheCoordinateRange
#[test]
fn water_jumps_across_the_coordinate_range() {
    water_jumps_across_the_range::<f32>(LATTICE_SPAN_BITS_F32, |v| v as f32);
    water_jumps_across_the_range::<f64>(LATTICE_SPAN_BITS_F64, |v| v);
}

// BioMD.WaterGeometryAtTheLatticeLimitDecodes: the O-H distance at 16384 - 1e-5 lattice steps: its
// square rounds to 2^28, which load() refuses, so the codec must not code the water on it.
#[test]
fn water_geometry_at_the_lattice_limit_decodes() {
    let (x, n) = make_system(&SystemSpec::default());
    let bound = 3.1597940068971598e-06;
    assert!(rt(&x, 1, n, bound).max_err <= bound);
}

// BioMD.ChunkOfFillOnly: an unwritten chunk is all fill, NaN included.
#[test]
fn chunk_of_fill_only() {
    let x = vec![f32::NAN; 3 * 10 * 3];
    let r = rt(&x, 3, 10, 5e-4);
    assert_eq!(r.algo, A::BIOMD);
    assert!(r.out.iter().all(|v| v.is_nan()));
}

// BioMD.FewAtomsManyFrames: what each frame stores beyond its symbols must not outgrow the data.
#[test]
fn few_atoms_many_frames() {
    let (frames, atoms) = (5000, 2);
    let mut g = Mt32::new(3);
    let mut normal = Normal::default();
    let mut x = vec![0f32; frames * atoms * 3];
    for i in 0..x.len() {
        let prev = if i < atoms * 3 { 1.0 } else { x[i - atoms * 3] };
        x[i] = prev + (normal.sample(&mut g) * 0.01) as f32;
    }
    let r = rt(&x, frames, atoms, 5e-4);
    assert_eq!(r.algo, A::BIOMD);
    assert!(r.max_err <= 5e-4);
}
