//! Compatibility with C++ SZ3's own test vectors.

use sha2::{Digest, Sha256};
use sz3::config::{SZ3_DATA_VER, version_int};
use sz3::{Config, Error, compress, decompress};

fn data(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn stream(name: &str) -> Vec<u8> {
    data(&format!("streams/3.3.2/{name}.sz"))
}

/// FNV-1a over the decoded bytes, as SZ3's test_previous_data_version.cpp computes it.
fn fnv1a(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[test]
fn reads_3_3_2_streams_as_that_release_did() {
    for (name, want) in [
        ("interp", 0xe49ee9e883e32415u64),
        ("interp_omp", 0x1172c536c4c77855),
        ("lorenzo_reg", 0x3f7750a23094223d),
        ("lorenzo_reg_omp", 0xe32514629ac5fd3d),
        ("lorenzo_reg_rel1e-5", 0xe0885e9322ea80cc),
        ("lossless", 0x51c93bc7f98c12a5),
    ] {
        let (conf, out) = decompress::<f32>(&stream(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(conf.sz3_data_ver, version_int("3.3.2"), "{name}");
        assert_eq!(out.len(), 8 * 8 * 128);
        assert_eq!(fnv1a(&out), want, "{name}");
    }
}

#[test]
fn refuses_3_3_2_nopred_and_biomd() {
    for name in ["nopred", "biomd", "biomdxtc"] {
        assert!(
            matches!(decompress::<f32>(&stream(name)), Err(Error::InvalidArgument(_))),
            "{name}"
        );
    }
}

#[test]
fn refuses_versions_outside_the_range() {
    let mut cmp = stream("lorenzo_reg");
    for (v, msg) in [
        (version_int("3.3.0"), "Use SZ3 v3.3.0"),
        (version_int(SZ3_DATA_VER) + (1 << 8), "Upgrade SZ3"),
    ] {
        cmp[4..8].copy_from_slice(&v.to_le_bytes());
        let e = decompress::<f32>(&cmp).unwrap_err().to_string();
        assert!(e.contains(msg), "{e}");
    }
}

#[test]
fn recompressing_writes_this_version() {
    let (mut conf, out) = decompress::<f32>(&stream("lorenzo_reg")).unwrap();
    conf.openmp = false;
    let again = compress(&conf, &out).unwrap();
    let (back, _) = decompress::<f32>(&again).unwrap();
    assert_eq!(back.sz3_data_ver, version_int(SZ3_DATA_VER));
}

/// SZ3's tools/test/data_format_digest.txt: the 3.4.0 format's bytes for its test file.
#[test]
fn matches_sz3_format_digest() {
    let raw = data("testfloat_8_8_128.dat");
    let v: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // `sz3 -3 8 8 128` lists the fastest dimension first.
    let mut conf = Config::new(&[128, 8, 8]);
    conf.abs_error_bound = 1e-3;
    let out = compress(&conf, &v).unwrap();
    let digest: String = Sha256::digest(&out).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        digest,
        "2353117ba820a0b2c77e059b67df98ba267c713f7829b5c4739f3fe3853a67ed"
    );
}
