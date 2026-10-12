//! SZ3's test_huffman_encoder.cpp and test_segmented_encoder.cpp: exact round trips, save() ==
//! size_est(), size_bound() covers save() + encode(), optimal payloads, stable bytes, and decoders
//! that return errors on corrupted streams. The Rust encoder codes i32 bins only.

mod common;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use common::{Mt32, Mt64};
use sz3::Error;
use sz3::bytes::Reader;
use sz3::encoder::{Encoder, HuffmanEncoder, SegmentedEncoder};

struct Encoded {
    tree: Vec<u8>,
    data: Vec<u8>,
    est: usize,
}

fn encode(bins: &[i32]) -> Encoded {
    let mut enc = HuffmanEncoder::new();
    enc.preprocess_encode(bins, 0);
    let (mut tree, mut data) = (Vec::new(), Vec::new());
    enc.save(&mut tree);
    enc.encode(bins, &mut data).unwrap();
    Encoded {
        tree,
        data,
        est: enc.size_est(),
    }
}

fn decode(e: &Encoded, n: usize) -> Vec<i32> {
    let mut dec = HuffmanEncoder::new();
    let mut t = Reader::new(&e.tree);
    dec.load(&mut t).unwrap();
    assert_eq!(t.remaining(), 0);
    let mut d = Reader::new(&e.data);
    let out = dec.decode(&mut d, n).unwrap();
    assert_eq!(d.remaining(), 0);
    out
}

/// The bit count encode() writes in front of the codes.
fn payload_bits(v: &[u8]) -> u64 {
    u64::from_le_bytes(v[..8].try_into().unwrap())
}

/// Cost in bits of an optimal prefix code for these bins, and the depth of one optimal tree.
fn optimal_bits(bins: &[i32]) -> (u64, u32) {
    let mut freq: HashMap<i32, u64> = HashMap::new();
    for &b in bins {
        *freq.entry(b).or_default() += 1;
    }
    if freq.len() < 2 {
        return (0, 0);
    }
    let mut q: BinaryHeap<Reverse<(u64, u32)>> = freq.values().map(|&f| Reverse((f, 0))).collect();
    let (mut cost, mut depth) = (0, 0);
    while q.len() > 1 {
        let Reverse(a) = q.pop().unwrap();
        let Reverse(b) = q.pop().unwrap();
        cost += a.0 + b.0;
        depth = a.1.max(b.1) + 1;
        q.push(Reverse((a.0 + b.0, depth)));
    }
    (cost, depth)
}

fn distinct(bins: &[i32]) -> usize {
    let mut v = bins.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Everything a valid stream must satisfy.
fn check_opt(bins: &[i32], expect_optimal: bool) -> Encoded {
    let e = encode(bins);
    assert_eq!(e.tree.len(), e.est, "save() must write exactly size_est()");
    assert!(e.tree.len() + e.data.len() <= HuffmanEncoder::size_bound(bins.len(), distinct(bins)));
    let bits = payload_bits(&e.data);
    assert_eq!(e.data.len() as u64, 8 + bits.div_ceil(8));
    if expect_optimal {
        assert_eq!(bits, optimal_bits(bins).0);
    }
    assert!(decode(&e, bins.len()) == bins, "round trip differs, n = {}", bins.len());
    e
}

fn check(bins: &[i32]) -> Encoded {
    check_opt(bins, true)
}

fn laplace(n: usize, scale: f64, center: i32, seed: u64) -> Vec<i32> {
    let mut g = Mt64::new(seed);
    (0..n)
        .map(|_| {
            let x = g.exponential(1.0 / scale);
            center + (if g.next_u64() & 1 != 0 { x } else { -x }).round() as i32
        })
        .collect()
}

// SZ3_HuffmanEncoder.Empty
#[test]
fn empty() {
    let e = check(&[]);
    assert_eq!(e.tree.len(), 4);
    assert_eq!(e.data.len(), 8);
    // An empty table cannot produce values.
    let mut dec = HuffmanEncoder::new();
    dec.load(&mut Reader::new(&e.tree)).unwrap();
    assert!(dec.decode(&mut Reader::new(&e.data), 1).is_err());
}

// SZ3_HuffmanEncoder.SingleSymbol
#[test]
fn single_symbol() {
    for v in [0, 1, -1, 32768, i32::MAX, i32::MIN] {
        for n in [1, 2, 1000] {
            assert_eq!(check(&vec![v; n]).data.len(), 8);
        }
    }
}

// SZ3_HuffmanEncoder.SmallAlphabetsEveryBitBoundary
#[test]
fn small_alphabets_every_bit_boundary() {
    let mut g = Mt64::new(1);
    for k in 2..=9 {
        for n in 1..=130 {
            let bins: Vec<i32> = (0..n).map(|_| 100 + (g.next_u64() % k) as i32).collect();
            check(&bins);
        }
    }
}

// SZ3_HuffmanEncoder.QuantizationLike
#[test]
fn quantization_like() {
    for scale in [0.3, 1.0, 5.0, 60.0, 3000.0] {
        for n in [10, 1000, 7956, 100000] {
            let mut bins = laplace(n, scale, 32768, (scale * 1000.0) as u64 + n as u64);
            for i in (0..n).step_by(97) {
                bins[i] = 0; // unpredictable
            }
            check(&bins);
        }
    }
}

// SZ3_HuffmanEncoder.LongRunsOfTheCommonestSymbol: a dominant symbol gets the one-bit all-zero
// code, and the decoder takes its runs in one step.
#[test]
fn long_runs_of_the_commonest_symbol() {
    let mut g = Mt64::new(16);
    for rarity in [3, 20, 200, 5000] {
        for n in [2, 9, 63, 64, 65, 200, 4099, 60000] {
            let mut bins = vec![32768; n];
            for b in bins.iter_mut() {
                if g.next_u64().is_multiple_of(rarity) {
                    *b = 32768 + (g.next_u64() % 9) as i32 - 4;
                }
            }
            check(&bins);
            let mut bins = vec![7; n];
            bins[n - 1] = 8; // a run up to the last code
            check(&bins);
            let mut bins = vec![7; n];
            bins[0] = -8;
            check(&bins);
        }
    }
}

// SZ3_HuffmanEncoder.CodeLengthsAroundTheTable: table-length codes and longer ones, around every
// length the decoder switches on.
#[test]
fn code_lengths_around_the_table() {
    for top in 8..=22 {
        let mut bins = Vec::new();
        // Geometric weights 2^-k give code lengths 1..top.
        for k in 0..=top {
            bins.extend(std::iter::repeat_n(k * 3 - 20, 1 << (top - k)));
        }
        Mt64::new(top as u64).shuffle(&mut bins);
        check(&bins);
    }
}

// SZ3_HuffmanEncoder.NegativeAndExtremeBins (the int64_t and uint64_t bins are skipped: the Rust
// encoder codes i32 bins only)
#[test]
fn negative_and_extreme_bins() {
    check(&[i32::MIN, i32::MAX, i32::MIN, 0, -1, 1, i32::MAX]);
    check(&laplace(5000, 20.0, -100000, 3));
}

// SZ3_HuffmanEncoder.OtherTypes: the uint8_t, int16_t and uint16_t bins, as i32 bins.
#[test]
fn other_types() {
    let mut g = Mt64::new(4);
    let a: Vec<i32> = (0..3000)
        .map(|_| {
            let x = (g.next_u64() % 7) as u8;
            x.wrapping_add(250 * g.next_u64().is_multiple_of(50) as u8) as i32
        })
        .collect();
    check(&a);
    let b: Vec<i32> = (0..3000)
        .map(|_| {
            let x = (g.next_u64() % 11) as i32 - 5;
            (x + if g.next_u64().is_multiple_of(100) { -32768 } else { 0 }) as i16 as i32
        })
        .collect();
    check(&b);
    let c: Vec<i32> = (0..3000)
        .map(|_| {
            if !g.next_u64().is_multiple_of(3) {
                65535
            } else {
                (g.next_u64() % 40) as i32
            }
        })
        .collect();
    check(&c);
}

// SZ3_HuffmanEncoder.SparseWideRange: a range far wider than the input takes the hash path.
#[test]
fn sparse_wide_range() {
    let mut g = Mt64::new(5);
    let vals: Vec<i32> = (0..3000).map(|_| g.next_u64() as i32).collect();
    for n in [2, 50, 4000, 200000] {
        let bins: Vec<i32> = (0..n).map(|_| vals[g.geometric(0.003).min(2999)]).collect();
        check(&bins);
    }
    // All distinct.
    let all: Vec<i32> = (0..20000u32)
        .map(|i| (i.wrapping_mul(104729) as i32).wrapping_sub(1000000000))
        .collect();
    check(&all);
}

// SZ3_HuffmanEncoder.LengthLimited: Fibonacci weights give an optimal code 34 bits deep, so the
// lengths are limited to 32.
#[test]
fn length_limited() {
    let mut bins = Vec::new();
    let (mut a, mut b) = (1u64, 1u64);
    for k in 0..35 {
        bins.extend(std::iter::repeat_n(k, a as usize));
        (a, b) = (b, a + b);
    }
    Mt64::new(6).shuffle(&mut bins);
    let (opt, depth) = optimal_bits(&bins);
    assert!(depth > 32);
    let e = check_opt(&bins, false);
    let bits = payload_bits(&e.data);
    assert!(bits >= opt);
    assert!(bits <= opt + opt / 1000000);
}

// SZ3_HuffmanEncoder.StableBytes: the same bins give the same bytes on every host, as C++ SZ3
// writes them.
#[test]
fn stable_bytes() {
    let bins = [5, 5, 5, 7, 7, 9, 5, 12, 5, 7];
    let e = check(&bins);
    // Lengths 5:1 7:2 9:3 12:3, so codes 0, 10, 110, 111 and 17 payload bits. After the count 4
    // and the offset 5 (little-endian), the table holds gaps 2 2 3 and length changes +1 +1 +1 0.
    let tree = [4, 0, 0, 0, 5, 0, 0, 0, 0x69, 0xa6, 0xe0];
    let data = [17, 0, 0, 0, 0, 0, 0, 0, 0x15, 0x9d, 0x00];
    assert_eq!(e.tree, tree);
    assert_eq!(e.data, data);
    // Independent of what the encoder was used for before.
    let mut enc = HuffmanEncoder::new();
    enc.preprocess_encode(&laplace(1000, 3.0, 0, 7), 0);
    enc.preprocess_encode(&bins, 0);
    let mut t = Vec::new();
    enc.save(&mut t);
    assert_eq!(t.len(), enc.size_est());
    assert_eq!(t, tree);
}

// SZ3_HuffmanEncoder.CopiesAreIndependent
#[test]
fn copies_are_independent() {
    let bins = laplace(3000, 4.0, 10, 8);
    let mut enc = HuffmanEncoder::new();
    enc.preprocess_encode(&bins, 0);
    let copy = enc.clone();
    enc.preprocess_encode(&[1, 2, 3], 0);
    let mut buf = Vec::new();
    copy.save(&mut buf);
    copy.encode(&bins, &mut buf).unwrap();
    let mut dec = HuffmanEncoder::new();
    let mut r = Reader::new(&buf);
    dec.load(&mut r).unwrap();
    assert_eq!(dec.decode(&mut r, bins.len()).unwrap(), bins);
}

// SZ3_HuffmanEncoder.RefusesBinsItWasNotBuiltFor
#[test]
fn refuses_bins_it_was_not_built_for() {
    let refused = |built: &[i32], bins: &[i32]| {
        let mut enc = HuffmanEncoder::new();
        enc.preprocess_encode(built, 0);
        matches!(enc.encode(bins, &mut Vec::new()), Err(Error::InvalidArgument(_)))
    };
    assert!(refused(&[1, 3, 3], &[1, 3, 9])); // outside the range
    assert!(refused(&[1, 3, 3], &[1, 2, 3])); // inside, never counted
    assert!(refused(&[4, 4], &[4, 5])); // one symbol
    assert!(refused(&[], &[0])); // no symbol
    assert!(refused(&[0, 1000000, 1000000], &[0, 1000000, 7])); // hashed
}

// SZ3_HuffmanEncoder.WrongValueCountThrows
#[test]
fn wrong_value_count_is_an_error() {
    let bins = laplace(5000, 6.0, 0, 9);
    let e = encode(&bins);
    for n in [0, 1, bins.len() - 1, bins.len() + 1, bins.len() * 40] {
        let mut dec = HuffmanEncoder::new();
        dec.load(&mut Reader::new(&e.tree)).unwrap();
        assert!(dec.decode(&mut Reader::new(&e.data), n).is_err(), "{n}");
    }
}

/// Decodes a possibly corrupted stream; true if it decoded.
fn try_decode(tree: &[u8], data: &[u8], n: usize) -> bool {
    let mut dec = HuffmanEncoder::new();
    if dec.load(&mut Reader::new(tree)).is_err() {
        return false;
    }
    match dec.decode(&mut Reader::new(data), n) {
        Ok(out) => {
            assert_eq!(out.len(), n, "decode returned {} values", out.len());
            true
        }
        Err(_) => false,
    }
}

// SZ3_HuffmanEncoder.TruncationsThrow
#[test]
fn truncations_are_errors() {
    for bins in [laplace(3000, 2.0, 32768, 10), laplace(3000, 300.0, 32768, 11)] {
        let e = encode(&bins);
        for k in 0..e.tree.len() {
            assert!(!try_decode(&e.tree[..k], &e.data, bins.len()), "tree cut at {k}");
        }
        for k in 0..e.data.len() {
            assert!(!try_decode(&e.tree, &e.data[..k], bins.len()), "data cut at {k}");
        }
    }
}

// SZ3_HuffmanEncoder.MutationsNeverOverrun: some mutations still decode (a flipped payload bit can
// map to another valid code); none may panic.
#[test]
fn mutations_never_panic() {
    let mut g = Mt64::new(12);
    let mut inputs = vec![
        laplace(2000, 1.0, 32768, 13),
        laplace(2000, 50.0, 32768, 14),
        laplace(300, 3000.0, 0, 15),
        vec![7; 100],
    ];
    let mut fib = Vec::new(); // long codes
    let (mut a, mut b) = (1u64, 1u64);
    for k in 0..20 {
        fib.extend(std::iter::repeat_n(k * 1000, a as usize));
        (a, b) = (b, a + b);
    }
    inputs.push(fib);
    for bins in &inputs {
        let e = encode(bins);
        for _ in 0..3000 {
            let (mut t, mut d) = (e.tree.clone(), e.data.clone());
            let flips = 1 + g.next_u64() % 4;
            for _ in 0..flips {
                let in_tree = g.next_u64() & 1 != 0 && !t.is_empty();
                let v = if in_tree { &mut t } else { &mut d };
                if v.is_empty() {
                    continue;
                }
                match g.next_u64() % 3 {
                    0 => {
                        let i = (g.next_u64() % v.len() as u64) as usize;
                        v[i] ^= 1 << (g.next_u64() % 8);
                    }
                    1 => {
                        let i = (g.next_u64() % v.len() as u64) as usize;
                        v[i] = g.next_u64() as u8;
                    }
                    _ => {
                        if v.len() > 1 {
                            let n = (g.next_u64() % v.len() as u64) as usize;
                            v.truncate(n);
                        }
                    }
                }
            }
            try_decode(&t, &d, bins.len());
        }
    }
}

fn segmented_round_trip(bins: &[i32]) -> Vec<i32> {
    let mut enc = SegmentedEncoder::<HuffmanEncoder>::new(64);
    enc.preprocess_encode(bins, 0);
    let mut buf = Vec::new();
    enc.save(&mut buf);
    assert!(buf.len() <= enc.size_est());
    enc.encode(bins, &mut buf).unwrap();
    let mut dec = SegmentedEncoder::<HuffmanEncoder>::new(u32::MAX);
    let mut r = Reader::new(&buf);
    dec.load(&mut r).unwrap();
    dec.decode(&mut r, bins.len()).unwrap()
}

// test_segmented_encoder.cpp: SZ3_SegmentedEncoder.RoundTrip
#[test]
fn segmented_round_trips() {
    let mut g = Mt32::new(5);
    let mut bins = Vec::new();
    for s in 0..30 {
        let n = if s % 3 == 0 { 0 } else { g.next_u32() % 6000 }; // empty segments too
        bins.push(n as i32);
        for _ in 0..n {
            bins.push((g.next_u32() % (s + 2)) as i32 - (s / 2) as i32);
        }
    }
    assert_eq!(segmented_round_trip(&bins), bins);
    assert_eq!(segmented_round_trip(&[]), Vec::<i32>::new());
    assert_eq!(segmented_round_trip(&[0, 0, 0]), vec![0, 0, 0]);
}

// test_segmented_encoder.cpp: SZ3_SegmentedEncoder.CorruptSegmentTables: sizes beyond an int,
// sizes that do not add up, and many empty segments, which must cost their bytes rather than an
// encoder each.
#[test]
fn segmented_corrupt_segment_tables() {
    let stream = |count: u32, rest: &[u8]| {
        let mut b = count.to_le_bytes().to_vec();
        b.extend_from_slice(rest);
        b
    };
    let load = |b: &[u8], dec: &mut SegmentedEncoder<HuffmanEncoder>| dec.load(&mut Reader::new(b));

    let mut dec = SegmentedEncoder::<HuffmanEncoder>::new(u32::MAX);
    let max = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]; // 2^64 - 1
    assert!(load(&stream(1, &max), &mut dec).is_err());
    assert!(load(&stream(1, &[0x80, 0x80, 0x80, 0x80, 0x08]), &mut dec).is_err()); // 2^31

    let mut dec = SegmentedEncoder::<HuffmanEncoder>::new(u32::MAX);
    load(&stream(3, &[0, 0, 0]), &mut dec).unwrap();
    assert!(dec.decode(&mut Reader::new(&[]), 2).is_err()); // three counts are three bins
    assert_eq!(dec.decode(&mut Reader::new(&[]), 3).unwrap(), vec![0, 0, 0]);

    let count = 1000000;
    let mut dec = SegmentedEncoder::<HuffmanEncoder>::new(u32::MAX);
    load(&stream(count, &vec![0; count as usize]), &mut dec).unwrap();
    assert_eq!(
        dec.decode(&mut Reader::new(&[]), count as usize).unwrap().len(),
        count as usize
    );

    // more segments than the caller allows: refused before an encoder is built for each
    let mut dec = SegmentedEncoder::<HuffmanEncoder>::new(12);
    assert!(load(&stream(13, &[0; 13 * 5]), &mut dec).is_err());
    assert!(load(&stream(12, &[0; 12]), &mut dec).is_ok());
}
