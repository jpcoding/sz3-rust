//! SZ3's module tests (tools/test/modules) for the byte utilities, Config, LinearQuantizer, the
//! lossless stages and the encoders.

mod common;

use common::{Mt32, Mt64};
use sz3::bytes::{BitAppender, BitConsumer, Reader, read_varint, unzigzag, write_varint, zigzag};
use sz3::encoder::{BypassEncoder, Encoder, HuffmanEncoder};
use sz3::lossless::{Bypass, Lossless, Zstd};
use sz3::quantizer::{LinearQuantizer, Quantizer};
use sz3::{Config, Error};

// test_byte_util.cpp: ByteUtilTest.Zigzag
#[test]
fn byte_util_zigzag() {
    let values = [0, -1, 1, -2, 2, i64::MAX, i64::MIN];
    let expected = [0, 1, 2, 3, 4, u64::MAX - 1, u64::MAX];
    for (&v, &e) in values.iter().zip(&expected) {
        assert_eq!(zigzag(v), e);
        assert_eq!(unzigzag(e), v);
    }
}

// test_byte_util.cpp: ByteUtilTest.VarintRoundTrip
#[test]
fn byte_util_varint_round_trip() {
    let values = [0, 1, 127, 128, 16383, 16384, 1u64 << 35, u64::MAX];
    let mut buf = Vec::new();
    for &v in &values {
        write_varint(v, &mut buf);
    }
    assert_eq!(buf.len(), 1 + 1 + 1 + 2 + 2 + 3 + 6 + 10);
    let mut r = Reader::new(&buf);
    for &v in &values {
        assert_eq!(read_varint(&mut r).unwrap(), v);
    }
    assert_eq!(r.remaining(), 0);
}

// test_byte_util.cpp: ByteUtilTest.VarintRejectsTruncatedAndOverlong
#[test]
fn byte_util_varint_rejects_truncated_and_overlong() {
    assert!(read_varint(&mut Reader::new(&[0x80, 0x80])).is_err());
    let mut overlong = vec![0x80u8; 11];
    *overlong.last_mut().unwrap() = 0;
    assert!(read_varint(&mut Reader::new(&overlong)).is_err());
    // a 10th byte above 1 holds bits past 64
    let mut past64 = vec![0x80u8; 10];
    *past64.last_mut().unwrap() = 2;
    assert!(read_varint(&mut Reader::new(&past64)).is_err());
}

// test_byte_util.cpp: ByteUtilTest.BitsRoundTrip
#[test]
fn byte_util_bits_round_trip() {
    let mut g = Mt64::new(7);
    let mut fields = Vec::new();
    let mut bytes = Vec::new();
    let mut appender = BitAppender::new(&mut bytes);
    let mut total_bits = 0;
    for _ in 0..1000 {
        let bits = (g.next_u64() % 57) as u32;
        let v = if bits == 0 { 0 } else { g.next_u64() >> (64 - bits) };
        fields.push((v, bits));
        appender.put(v, bits);
        total_bits += bits as usize;
    }
    appender.flush();
    assert_eq!(bytes.len(), total_bits.div_ceil(8));

    let mut consumer = BitConsumer::new(&bytes);
    for &(v, bits) in &fields {
        assert_eq!(consumer.get(bits).unwrap(), v);
    }
    assert!(consumer.at_end());
    assert!(consumer.get(8).is_err());
}

/// A config as a later version would write it: three more bytes at the end, counted in its size byte.
fn with_appended_fields(conf: &Config) -> Vec<u8> {
    let mut buf = Vec::new();
    let size = conf.save(&mut buf).unwrap();
    buf.extend_from_slice(&[7, 7, 7]);
    buf[0] = (size + 3) as u8;
    buf
}

// test_config.cpp: SZ3_Config.LoadSkipsFieldsAddedAtTheEnd
#[test]
fn config_load_skips_fields_added_at_the_end() {
    let mut conf = Config::new(&[10, 20, 30]);
    conf.abs_error_bound = 1e-3;
    conf.openmp = true;
    let mut buf = with_appended_fields(&conf);
    buf.push(0xAB); // what follows the config, such as the next chunk's config in an OpenMP stream

    let mut loaded = Config::default();
    let mut r = Reader::new(&buf);
    loaded.load(&mut r).unwrap();
    assert_eq!(r.rest(), &[0xAB]);
    assert_eq!(r.remaining(), 1);
    assert_eq!(loaded.num, conf.num);
    assert_eq!(loaded.abs_error_bound, conf.abs_error_bound);
    assert!(loaded.openmp);
}

// test_config.cpp: SZ3_Config.LoadRefusesASizeShorterThanItsFields
#[test]
fn config_load_refuses_a_size_shorter_than_its_fields() {
    let conf = Config::new(&[10, 20, 30]);
    let mut buf = Vec::new();
    conf.save(&mut buf).unwrap();
    buf[0] = 4;
    assert!(Config::default().load(&mut Reader::new(&buf)).is_err());
}

// test_quantizer.cpp: QuantizerTest.LinearQuantizer (runQuantizeRecoverTest)
#[test]
fn linear_quantizer_quantize_recover() {
    let eb = 12.1973f32 as f64;
    let mut q = LinearQuantizer::<f32>::new(eb, 32768);
    let mut data = 100.11212f32;
    let ori = data;
    let bin = q.quantize_and_overwrite(&mut data, 0.0);
    let recovered = q.recover(0.0, bin).unwrap();
    assert!((recovered as f64 - ori as f64).abs() <= eb);
}

// test_quantizer.cpp: QuantizerTest.LinearQuantizer (runFunctionalTest)
#[test]
fn linear_quantizer_save_load() {
    let eb = 12.1973f32 as f64;
    let mut q = LinearQuantizer::<f32>::new(eb, 32768);
    let mut originals = Vec::new();
    let mut bins = Vec::new();
    for i in 0..100 {
        let mut data = (10.723 + i as f64 * 0.0005) as f32;
        originals.push(data);
        bins.push(q.quantize_and_overwrite(&mut data, 0.0));
    }
    let mut buf = Vec::new();
    q.save(&mut buf);
    assert!(!buf.is_empty(), "saved state must be non-empty");

    // The error bound is overwritten by load().
    let mut loaded = LinearQuantizer::<f32>::new(eb, 32768);
    loaded.load(&mut Reader::new(&buf)).unwrap();
    for (i, (&o, &b)) in originals.iter().zip(&bins).enumerate() {
        let r = loaded.recover(0.0, b).unwrap();
        assert!((r as f64 - o as f64).abs() <= eb, "mismatch at index {i}");
    }
}

fn lossless_round_trip<L: Lossless + Default>() {
    let mut g = Mt32::new(1);
    let src: Vec<u8> = (0..1000).map(|_| (g.next_u32() % 256) as u8).collect();
    let mut dst = Vec::new();
    let size = L::default()
        .compress(&src, &mut dst, Zstd::compress_bound(src.len()) + 8)
        .unwrap();
    assert_eq!(size, dst.len());
    assert_eq!(L::default().decompress(&dst, src.len()).unwrap(), src);
}

// test_lossless.cpp: LosslessTest.LosslessZstd
#[test]
fn lossless_zstd() {
    lossless_round_trip::<Zstd>();
}

// test_lossless.cpp: LosslessTest.LosslessBypass
#[test]
fn lossless_bypass() {
    lossless_round_trip::<Bypass>();
}

// test_lossless.cpp: LosslessTest.LosslessZstdRefusesBufferSmallerThanHeader
#[test]
fn lossless_zstd_refuses_buffer_smaller_than_header() {
    let src = vec![7u8; 100];
    for cap in 0..8 {
        let mut dst = Vec::new();
        assert_eq!(
            Zstd::default().compress(&src, &mut dst, cap),
            Err(Error::BufferTooSmall)
        );
        assert!(dst.is_empty());
    }
}

fn encoder_round_trip<E: Encoder + Default>() {
    let data: Vec<i32> = (0..1000).map(|i| i % 100).collect();
    let (mut table, mut payload) = (Vec::new(), Vec::new());
    let mut enc = E::default();
    enc.preprocess_encode(&data, 100);
    enc.encode(&data, &mut payload).unwrap();
    enc.save(&mut table);

    let mut dec = E::default();
    dec.load(&mut Reader::new(&table)).unwrap();
    assert_eq!(dec.decode(&mut Reader::new(&payload), data.len()).unwrap(), data);
}

// test_encoder.cpp: EncoderTest.HuffmanEncoder
#[test]
fn encoder_huffman() {
    encoder_round_trip::<HuffmanEncoder>();
}

// test_encoder.cpp: EncoderTest.BypassEncoder
#[test]
fn encoder_bypass() {
    encoder_round_trip::<BypassEncoder>();
}
