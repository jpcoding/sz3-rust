# SZ3 in Rust — rewrite plan

Port of [SZ3](https://github.com/szcompressor/SZ3) (reference: v3.4.0, data format 3.4.0) to Rust.

## Goals

1. **Same modular design as SZ3.** The pipeline is built from small, swappable stages, each a Rust trait:

   ```
   data ─► Decomposition (Predictor + Quantizer) ─► bins ─► Encoder ─► bytes ─► Lossless (zstd)
   ```

   | SZ3 concept (C++)          | Rust trait / type                          |
   |----------------------------|--------------------------------------------|
   | `concepts::PredictorInterface` | `predictor::Predictor<T, N>`            |
   | `concepts::QuantizerInterface` | `quantizer::Quantizer<T>`               |
   | `concepts::DecompositionInterface` | `decomposition::Decomposition<T>`   |
   | `concepts::EncoderInterface`   | `encoder::Encoder`                      |
   | `concepts::LosslessInterface`  | `lossless::Lossless`                    |
   | `SZGenericCompressor`          | `compressor::GenericCompressor<D,E,L>`  |

   Everything above zstd is native Rust that we own and control. zstd is the only
   external codec (via `zstd-safe`).

2. **Byte-compatible format.** A stream written by the Rust port decodes with C++ SZ3 and
   vice versa. Where the C++ code is deterministic (all non-OpenMP paths), the Rust port
   aims to produce *identical* bytes, which makes the C++ build a test oracle.
   Floating-point arithmetic replicates the C++ evaluation order and float/double promotion.

3. **Safe Rust.** No `unsafe` in the pipeline; every read of untrusted input is bounds checked
   and returns `Err`, never panics.

## Order of work

1. **1:1 parity** with C++ SZ3 — every feature, format path and tool, verified byte-for-byte.
2. **Performance** — then beat C++ on both AMD (Zen 4) and Intel (Alder Lake) machines, keeping
   streams byte-identical.

## Parity checklist

| Area | Item | Status |
|------|------|--------|
| Core | Config (binary + INI), header, version checks | done |
| Core | Huffman (3.4.0), linear quantizer, zstd/bypass lossless | done |
| Algorithms | INTERP_LORENZO (tuning), INTERP, LORENZO_REG, NOPRED, LOSSLESS | done |
| Algorithms | ALGO_BIOMD (SZBioMDDecomposition + SegmentedEncoder) | done |
| Data | f32/f64, 8/16/32/64-bit integers, 1–4 D, all error-bound modes | done |
| Format | read OpenMP-chunked streams | done |
| Format | write OpenMP-chunked streams (multithreaded compression) | done |
| Format | read 3.3.2 streams (HuffmanEncoderV1), golden streams from SZ3's test suite | done |
| Format | SZ3's official format digest (`data_format_digest.txt`, 3.4.0) | done |
| Modules | BypassEncoder, TimeSeriesDecomposition, SZExaaltCompressor, k-means utils | todo |
| Tools | `sz3` CLI | done |
| Tools | C API (`sz3c.h`), Python bindings (pysz), HDF5 filter (H5Z-SZ3), `mdz` | todo |
| Tests | port SZ3's module tests (tools/test/modules) | todo |
| Verification | cross-check matrix (692 cases incl. OpenMP and BioMD) + SDRBench real data, AMD and Intel | done |

## Layout

```
src/
  lib.rs            public API: compress / decompress / size bound
  error.rs          Error type
  bytes.rs          little-endian writer/reader, bit-packed dims
  config.rs         Config, enums, binary save/load, INI
  stats.rs          error-bound conversion, verification metrics
  lossless.rs       Lossless trait, zstd + bypass
  encoder/          Encoder trait, Huffman (3.4.0 format)
  quantizer.rs      Quantizer trait, LinearQuantizer
  blockwise.rs      padded block storage + block iterator
  predictor/        Predictor trait, Lorenzo, Regression, Composed
  decomposition/    Decomposition trait, NoPrediction, Blockwise, Interpolation
  compressor.rs     GenericCompressor (decomposition + encoder + lossless)
  algo/             per-algorithm assembly, sampling and tuning
  api.rs            header, dispatcher, integer support
  bin/sz3.rs        command line tool
tests/              round trip and C++ cross-check tests
```
