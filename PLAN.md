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

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Core: `Config` (binary + INI), byte utils, zstd lossless, canonical Huffman encoder, linear quantizer, error-bound modes | done |
| 2 | Algorithms: `ALGO_NOPRED`, `ALGO_LORENZO_REG` (Lorenzo 1st/2nd order, regression, composed predictor, blockwise decomposition), `ALGO_INTERP`, `ALGO_INTERP_LORENZO` (sampling + auto-tuning), `ALGO_LOSSLESS`; f32/f64 and integer input | done |
| 3 | `sz3` CLI with the same arguments as the C++ tool; `-a` statistics | done |
| 4 | Cross-validation against C++ SZ3: Rust→C++ decode, C++→Rust decode, byte-identical compressed output, error-bound property tests | done (456/456 synthetic cases + SDRBench fields byte-identical) |
| 5 | OpenMP-compatible chunked layout (`openmp = true`): reading done; multithreaded writing planned |
| 6 | Read 3.3.2 streams (`HuffmanEncoderV1`) | planned |
| 7 | `ALGO_BIOMD` (molecular dynamics), segmented encoder | planned |
| 8 | C ABI (`sz3c.h` compatible) + Python bindings, HDF5 filter | planned |
| 9 | Performance: currently 1.2-2x slower than C++; profile hot loops (Huffman decode, blockwise predict, interpolation), remove per-element dyn dispatch and bounds checks | next |

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
