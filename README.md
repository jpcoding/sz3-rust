# sz3-rust

A Rust port of [SZ3](https://github.com/szcompressor/SZ3), the modular error-bounded lossy
compressor for scientific data.

* **Same modular design.** The pipeline is built from SZ3's modules as Rust traits —
  `Decomposition` (`Predictor` + `Quantizer`) → `Encoder` → `Lossless` — composed by
  `GenericCompressor`. Everything above zstd is native Rust.
* **Byte-compatible with SZ3 3.4.0.** Streams written by this port decode with C++ SZ3 and vice
  versa, and for the same input and settings the compressed bytes are identical (given the same
  zstd version). See [Verification](#verification).
* **Safe.** No `unsafe`; damaged or hostile streams return an error instead of panicking or
  allocating without bound.

## Status

| Feature | Status |
|---|---|
| `ALGO_INTERP_LORENZO` (default, auto-tuned), `ALGO_INTERP`, `ALGO_LORENZO_REG`, `ALGO_NOPRED`, `ALGO_LOSSLESS` | done |
| Error bounds ABS, REL, PSNR, NORM, ABS_AND_REL, ABS_OR_REL | done |
| `f32`, `f64`, and 8/16/32/64-bit integers; 1–4 dimensions | done |
| `sz3` CLI (same arguments as the C++ tool), INI config files | done |
| Reading OpenMP-chunked streams | done |
| Writing OpenMP-chunked streams (multithreaded compression) | planned |
| Reading 3.3.2 streams (`HuffmanEncoderV1`) | planned |
| `ALGO_BIOMD`, C API, Python bindings, HDF5 filter | planned |

See [PLAN.md](PLAN.md) for the roadmap.

## Library

```rust
use sz3::{Config, compress, decompress, config::{algo, eb}};

let dims = [100, 500, 500];                  // slowest dimension first
let data: Vec<f32> = load_field();
let mut conf = Config::new(&dims);
conf.cmpr_algo = algo::INTERP_LORENZO;
conf.error_bound_mode = eb::REL;
conf.rel_error_bound = 1e-4;

let bytes = compress(&conf, &data)?;
let (conf, out) = decompress::<f32>(&bytes)?;
```

Custom pipelines are assembled from the modules, as in SZ3:

```rust
use sz3::compressor::GenericCompressor;
use sz3::decomposition::BlockwiseDecomposition;
use sz3::encoder::HuffmanEncoder;
use sz3::lossless::Zstd;
use sz3::predictor::LorenzoPredictor;
use sz3::quantizer::LinearQuantizer;

let pred = LorenzoPredictor::<f32, 3, 1>::new(conf.abs_error_bound);
let quant = LinearQuantizer::new(conf.abs_error_bound, conf.quantbin_cnt / 2);
let mut c = GenericCompressor::new(
    BlockwiseDecomposition::<f32, 3, _, _>::new(&conf, pred, quant),
    HuffmanEncoder::new(),
    Zstd::default(),
);
```

## Command line

```sh
cargo build --release
./target/release/sz3 -f -i data.f32 -z data.sz -3 500 500 100 -M REL 1e-4
./target/release/sz3 -f -s data.sz -o data.out -i data.f32 -a
```

## Verification

`scripts/crosscheck.py` runs a matrix of synthetic datasets (1–4 D, `f32`/`f64`/`i32`/`i64`, all
algorithms, ABS/REL/ABS_AND_REL bounds, predictor variants) through both the C++ and the Rust
`sz3`, and checks that the compressed streams are byte-identical and that each tool decodes the
other's stream to the same values. `scripts/realdata.py` does the same on SDRBench fields
(Hurricane, NYX, CESM, SCALE) and reports ratio and speed.

Byte-identical output needs both builds on the same zstd version: build C++ SZ3 with
`-DSZ3_USE_BUNDLED_ZSTD=OFF` against the zstd the `zstd-sys` crate bundles (1.5.7). With
different zstd versions the streams still cross-decode.

```sh
python3 scripts/crosscheck.py /path/to/cpp/sz3 target/release/sz3 <data dir> <work dir>
```
