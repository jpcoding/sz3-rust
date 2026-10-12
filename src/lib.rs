//! A Rust port of [SZ3](https://github.com/szcompressor/SZ3), the error-bounded lossy compressor
//! for scientific data.
//!
//! The pipeline is assembled from the same modules as SZ3:
//!
//! ```text
//! data ─► Decomposition (Predictor + Quantizer) ─► bins ─► Encoder ─► Lossless (zstd)
//! ```
//!
//! Streams are byte-compatible with SZ3 3.4.0.
//!
//! ```
//! use sz3::{Config, compress, decompress, config::eb};
//!
//! let data: Vec<f32> = (0..64 * 64).map(|i| ((i % 64) as f32 * 0.1).sin()).collect();
//! let mut conf = Config::new(&[64, 64]);
//! conf.error_bound_mode = eb::ABS;
//! conf.abs_error_bound = 1e-3;
//! let bytes = compress(&conf, &data).unwrap();
//! let (_, out) = decompress::<f32>(&bytes).unwrap();
//! assert!(data.iter().zip(&out).all(|(a, b)| (a - b).abs() <= 1e-3));
//! ```

// Comparisons are written as SZ3 writes them so NaN takes the same branch; loops index as SZ3 does.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

pub mod algo;
pub mod api;
pub mod blockwise;
pub mod bytes;
pub mod compressor;
pub mod config;
pub mod decomposition;
pub mod encoder;
pub mod error;
pub mod kmeans;
pub mod lossless;
pub mod mdz;
pub mod predictor;
pub mod quantizer;
pub mod stats;
pub mod types;

pub use api::{compress, compress_size_bound, compress_with, compress_with_capacity, decompress};
pub use config::Config;
pub use error::{Error, Result};
pub use types::{SzData, SzFloat};
