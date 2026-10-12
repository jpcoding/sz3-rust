# Benchmarks: sz3-rust vs C++ SZ3

`scripts/bench.py`, best of 3 runs, single thread, times reported by each `sz3` tool for the
compression and decompression calls. Both builds use zstd 1.5.7; every Rust stream is
byte-identical to the C++ one (`same`). Speedup > 1 means Rust is faster.

C++ SZ3 3.4.0 (commit d5ab1d1), GCC, `-O3` Release. Rust 1.93 (AMD) / 1.98 (Intel), `--release`.

## AMD Ryzen 7 8845HS (Zen 4)

```
file                         algo                 rel        ratio  C++ comp Rust comp speedup  C++ dec Rust dec speedup same
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-2       125.0     0.062     0.058    1.06    0.030    0.021    1.42 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-4         7.2     0.074     0.070    1.05    0.043    0.030    1.45 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-2        55.9     0.089     0.088    1.02    0.054    0.051    1.07 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-4         7.0     0.133     0.130    1.02    0.070    0.068    1.03 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-2       939.5     0.241     0.221    1.09    0.108    0.072    1.49 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-4        22.3     0.295     0.286    1.03    0.157    0.112    1.41 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-2       647.7     0.374     0.353    1.06    0.221    0.159    1.39 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-4        16.5     0.482     0.447    1.08    0.300    0.235    1.28 True
temperature.f32              ALGO_INTERP_LORENZO  1e-2      7148.6     1.245     1.209    1.03    0.587    0.389    1.51 True
temperature.f32              ALGO_INTERP_LORENZO  1e-4        74.0     1.491     1.440    1.04    0.788    0.557    1.42 True
temperature.f32              ALGO_LORENZO_REG     1e-2      3666.7     1.852     1.700    1.09    1.099    0.827    1.33 True
temperature.f32              ALGO_LORENZO_REG     1e-4        33.0     2.074     1.924    1.08    1.272    1.009    1.26 True
QG-98x1200x1200.f32          ALGO_INTERP_LORENZO  1e-2      5874.7     1.304     1.226    1.06    0.600    0.401    1.50 True
QG-98x1200x1200.f32          ALGO_INTERP_LORENZO  1e-4       229.8     1.483     1.427    1.04    0.747    0.535    1.40 True
QG-98x1200x1200.f32          ALGO_LORENZO_REG     1e-2      2921.7     2.098     1.975    1.06    1.205    0.890    1.35 True
QG-98x1200x1200.f32          ALGO_LORENZO_REG     1e-4       194.1     2.171     2.031    1.07    1.245    0.947    1.31 True
total                                                        15.466    14.584    1.06    8.526    6.303    1.35
```

## Intel Core i5-12500H (Alder Lake)

```
672/672 cases byte-identical and cross-decodable
temperature.dat              ALGO_INTERP_LORENZO  1e-2      7148.6     1.458     1.309    1.11    0.548    0.412    1.33 True
temperature.dat              ALGO_INTERP_LORENZO  1e-4        74.0     1.647     1.505    1.09    0.732    0.583    1.26 True
temperature.dat              ALGO_LORENZO_REG     1e-2      3666.7     2.007     1.757    1.14    0.975    0.662    1.47 True
temperature.dat              ALGO_LORENZO_REG     1e-4        33.0     2.184     1.974    1.11    1.161    0.852    1.36 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-2       939.5     0.257     0.240    1.07    0.105    0.080    1.31 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-4        22.3     0.309     0.287    1.08    0.158    0.125    1.26 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-2       647.7     0.374     0.322    1.16    0.189    0.131    1.44 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-4        16.5     0.443     0.427    1.04    0.253    0.211    1.20 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-2       125.0     0.068     0.061    1.11    0.033    0.031    1.08 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-4         7.2     0.080     0.073    1.09    0.050    0.040    1.25 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-2        55.9     0.096     0.097    0.99    0.058    0.051    1.13 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-4         7.0     0.143     0.144    0.99    0.078    0.072    1.09 True
total                                                         9.066     8.196    1.11    4.340    3.250    1.34
```

Data: SDRBench NYX temperature (512³), Hurricane Pf48 (100×500×500), CESM CLDHGH (1800×3600),
SCALE QG (98×1200×1200), all float32.

## Where the speed comes from

The stream format is SZ3's; the differences are in how the decoder is organized, and C++ could
adopt each of them.

- **Bins are streamed.** The Huffman decoder fills a small chunk at a time that the decomposition
  consumes in order (`BinStream`), instead of materializing every bin first. On NYX that removes a
  512 MB array, its page faults and its release: decompression peak memory is 545 MB against
  1.05 GB for C++, and about 200 ms of 590 ms.
- **Several Huffman codes per lookup.** A 12-bit table entry holds up to three whole short codes,
  breaking the serial lookup-shift chain of one code per lookup.
- **Kernels specialized per predictor and quantizer.** Blockwise Lorenzo/regression and the
  interpolation rows run as monomorphic loops with their state in registers, rather than through
  per-value generic calls.
- **No zero fill** of the output before decompression overwrites it.
