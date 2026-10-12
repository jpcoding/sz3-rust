# Benchmarks: sz3-rust vs C++ SZ3

`scripts/bench.py`, best of 3 runs, single thread, times reported by each `sz3` tool for the
compression and decompression calls. Both builds use zstd 1.5.7; every Rust stream is
byte-identical to the C++ one (`same`). Speedup > 1 means Rust is faster.

C++ SZ3 3.4.0 (commit d5ab1d1), GCC, `-O3` Release. Rust 1.93 (AMD) / 1.98 (Intel), `--release`.

## AMD Ryzen 7 8845HS (Zen 4)

```
file                         algo                 rel        ratio  C++ comp Rust comp speedup  C++ dec Rust dec speedup same
temperature.f32              ALGO_INTERP_LORENZO  1e-2      7148.6     1.260     1.212    1.04    0.597    0.608    0.98 True
temperature.f32              ALGO_INTERP_LORENZO  1e-4        74.0     1.492     1.428    1.04    0.755    0.755    1.00 True
temperature.f32              ALGO_LORENZO_REG     1e-2      3666.7     1.816     1.702    1.07    1.090    0.825    1.32 True
temperature.f32              ALGO_LORENZO_REG     1e-4        33.0     2.064     1.933    1.07    1.283    1.063    1.21 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-2       939.5     0.238     0.228    1.05    0.107    0.108    1.00 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-4        22.3     0.295     0.286    1.03    0.157    0.161    0.98 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-2       647.7     0.372     0.351    1.06    0.221    0.156    1.41 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-4        16.5     0.479     0.446    1.07    0.298    0.261    1.14 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-2       125.0     0.061     0.059    1.03    0.029    0.033    0.90 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-4         7.2     0.074     0.070    1.05    0.044    0.049    0.88 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-2        55.9     0.088     0.087    1.02    0.054    0.046    1.17 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-4         7.0     0.132     0.129    1.02    0.069    0.069    1.00 True
QG-98x1200x1200.f32          ALGO_INTERP_LORENZO  1e-2      5874.7     1.305     1.248    1.05    0.604    0.612    0.99 True
QG-98x1200x1200.f32          ALGO_INTERP_LORENZO  1e-4       229.8     1.490     1.425    1.05    0.746    0.710    1.05 True
QG-98x1200x1200.f32          ALGO_LORENZO_REG     1e-2      2921.7     2.109     1.967    1.07    1.216    0.872    1.39 True
QG-98x1200x1200.f32          ALGO_LORENZO_REG     1e-4       194.1     2.184     2.026    1.08    1.269    0.933    1.36 True
total                                                        15.459    14.599    1.06    8.538    7.261    1.18
```

## Intel Core i5-12500H (Alder Lake)

```
file                         algo                 rel        ratio  C++ comp Rust comp speedup  C++ dec Rust dec speedup same
temperature.dat              ALGO_INTERP_LORENZO  1e-2      7148.6     1.440     1.300    1.11    0.542    0.546    0.99 True
temperature.dat              ALGO_INTERP_LORENZO  1e-4        74.0     1.618     1.498    1.08    0.729    0.730    1.00 True
temperature.dat              ALGO_LORENZO_REG     1e-2      3666.7     2.000     1.745    1.15    0.975    0.695    1.40 True
temperature.dat              ALGO_LORENZO_REG     1e-4        33.0     2.177     1.956    1.11    1.158    0.949    1.22 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-2       939.5     0.264     0.244    1.08    0.106    0.101    1.05 True
Pf48.bin.f32                 ALGO_INTERP_LORENZO  1e-4        22.3     0.312     0.287    1.08    0.159    0.161    0.98 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-2       647.7     0.367     0.326    1.12    0.188    0.136    1.38 True
Pf48.bin.f32                 ALGO_LORENZO_REG     1e-4        16.5     0.447     0.426    1.05    0.255    0.243    1.05 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-2       125.0     0.067     0.061    1.09    0.033    0.035    0.96 True
CLDHGH_1_1800_3600.dat       ALGO_INTERP_LORENZO  1e-4         7.2     0.079     0.073    1.08    0.049    0.053    0.94 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-2        55.9     0.093     0.093    1.00    0.059    0.048    1.22 True
CLDHGH_1_1800_3600.dat       ALGO_LORENZO_REG     1e-4         7.0     0.143     0.144    0.99    0.075    0.073    1.04 True
total                                                         9.006     8.156    1.10    4.328    3.769    1.15
```

Data: SDRBench NYX temperature (512³), Hurricane Pf48 (100×500×500), CESM CLDHGH (1800×3600),
SCALE QG (98×1200×1200), all float32.
