"""The sz3-rust HDF5 filter in h5py: round trips, SZ3's 3.3.2 fixture, cd_values layout, and chunks
that are SZ3 streams.

usage: HDF5_PLUGIN_PATH=<dir with libhdf5sz3.so> python test_h5z.py [cpp sz3 binary]
With the C++ sz3 CLI given, chunks written by this filter are decoded by C++ SZ3, and a chunk
compressed by C++ SZ3 is read through this filter.
"""
import os
import subprocess
import sys
import tempfile

# The plugin resolves HDF5's symbols from the process, so h5py's HDF5 must be loaded globally.
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
import h5py  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from cdvalueHelper import SZ3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_SZ3 = sys.argv[1] if len(sys.argv) > 1 else None
failures = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        failures.append(name)


def field(shape, dtype, scale=1.0):
    g = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    f = sum(np.sin((3 + k) * np.pi * x) for k, x in enumerate(g)) + np.random.default_rng(1).normal(0, 0.01, shape)
    return (f * scale).astype(dtype)


def main():
    check("filter-available", h5py.h5z.filter_avail(32024))
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "t.h5")
    cases = [
        ("f32-interp-lorenzo", field((40, 50, 60), np.float32), (20, 50, 60), "ALGO_INTERP_LORENZO", 1e-3),
        ("f64-lorenzo-reg", field((300, 400), np.float64), (100, 400), "ALGO_LORENZO_REG", 1e-6),
        ("i32-interp", field((100, 120), np.int32, 1000), (100, 120), "ALGO_INTERP", 2.0),
        ("f32-1d-tiny-chunks", field((1000,), np.float32), (7,), "ALGO_INTERP_LORENZO", 1e-3),
    ]
    with h5py.File(path, "w") as f:
        for name, data, chunks, algo, eb in cases:
            f.create_dataset(name, data=data, chunks=chunks, compression=32024,
                             compression_opts=tuple(SZ3(algo, absolute=eb).cd_values))
        # no cd_values: the defaults (ALGO_INTERP_LORENZO, ABS 1e-3)
        f.create_dataset("defaults", data=field((64, 64), np.float32), chunks=(32, 64), compression=32024)
    with h5py.File(path, "r") as f:
        for name, data, chunks, algo, eb in cases:
            out = f[name][...]
            err = np.max(np.abs(out.astype(np.float64) - data.astype(np.float64)))
            check(f"round-trip-{name}", err <= eb, f"max error {err} > {eb}")
            raw = data.nbytes
            stored = f[name].id.get_storage_size()
            check(f"compresses-{name}", stored < raw or name.endswith("tiny-chunks"), f"{stored} >= {raw}")
        err = np.max(np.abs(f["defaults"][...] - field((64, 64), np.float32)))
        check("default-cd-values", err <= 1e-3, f"max error {err}")
        # set_local stores [data version, Config] in cd_values
        cd = f["f32-interp-lorenzo"].id.get_create_plist().get_filter(0)[2]  # (code, flags, values, name)
        check("cd-values-carry-data-version", cd[0] == (3 << 24 | 4 << 16), f"cd_values[0] = {cd[0]:#x}")
        ds = f["f32-interp-lorenzo"]
        chunk = ds.id.read_direct_chunk((0, 0, 0))[1]
    check("chunk-is-sz3-stream", chunk[:4] == (0xF342F310).to_bytes(4, "little"))

    # SZ3's fixture, written by the 3.3.2 filter
    with h5py.File(os.path.join(HERE, "data", "sz3_data_3.3.2.h5"), "r") as f:
        ok = True
        for k in f:
            if isinstance(f[k], h5py.Dataset):
                v = f[k][...]
                ok = ok and v.size > 0 and np.all(np.isfinite(v))
        check("previous-data-version-reads-back", ok)

    if CPP_SZ3:
        # a chunk this filter wrote, decoded by C++ SZ3
        cpath = os.path.join(tmp, "chunk.sz")
        with open(cpath, "wb") as fh:
            fh.write(chunk)
        out = os.path.join(tmp, "chunk.out")
        r = subprocess.run([CPP_SZ3, "-f", "-s", cpath, "-o", out], capture_output=True, text=True)
        dec = np.fromfile(out, np.float32).reshape(20, 50, 60) if r.returncode == 0 else None
        with h5py.File(path, "r") as f:
            mine = f["f32-interp-lorenzo"][0:20]
        check("cpp-decodes-rust-chunk", dec is not None and np.array_equal(dec, mine), r.stderr[:200])
        # a chunk C++ SZ3 compressed, read through this filter
        data = field((20, 50, 60), np.float32)
        src = os.path.join(tmp, "src.f32")
        data.tofile(src)
        czp = os.path.join(tmp, "cpp.sz")
        subprocess.run([CPP_SZ3, "-f", "-i", src, "-z", czp, "-3", "60", "50", "20", "-M", "ABS", "1e-3"],
                       check=True, capture_output=True)
        cpp_chunk = open(czp, "rb").read()
        subprocess.run([CPP_SZ3, "-f", "-s", czp, "-o", out], check=True, capture_output=True)
        cpp_dec = np.fromfile(out, np.float32).reshape(20, 50, 60)
        p2 = os.path.join(tmp, "direct.h5")
        with h5py.File(p2, "w") as f:
            ds = f.create_dataset("d", shape=(20, 50, 60), dtype=np.float32, chunks=(20, 50, 60), compression=32024,
                                  compression_opts=tuple(SZ3("ALGO_INTERP_LORENZO", absolute=1e-3).cd_values))
            ds.id.write_direct_chunk((0, 0, 0), cpp_chunk)
        with h5py.File(p2, "r") as f:
            check("rust-reads-cpp-chunk", np.array_equal(f["d"][...], cpp_dec))

    print(f"{'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
