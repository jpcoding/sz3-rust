# HDF5 filter SZ3 (id 32024)

`libhdf5sz3.so`: the SZ3 HDF5 filter plugin, implemented with sz3-rust. Same filter id, cd_values
layout and chunk format as SZ3's H5Z-SZ3, so files written by either read with the other; it also
reads files written by SZ3 3.3.2's filter. `H5Pset_sz3` is exported (`include/H5Z_SZ3.h`).

```sh
cargo build --release -p sz3-h5z
mkdir -p plugins && cp target/release/libhdf5sz3.so plugins/
export HDF5_PLUGIN_PATH=$PWD/plugins
```

The plugin calls HDF5 from the process that loads it. In Python, load h5py's HDF5 globally first:

```python
import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
import h5py
from cdvalueHelper import SZ3          # tests/cdvalueHelper.py, from SZ3
f.create_dataset("x", data=a, chunks=True, compression=32024,
                 compression_opts=tuple(SZ3("ALGO_INTERP_LORENZO", absolute=1e-3).cd_values))
```

`tests/test_h5z.py` checks round trips, SZ3's 3.3.2 fixture, the stored cd_values, and, given C++
SZ3's `sz3` binary, that C++ SZ3 decodes this filter's chunks and this filter reads C++ chunks.
