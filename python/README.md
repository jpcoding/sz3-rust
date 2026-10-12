# pysz (Rust)

Python bindings of sz3-rust with the API of SZ3's `pysz`, so code written for pysz runs
unchanged: `sz.compress`, `sz.decompress`, `sz.verify`, `szConfig`, `szErrorBoundMode`,
`szAlgorithm`. Streams are SZ3 3.4.0 streams. The GIL is released while compressing.

```sh
pip install maturin
maturin develop --release        # or: maturin build --release
python -m pytest tests/          # tests/test_pysz.py is pysz's own test file
```

```python
import numpy as np
from pysz import sz, szConfig, szErrorBoundMode

data = np.random.rand(100, 200).astype(np.float32)
conf = szConfig()
conf.errorBoundMode = szErrorBoundMode.REL
conf.relErrorBound = 1e-4
compressed, ratio = sz.compress(data, conf)
out, stored = sz.decompress(compressed, data.dtype, data.shape)
```
