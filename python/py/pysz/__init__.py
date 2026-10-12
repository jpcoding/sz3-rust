"""SZ3 error-bounded lossy compression for NumPy arrays, implemented in Rust (sz3-rust).

Same API as SZ3's pysz: ``sz.compress``, ``sz.decompress``, ``sz.verify``, ``szConfig``,
``szErrorBoundMode``, ``szAlgorithm``.
"""

from typing import Tuple

import numpy as np

from pysz._pysz import SZ3_VERSION, szConfig
from pysz import _pysz

__all__ = ["sz", "szConfig", "szErrorBoundMode", "szAlgorithm"]


class szErrorBoundMode:
    """Error bound modes"""

    ABS = 0
    REL = 1
    PSNR = 2
    L2NORM = 3
    ABS_AND_REL = 4
    ABS_OR_REL = 5


class szAlgorithm:
    """Compression algorithms"""

    LORENZO_REG = 0
    INTERP_LORENZO = 1
    INTERP = 2
    NOPRED = 3
    LOSSLESS = 4
    BIOMD = 5


class sz:
    """SZ3 compression and decompression of NumPy arrays."""

    _SUPPORTED_DTYPES = frozenset(np.dtype(t) for t in (np.float32, np.float64, np.int32, np.int64))

    @staticmethod
    def compress(data: np.ndarray, config: szConfig) -> Tuple[np.ndarray, float]:
        """Compress ``data``; ``config`` takes its shape. Returns (uint8 array, compression ratio)."""
        if not isinstance(data, np.ndarray) or data.dtype not in sz._SUPPORTED_DTYPES:
            raise TypeError(
                f"Unsupported dtype: {getattr(data, 'dtype', type(data))}. Supported: float32, float64, int32, int64"
            )
        if not isinstance(config, szConfig):
            raise TypeError(f"config must be szConfig, got {type(config)}")
        return _pysz.compress(np.ascontiguousarray(data), config)

    @staticmethod
    def decompress(compressed: np.ndarray, dtype, shape) -> Tuple[np.ndarray, szConfig]:
        """Decompress to an array of ``dtype`` and ``shape``. Returns (array, szConfig of the stream)."""
        if compressed.dtype != np.uint8:
            raise TypeError(f"Compressed data must be uint8, got {compressed.dtype}")
        dtype = np.dtype(dtype)
        if dtype not in sz._SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype: {dtype}. Supported: float32, float64, int32, int64")
        if isinstance(shape, int):
            shape = (shape,)
        num = szConfig(*shape).num_elements
        out, conf = _pysz.decompress(np.ascontiguousarray(compressed).reshape(-1), dtype.name, num)
        return out.reshape(shape), conf

    @staticmethod
    def verify(src_data: np.ndarray, dec_data: np.ndarray) -> Tuple[float, float, float]:
        """(max absolute difference, PSNR in dB, NRMSE) of ``dec_data`` against ``src_data``."""
        if src_data.ndim != dec_data.ndim or src_data.size != dec_data.size:
            raise ValueError("Shape mismatch between src_data and dec_data")
        if src_data.dtype != dec_data.dtype:
            raise ValueError("Dtype mismatch between src_data and dec_data")
        if np.issubdtype(src_data.dtype, np.integer):
            src_data = src_data.astype(np.float64)
            dec_data = dec_data.astype(np.float64)
        data_range = float(np.max(src_data) - np.min(src_data))
        diff = src_data - dec_data
        max_diff = float(np.max(np.abs(diff)))
        mse = float(np.mean(diff**2))
        nrmse = float(np.sqrt(mse) / data_range) if data_range > 0 else 0.0
        psnr = float(20 * np.log10(data_range) - 10 * np.log10(mse)) if mse > 0 else float("inf")
        return max_diff, psnr, nrmse
