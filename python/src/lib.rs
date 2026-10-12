//! Python bindings: `szConfig` and the compress/decompress core of pysz's `sz` class.

use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use sz3::{Config, SzData};

fn err(e: sz3::Error) -> PyErr {
    match e {
        sz3::Error::InvalidArgument(m) => PyValueError::new_err(m),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Configuration for SZ3 compression: `szConfig(data.shape)`, `szConfig(100, 200, 300)`.
#[pyclass(name = "szConfig", module = "pysz", skip_from_py_object)]
#[derive(Clone)]
pub struct SzConfig {
    pub conf: Config,
}

fn dims_from(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Vec<usize>> {
    let items: Vec<Bound<'_, PyAny>> = if args.len() == 1 && args.get_item(0)?.hasattr("__iter__")? {
        args.get_item(0)?.try_iter()?.collect::<PyResult<_>>()?
    } else {
        args.iter().collect()
    };
    if items.is_empty() {
        return Err(PyValueError::new_err("At least one dimension required"));
    }
    items
        .iter()
        .map(|a| match a.extract::<i64>() {
            Ok(v) if v > 0 => Ok(v as usize),
            _ => Err(PyValueError::new_err(format!(
                "Dimension must be positive integer, got {a}"
            ))),
        })
        .collect()
}

#[pymethods]
impl SzConfig {
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let mut c = SzConfig {
            conf: Config::default(),
        };
        if !args.is_empty() {
            c.set_dims(args)?;
        }
        Ok(c)
    }

    /// Set dimensions: a tuple/list or individual integers.
    #[pyo3(name = "setDims", signature = (*args))]
    fn set_dims(&mut self, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<()> {
        let d = dims_from(args)?;
        self.conf.set_dims(&d);
        Ok(())
    }

    /// Load configuration from an INI file.
    fn loadcfg(&mut self, cfgpath: &str) -> PyResult<()> {
        self.conf
            .load_cfg(cfgpath)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load '{cfgpath}': {e}")))
    }

    #[getter]
    fn dims(&self) -> Vec<usize> {
        self.conf.dims.clone()
    }
    #[getter]
    fn num_elements(&self) -> usize {
        self.conf.num
    }
    #[getter]
    fn ndim(&self) -> u8 {
        self.conf.n
    }
    #[getter(absErrorBound)]
    fn abs_error_bound(&self) -> f64 {
        self.conf.abs_error_bound
    }
    #[setter(absErrorBound)]
    fn set_abs_error_bound(&mut self, v: f64) {
        self.conf.abs_error_bound = v;
    }
    #[getter(relErrorBound)]
    fn rel_error_bound(&self) -> f64 {
        self.conf.rel_error_bound
    }
    #[setter(relErrorBound)]
    fn set_rel_error_bound(&mut self, v: f64) {
        self.conf.rel_error_bound = v;
    }
    #[getter(psnrErrorBound)]
    fn psnr_error_bound(&self) -> f64 {
        self.conf.psnr_error_bound
    }
    #[setter(psnrErrorBound)]
    fn set_psnr_error_bound(&mut self, v: f64) {
        self.conf.psnr_error_bound = v;
    }
    #[getter(l2normErrorBound)]
    fn l2norm_error_bound(&self) -> f64 {
        self.conf.l2norm_error_bound
    }
    #[setter(l2normErrorBound)]
    fn set_l2norm_error_bound(&mut self, v: f64) {
        self.conf.l2norm_error_bound = v;
    }
    #[getter(errorBoundMode)]
    fn error_bound_mode(&self) -> u8 {
        self.conf.error_bound_mode
    }
    #[setter(errorBoundMode)]
    fn set_error_bound_mode(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.conf.error_bound_mode = enum_value(v)?;
        Ok(())
    }
    #[getter(cmprAlgo)]
    fn cmpr_algo(&self) -> u8 {
        self.conf.cmpr_algo
    }
    #[setter(cmprAlgo)]
    fn set_cmpr_algo(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.conf.cmpr_algo = enum_value(v)?;
        Ok(())
    }
    #[getter]
    fn openmp(&self) -> bool {
        self.conf.openmp
    }
    #[setter]
    fn set_openmp(&mut self, v: bool) {
        self.conf.openmp = v;
    }

    fn __repr__(&self) -> String {
        let d: Vec<String> = self.conf.dims.iter().map(|d| d.to_string()).collect();
        let t = if d.len() == 1 {
            format!("({},)", d[0])
        } else {
            format!("({})", d.join(", "))
        };
        format!("szConfig(dims={t}, num_elements={})", self.conf.num)
    }
}

/// An int, or an enum member with an int `value`.
fn enum_value(v: &Bound<'_, PyAny>) -> PyResult<u8> {
    if let Ok(i) = v.extract::<u8>() {
        return Ok(i);
    }
    if let Ok(i) = v.getattr("value").and_then(|x| x.extract::<u8>()) {
        return Ok(i);
    }
    Err(PyTypeError::new_err(format!(
        "Expected int or Enum, got {}",
        v.get_type().name()?
    )))
}

fn compress_as<'py, T: SzData + numpy::Element>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    conf: &Config,
) -> PyResult<Option<(Bound<'py, PyArray1<u8>>, usize)>> {
    let Ok(arr) = data.cast::<PyArrayDyn<T>>() else {
        return Ok(None);
    };
    let ro = arr.readonly();
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let conf = conf.clone();
    let bytes = py.detach(|| sz3::compress(&conf, slice)).map_err(err)?;
    let n = bytes.len();
    Ok(Some((bytes.into_pyarray(py), n)))
}

/// Compresses a C-contiguous array (float32, float64, int32, int64); its shape becomes
/// `config`'s dimensions. Returns (uint8 array, ratio).
#[pyfunction]
fn compress<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    config: &Bound<'py, SzConfig>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, f64)> {
    let shape: Vec<usize> = data.getattr("shape")?.extract()?;
    let nbytes: usize = data.getattr("nbytes")?.extract()?;
    {
        let mut c = config.borrow_mut();
        if shape.is_empty() || shape.contains(&0) {
            return Err(PyValueError::new_err("data must have at least one element"));
        }
        c.conf.set_dims(&shape);
    }
    let conf = config.borrow().conf.clone();
    let r = if let Some(r) = compress_as::<f32>(py, data, &conf)? {
        r
    } else if let Some(r) = compress_as::<f64>(py, data, &conf)? {
        r
    } else if let Some(r) = compress_as::<i32>(py, data, &conf)? {
        r
    } else if let Some(r) = compress_as::<i64>(py, data, &conf)? {
        r
    } else {
        let dt = data.getattr("dtype")?;
        return Err(PyTypeError::new_err(format!(
            "Unsupported dtype: {dt}. Supported: float32, float64, int32, int64"
        )));
    };
    Ok((r.0, nbytes as f64 / r.1 as f64))
}

fn decompress_as<'py, T: SzData + numpy::Element>(
    py: Python<'py>,
    bytes: &[u8],
) -> PyResult<(Bound<'py, PyAny>, Config)> {
    let (conf, v) = py.detach(|| sz3::decompress::<T>(bytes)).map_err(err)?;
    Ok((v.into_pyarray(py).into_any(), conf))
}

/// Decompresses into a flat array of `dtype` ("float32", "float64", "int32", "int64"), checking
/// the element count against `num`. Returns (array, szConfig of the stream).
#[pyfunction]
fn decompress<'py>(
    py: Python<'py>,
    compressed: &Bound<'py, PyArray1<u8>>,
    dtype: &str,
    num: usize,
) -> PyResult<(Bound<'py, PyAny>, SzConfig)> {
    let ro = compressed.readonly();
    let bytes = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("compressed data must be contiguous"))?;
    // Check the stored element count before decompressing anything.
    let stored = sz3::api::stream_config(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if stored.num != num {
        return Err(PyValueError::new_err(format!(
            "shape holds {num} values, the compressed data {}",
            stored.num
        )));
    }
    let (arr, conf) = match dtype {
        "float32" => decompress_as::<f32>(py, bytes)?,
        "float64" => decompress_as::<f64>(py, bytes)?,
        "int32" => decompress_as::<i32>(py, bytes)?,
        "int64" => decompress_as::<i64>(py, bytes)?,
        d => {
            return Err(PyTypeError::new_err(format!(
                "Unsupported dtype: {d}. Supported: float32, float64, int32, int64"
            )));
        }
    };
    Ok((arr, SzConfig { conf }))
}

#[pymodule]
fn _pysz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SzConfig>()?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add("SZ3_VERSION", sz3::config::SZ3_VERSION)?;
    Ok(())
}
