#![allow(non_snake_case)]
//! SZ3's C API (`sz3c.h`) over sz3-rust: the same symbols and semantics, so C, Fortran and other
//! callers of `libSZ3c` can switch libraries without changes.
//!
//! Errors return NULL with a message on stderr. Buffers are `malloc`ed; free them with
//! `free_buf()` (or `free()`).

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use sz3::config::eb;
use sz3::{Config, SzData};

const SZ_FLOAT: c_int = 0;
const SZ_DOUBLE: c_int = 1;
// SZ2's error-bound modes, as sz3c.h numbers them
const ABS: c_int = 0;
const REL: c_int = 1;
const ABS_AND_REL: c_int = 2;
const ABS_OR_REL: c_int = 3;

/// Copies `v` into a `malloc`ed buffer.
fn to_malloc(v: &[u8]) -> *mut u8 {
    // SAFETY: a fresh allocation of v.len() bytes (at least 1), filled before it is returned.
    unsafe {
        let p = libc::malloc(v.len().max(1)) as *mut u8;
        if !p.is_null() {
            std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len());
        }
        p
    }
}

fn guard<R>(null: R, f: impl FnOnce() -> Result<R, String>) -> R {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            eprintln!("SZ3: {e}");
            null
        }
        Err(_) => {
            eprintln!("SZ3: internal error");
            null
        }
    }
}

fn compress_typed<T: SzData>(conf: &Config, data: *const c_void, out_size: *mut usize) -> Result<*mut u8, String> {
    if data.is_null() || out_size.is_null() {
        return Err("null argument".into());
    }
    // SAFETY: the caller passes conf.num values of T, as sz3c.h requires.
    let values = unsafe { std::slice::from_raw_parts(data as *const T, conf.num) };
    let bytes = sz3::compress(conf, values).map_err(|e| e.to_string())?;
    // SAFETY: checked non-null above.
    unsafe { *out_size = bytes.len() };
    Ok(to_malloc(&bytes))
}

/// Compresses `data` of `dataType` with dimensions `r5..r1` (r1 fastest; trailing zeros unused).
///
/// # Safety
/// `data` must hold the values the dimensions give; `outSize` must be writable.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn SZ_compress_args(
    data_type: c_int,
    data: *mut c_void,
    out_size: *mut usize,
    err_bound_mode: c_int,
    abs_err_bound: f64,
    rel_bound_ratio: f64,
    _pwr_bound_ratio: f64,
    r5: usize,
    r4: usize,
    r3: usize,
    r2: usize,
    r1: usize,
) -> *mut u8 {
    guard(std::ptr::null_mut(), || {
        let dims: Vec<usize> = if r2 == 0 {
            vec![r1]
        } else if r3 == 0 {
            vec![r2, r1]
        } else if r4 == 0 {
            vec![r3, r2, r1]
        } else if r5 == 0 {
            vec![r4, r3, r2, r1]
        } else {
            vec![r5 * r4, r3, r2, r1]
        };
        let mut conf = Config::new(&dims);
        conf.abs_error_bound = abs_err_bound;
        conf.rel_error_bound = rel_bound_ratio;
        conf.error_bound_mode = match err_bound_mode {
            ABS => eb::ABS,
            REL => eb::REL,
            ABS_AND_REL => eb::ABS_AND_REL,
            ABS_OR_REL => eb::ABS_OR_REL,
            m => return Err(format!("errBoundMode {m} not supported")),
        };
        match data_type {
            SZ_FLOAT => compress_typed::<f32>(&conf, data, out_size),
            SZ_DOUBLE => compress_typed::<f64>(&conf, data, out_size),
            t => Err(format!("dataType {t} not supported")),
        }
    })
}

fn decompress_typed<T: SzData>(bytes: &[u8], n: usize) -> Result<*mut c_void, String> {
    let (conf, out) = sz3::decompress::<T>(bytes).map_err(|e| e.to_string())?;
    if conf.num != n {
        return Err(format!(
            "the data holds {} values, not the {n} the dimensions give",
            conf.num
        ));
    }
    let mut raw = Vec::with_capacity(n * T::SIZE);
    for v in out {
        v.write_le(&mut raw);
    }
    Ok(to_malloc(&raw) as *mut c_void)
}

/// Decompresses `byteLength` bytes; the dimensions must match the stream's.
///
/// # Safety
/// `bytes` must point to `byteLength` readable bytes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn SZ_decompress(
    data_type: c_int,
    bytes: *mut u8,
    byte_length: usize,
    r5: usize,
    r4: usize,
    r3: usize,
    r2: usize,
    r1: usize,
) -> *mut c_void {
    guard(std::ptr::null_mut(), || {
        let n = if r2 == 0 {
            r1
        } else if r3 == 0 {
            r1 * r2
        } else if r4 == 0 {
            r1 * r2 * r3
        } else if r5 == 0 {
            r1 * r2 * r3 * r4
        } else {
            r1 * r2 * r3 * r4 * r5
        };
        if bytes.is_null() {
            return Err("null argument".into());
        }
        // SAFETY: the caller passes byteLength readable bytes.
        let b = unsafe { std::slice::from_raw_parts(bytes, byte_length) };
        match data_type {
            SZ_FLOAT => decompress_typed::<f32>(b, n),
            SZ_DOUBLE => decompress_typed::<f64>(b, n),
            t => Err(format!("dataType {t} not supported")),
        }
    })
}

/// Frees a buffer returned by this library.
///
/// # Safety
/// `p` must come from `SZ_compress_args` or `SZ_decompress`, or be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_buf(p: *mut c_void) {
    // SAFETY: p was malloc'ed by this library, or is NULL.
    unsafe { libc::free(p) }
}
