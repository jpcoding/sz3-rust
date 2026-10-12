//! HDF5 filter plugin SZ3 (filter id 32024), compatible with SZ3's H5Z-SZ3: the same cd_values
//! layout and chunk format, so files written by either filter read with the other.
//!
//! Build it as a shared library and put it in `HDF5_PLUGIN_PATH`. The HDF5 symbols it calls
//! resolve from the HDF5 library already loaded in the process, as HDF5 plugins usually do.
//!
//! cd_values: the data version, then the serialized SZ3 `Config` (little-endian u32 words); SZ3
//! 3.3.2 stored the `Config` alone. Each chunk is a complete SZ3 stream.
#![allow(non_upper_case_globals, non_camel_case_types)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use sz3::bytes::Reader;
use sz3::config::{SZ3_DATA_VER, SZ3_DATA_VER_OLDEST, SZ3_MAGIC_NUMBER, algo, version_int, version_str};
use sz3::{Config, SzData};

pub const H5Z_FILTER_SZ3: c_int = 32024;

pub type hid_t = i64;
pub type herr_t = c_int;
pub type htri_t = c_int;
pub type hsize_t = u64;
type H5Z_filter_t = c_int;

const H5Z_CLASS_T_VERS: c_int = 1;
const H5Z_FLAG_MANDATORY: c_uint = 0x0000;
const H5Z_FLAG_REVERSE: c_uint = 0x0100;
const H5PL_TYPE_FILTER: c_int = 0;
const H5S_MAX_RANK: usize = 32;
const H5E_DEFAULT: hid_t = 0;

type CanApply = unsafe extern "C" fn(hid_t, hid_t, hid_t) -> htri_t;
type SetLocal = unsafe extern "C" fn(hid_t, hid_t, hid_t) -> herr_t;
type FilterFn = unsafe extern "C" fn(c_uint, usize, *const c_uint, usize, *mut usize, *mut *mut c_void) -> usize;

#[repr(C)]
pub struct H5Z_class2_t {
    version: c_int,
    id: H5Z_filter_t,
    encoder_present: c_uint,
    decoder_present: c_uint,
    name: *const c_char,
    can_apply: Option<CanApply>,
    set_local: Option<SetLocal>,
    filter: Option<FilterFn>,
}

// SAFETY: immutable after construction; HDF5 only reads it.
unsafe impl Sync for H5Z_class2_t {}

unsafe extern "C" {
    fn H5open() -> herr_t;
    fn H5Pget_nfilters(plist: hid_t) -> c_int;
    fn H5Pget_filter2(
        plist: hid_t,
        idx: c_uint,
        flags: *mut c_uint,
        cd_nelmts: *mut usize,
        cd_values: *mut c_uint,
        namelen: usize,
        name: *mut c_char,
        filter_config: *mut c_uint,
    ) -> H5Z_filter_t;
    fn H5Pget_filter_by_id2(
        plist: hid_t,
        id: H5Z_filter_t,
        flags: *mut c_uint,
        cd_nelmts: *mut usize,
        cd_values: *mut c_uint,
        namelen: usize,
        name: *mut c_char,
        filter_config: *mut c_uint,
    ) -> herr_t;
    fn H5Pmodify_filter(plist: hid_t, id: H5Z_filter_t, flags: c_uint, n: usize, cd: *const c_uint) -> herr_t;
    fn H5Pset_filter(plist: hid_t, id: H5Z_filter_t, flags: c_uint, n: usize, cd: *const c_uint) -> herr_t;
    fn H5Sget_simple_extent_dims(space: hid_t, dims: *mut hsize_t, maxdims: *mut hsize_t) -> c_int;
    fn H5Tequal(a: hid_t, b: hid_t) -> htri_t;
    fn H5Epush2(
        stack: hid_t,
        file: *const c_char,
        func: *const c_char,
        line: c_uint,
        cls: hid_t,
        maj: hid_t,
        min: hid_t,
        fmt: *const c_char,
        ...
    ) -> herr_t;
    static H5T_NATIVE_FLOAT_g: hid_t;
    static H5T_NATIVE_DOUBLE_g: hid_t;
    static H5T_NATIVE_INT8_g: hid_t;
    static H5T_NATIVE_UINT8_g: hid_t;
    static H5T_NATIVE_INT16_g: hid_t;
    static H5T_NATIVE_UINT16_g: hid_t;
    static H5T_NATIVE_INT32_g: hid_t;
    static H5T_NATIVE_UINT32_g: hid_t;
    static H5T_NATIVE_INT64_g: hid_t;
    static H5T_NATIVE_UINT64_g: hid_t;
    static H5E_ERR_CLS_g: hid_t;
    static H5E_PLINE_g: hid_t;
    static H5E_ARGS_g: hid_t;
    static H5E_BADVALUE_g: hid_t;
    static H5E_BADTYPE_g: hid_t;
    static H5E_CALLBACK_g: hid_t;
}

static NAME: &CStr =
    c"H5Z-SZ3 (sz3-rust; data format 3.4.0); see https://github.com/szcompressor/SZ3/tree/master/tools/H5Z-SZ3";

pub static H5Z_SZ3: H5Z_class2_t = H5Z_class2_t {
    version: H5Z_CLASS_T_VERS,
    id: H5Z_FILTER_SZ3,
    encoder_present: 1,
    decoder_present: 1,
    name: NAME.as_ptr(),
    can_apply: None,
    set_local: Some(set_local),
    filter: Some(filter),
};

#[unsafe(no_mangle)]
pub extern "C" fn H5PLget_plugin_type() -> c_int {
    H5PL_TYPE_FILTER
}

#[unsafe(no_mangle)]
pub extern "C" fn H5PLget_plugin_info() -> *const c_void {
    &H5Z_SZ3 as *const H5Z_class2_t as *const c_void
}

/// Major and minor error classes of a failure.
#[derive(Clone, Copy, Debug)]
enum Kind {
    PlineValue,
    PlineType,
    PlineCallback,
    ArgsType,
}

/// Pushes `msg` on the HDF5 error stack.
fn push_error(func: &CStr, kind: Kind, msg: &str) {
    let text = std::ffi::CString::new(msg.replace('\0', " ")).unwrap_or_default();
    // SAFETY: HDF5 is initialized (H5open), the class ids are its globals, and "%s" takes one C string.
    unsafe {
        H5open();
        let (maj, min) = match kind {
            Kind::PlineValue => (H5E_PLINE_g, H5E_BADVALUE_g),
            Kind::PlineType => (H5E_PLINE_g, H5E_BADTYPE_g),
            Kind::PlineCallback => (H5E_PLINE_g, H5E_CALLBACK_g),
            Kind::ArgsType => (H5E_ARGS_g, H5E_BADTYPE_g),
        };
        H5Epush2(
            H5E_DEFAULT,
            c"H5Z_SZ3.rs".as_ptr(),
            func.as_ptr(),
            line!(),
            H5E_ERR_CLS_g,
            maj,
            min,
            c"%s".as_ptr(),
            text.as_ptr(),
        );
    }
}

type FResult<T> = Result<T, (Kind, String)>;

fn fail<T>(kind: Kind, msg: impl Into<String>) -> FResult<T> {
    Err((kind, msg.into()))
}

/// The `Config` in cd_values (little-endian words).
fn conf_from_cd(cd: &[c_uint]) -> FResult<Config> {
    let mut bytes = Vec::with_capacity(cd.len() * 4);
    for &w in cd {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    // SZ3 3.2.0 to 3.3.0 stored the magic number and data version first.
    if cd.len() > 1 && cd[0] == SZ3_MAGIC_NUMBER {
        return fail(
            Kind::PlineValue,
            format!(
                "SZ3 HDF5 filter: data is in SZ3 data format v{}, this build reads v{SZ3_DATA_VER}",
                version_str(cd[1])
            ),
        );
    }
    let mut conf = Config::default();
    let load = |b: &[u8], conf: &mut Config| {
        conf.load(&mut Reader::new(b))
            .map_err(|e| (Kind::PlineValue, e.to_string()))
    };
    // 3.3.2 stored no version, so its first byte is the Config's length, never 0.
    if bytes.first().copied().unwrap_or(0) != 0 {
        load(&bytes, &mut conf)?;
        return Ok(conf);
    }
    if cd[0] < version_int(SZ3_DATA_VER_OLDEST) || cd[0] > version_int(SZ3_DATA_VER) {
        return fail(
            Kind::PlineValue,
            format!(
                "SZ3 HDF5 filter: data is in SZ3 data format v{}, this build reads v{SZ3_DATA_VER_OLDEST} to v{SZ3_DATA_VER}",
                version_str(cd[0])
            ),
        );
    }
    load(&bytes[4..], &mut conf)?;
    Ok(conf)
}

/// Whether the list carries the SZ3 filter.
fn filter_on_plist(plist: hid_t) -> bool {
    // SAFETY: plist is a property list id from HDF5; the out-pointers are valid locals.
    unsafe {
        let n = H5Pget_nfilters(plist);
        for i in 0..n.max(0) {
            let mut flags = 0;
            let mut nelmts = 0usize;
            let mut cd = [0 as c_uint; 1];
            let id = H5Pget_filter2(
                plist,
                i as c_uint,
                &mut flags,
                &mut nelmts,
                cd.as_mut_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if id == H5Z_FILTER_SZ3 {
                return true;
            }
        }
    }
    false
}

/// Stores `conf` in the list's SZ3 filter, adding the filter when absent.
fn set_conf_to_h5(plist: hid_t, conf: &Config) -> FResult<()> {
    let mut bytes = Vec::new();
    conf.save(&mut bytes).map_err(|e| (Kind::PlineValue, e.to_string()))?;
    let mut cd = vec![version_int(SZ3_DATA_VER)];
    for c in bytes.chunks(4) {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        cd.push(c_uint::from_le_bytes(w));
    }
    // SAFETY: plist is a property list id from HDF5; cd outlives the calls.
    let ok = unsafe {
        if filter_on_plist(plist) {
            // Keep the flags the filter was set with: H5Z_FLAG_OPTIONAL lets HDF5 store a chunk raw.
            let mut flags = H5Z_FLAG_MANDATORY;
            let mut none = 0usize;
            H5Pget_filter_by_id2(
                plist,
                H5Z_FILTER_SZ3,
                &mut flags,
                &mut none,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ) >= 0
                && H5Pmodify_filter(plist, H5Z_FILTER_SZ3, flags, cd.len(), cd.as_ptr()) >= 0
        } else {
            H5Pset_filter(plist, H5Z_FILTER_SZ3, H5Z_FLAG_MANDATORY, cd.len(), cd.as_ptr()) >= 0
        }
    };
    if ok {
        Ok(())
    } else {
        fail(Kind::PlineValue, "failed to modify cd_values")
    }
}

/// The list's SZ3 `Config`; `None` when it does not carry the filter, `Some(default)` when the
/// filter has no cd_values.
fn conf_from_h5(plist: hid_t) -> FResult<Option<Config>> {
    if !filter_on_plist(plist) {
        return Ok(None);
    }
    let mut n = 0usize;
    let mut probe = [0 as c_uint; 1];
    let mut flags = H5Z_FLAG_MANDATORY;
    // SAFETY: plist is a property list id; the buffers hold the counts passed.
    unsafe {
        if H5Pget_filter_by_id2(
            plist,
            H5Z_FILTER_SZ3,
            &mut flags,
            &mut n,
            probe.as_mut_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) < 0
        {
            return fail(Kind::PlineValue, "cannot read the SZ3 filter's cd_values");
        }
        if n == 0 {
            return Ok(Some(Config::default()));
        }
        let mut cd = vec![0 as c_uint; n];
        if H5Pget_filter_by_id2(
            plist,
            H5Z_FILTER_SZ3,
            &mut flags,
            &mut n,
            cd.as_mut_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) < 0
        {
            return fail(Kind::PlineValue, "cannot read the SZ3 filter's cd_values");
        }
        cd.truncate(n);
        conf_from_cd(&cd).map(Some)
    }
}

/// Sets the SZ3 filter on a dataset creation property list (`H5Pset_sz3`).
///
/// # Safety
/// `plist` must be a dataset creation property list id.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn H5Pset_sz3(
    plist: hid_t,
    cmpr_algo: c_int,
    error_bound_mode: c_int,
    abs_error_bound: f64,
    rel_error_bound: f64,
    psnr_error_bound: f64,
    l2norm_error_bound: f64,
) -> herr_t {
    let r = (|| -> FResult<()> {
        if !(0..=algo::BIOMDXTC as c_int).contains(&cmpr_algo) {
            return fail(Kind::PlineValue, format!("unknown SZ3 algorithm {cmpr_algo}"));
        }
        if cmpr_algo == algo::BIOMDXTC as c_int {
            return fail(
                Kind::PlineValue,
                "SZ3 ALGO_BIOMDXTC was removed in 3.4.0; use H5Z_SZ3_ALGO_BIOMD",
            );
        }
        if !(0..=5).contains(&error_bound_mode) {
            return fail(
                Kind::PlineValue,
                format!("unknown SZ3 error-bound mode {error_bound_mode}"),
            );
        }
        for b in [abs_error_bound, rel_error_bound, psnr_error_bound, l2norm_error_bound] {
            if !b.is_finite() || b < 0.0 {
                return fail(Kind::PlineValue, "SZ3 error bounds must be finite and not negative");
            }
        }
        let conf = Config {
            cmpr_algo: cmpr_algo as u8,
            error_bound_mode: error_bound_mode as u8,
            abs_error_bound,
            rel_error_bound,
            psnr_error_bound,
            l2norm_error_bound,
            ..Config::default()
        };
        set_conf_to_h5(plist, &conf)
    })();
    match r {
        Ok(()) => 1,
        Err((k, m)) => {
            push_error(c"H5Pset_sz3", k, &m);
            -1
        }
    }
}

fn set_local_impl(dcpl: hid_t, type_id: hid_t, space: hid_t) -> FResult<()> {
    let mut conf = conf_from_h5(dcpl)?.unwrap_or_default();
    let mut dims = [0 as hsize_t; H5S_MAX_RANK];
    // SAFETY: space is a dataspace id; dims holds H5S_MAX_RANK entries.
    let nd = unsafe { H5Sget_simple_extent_dims(space, dims.as_mut_ptr(), std::ptr::null_mut()) };
    if nd < 0 {
        return fail(Kind::ArgsType, "not a data space");
    }
    // The chunk's bytes are host values, so the datatype has to equal a host type.
    // SAFETY: H5open initializes the H5T_NATIVE_* globals read here.
    let data_type = unsafe {
        H5open();
        let host: [(hid_t, u8); 10] = [
            (H5T_NATIVE_FLOAT_g, 0),
            (H5T_NATIVE_DOUBLE_g, 1),
            (H5T_NATIVE_INT8_g, 3),
            (H5T_NATIVE_UINT8_g, 2),
            (H5T_NATIVE_INT16_g, 5),
            (H5T_NATIVE_UINT16_g, 4),
            (H5T_NATIVE_INT32_g, 7),
            (H5T_NATIVE_UINT32_g, 6),
            (H5T_NATIVE_INT64_g, 9),
            (H5T_NATIVE_UINT64_g, 8),
        ];
        host.iter().find(|(t, _)| H5Tequal(type_id, *t) > 0).map(|&(_, c)| c)
    };
    let Some(data_type) = data_type else {
        return fail(
            Kind::PlineType,
            "datatype must be the host's float, double, or 1, 2, 4 or 8-byte integer",
        );
    };
    conf.data_type = data_type;
    let d: Vec<usize> = dims[..nd as usize].iter().map(|&v| v as usize).collect();
    conf.set_dims(&d);
    if conf.n > 4 {
        return fail(
            Kind::PlineValue,
            format!("SZ3 compresses at most 4 dimensions longer than 1, not {}", conf.n),
        );
    }
    if conf.cmpr_algo == algo::BIOMD && (conf.n > 3 || conf.dims[conf.n as usize - 1] != 3) {
        let s: Vec<String> = d.iter().map(|v| v.to_string()).collect();
        return fail(
            Kind::PlineValue,
            format!(
                "SZ3 ALGO_BIOMD needs chunks whose last dimension is 3, (frames, atoms, 3); these are ({})",
                s.join(", ")
            ),
        );
    }
    conf.sz3_magic_number = SZ3_MAGIC_NUMBER;
    conf.sz3_data_ver = version_int(SZ3_DATA_VER);
    set_conf_to_h5(dcpl, &conf)
}

unsafe extern "C" fn set_local(dcpl: hid_t, type_id: hid_t, space: hid_t) -> herr_t {
    match catch_unwind(AssertUnwindSafe(|| set_local_impl(dcpl, type_id, space))) {
        Ok(Ok(())) => 1,
        Ok(Err((k, m))) => {
            push_error(c"H5Z_sz3_set_local", k, &m);
            -1
        }
        Err(_) => {
            push_error(c"H5Z_sz3_set_local", Kind::PlineCallback, "internal error");
            -1
        }
    }
}

/// A malloc'ed copy of `v`, which HDF5 frees.
fn malloc_copy(v: &[u8]) -> FResult<*mut c_void> {
    // SAFETY: a fresh allocation of v.len() bytes, filled before it is returned.
    unsafe {
        let p = libc::malloc(v.len().max(1));
        if p.is_null() {
            return fail(Kind::PlineCallback, "out of memory");
        }
        std::ptr::copy_nonoverlapping(v.as_ptr(), p as *mut u8, v.len());
        Ok(p)
    }
}

fn process<T: SzData>(conf: &mut Config, input: &[u8], decompress: bool) -> FResult<Vec<u8>> {
    let err = |e: sz3::Error| (Kind::PlineCallback, e.to_string());
    if decompress {
        let (_, v) = sz3::decompress::<T>(input).map_err(err)?;
        let mut out = Vec::with_capacity(v.len() * T::SIZE);
        for x in v {
            x.write_le(&mut out);
        }
        Ok(out)
    } else {
        if input.len() != conf.num * T::SIZE {
            return fail(
                Kind::PlineCallback,
                "SZ3 HDF5 filter: a filter before SZ3 changed the chunk's size",
            );
        }
        let values: Vec<T> = input.chunks_exact(T::SIZE).map(T::read_le).collect();
        // As SZ3's filter: headroom above the bound for output that reaches the raw size.
        let cap = sz3::compress_size_bound::<T>(conf).max(T::SIZE * conf.num * 2);
        sz3::compress_with_capacity(conf, &values, cap).map_err(err)
    }
}

fn filter_impl(flags: c_uint, cd: &[c_uint], input: &[u8]) -> FResult<Option<Vec<u8>>> {
    if cd.is_empty() {
        // special data such as strings, which are not values
        return Ok(None);
    }
    let decompress = flags & H5Z_FLAG_REVERSE != 0;
    let mut conf = if decompress {
        // The chunk carries its own Config.
        let mut r = Reader::new(input);
        let (magic, ver, size) = if input.len() >= 16 {
            (r.u32().unwrap(), r.u32().unwrap(), r.u64().unwrap())
        } else {
            (0, 0, 0)
        };
        if magic != SZ3_MAGIC_NUMBER {
            // 3.3.2 stored chunks of fewer than 20 elements raw, in under 20 * 8 bytes.
            if input.len() < 20 * 8 {
                return Ok(None);
            }
            return fail(
                Kind::PlineCallback,
                "SZ3 HDF5 filter: chunk is not in a data format this SZ3 reads, or is damaged",
            );
        }
        if ver < version_int(SZ3_DATA_VER_OLDEST) || ver > version_int(SZ3_DATA_VER) {
            let hint = if ver > version_int(SZ3_DATA_VER) {
                "; upgrade SZ3 to read it".to_string()
            } else {
                format!("; use SZ3 v{} to read it", version_str(ver))
            };
            return fail(
                Kind::PlineCallback,
                format!(
                    "SZ3 HDF5 filter: data is in SZ3 data format v{}, this build reads v{SZ3_DATA_VER_OLDEST} to v{SZ3_DATA_VER}{hint}",
                    version_str(ver)
                ),
            );
        }
        if size > (input.len() - 16) as u64 {
            return fail(Kind::PlineCallback, "SZ3 HDF5 filter: chunk is truncated");
        }
        r.skip(size as usize)
            .map_err(|e| (Kind::PlineCallback, e.to_string()))?;
        let mut c = Config::default();
        c.load(&mut r).map_err(|e| (Kind::PlineCallback, e.to_string()))?;
        c
    } else {
        conf_from_cd(cd)?
    };
    let out = match conf.data_type {
        0 => process::<f32>(&mut conf, input, decompress),
        1 => process::<f64>(&mut conf, input, decompress),
        3 => process::<i8>(&mut conf, input, decompress),
        2 => process::<u8>(&mut conf, input, decompress),
        5 => process::<i16>(&mut conf, input, decompress),
        4 => process::<u16>(&mut conf, input, decompress),
        7 => process::<i32>(&mut conf, input, decompress),
        6 => process::<u32>(&mut conf, input, decompress),
        9 => process::<i64>(&mut conf, input, decompress),
        8 => process::<u64>(&mut conf, input, decompress),
        _ => fail(Kind::PlineCallback, "SZ3 HDF5 filter: unknown datatype in cd_values"),
    }?;
    Ok(Some(out))
}

/// The filter callback: returns the valid bytes in `*buf`, or 0 on failure.
unsafe extern "C" fn filter(
    flags: c_uint,
    cd_nelmts: usize,
    cd_values: *const c_uint,
    nbytes: usize,
    buf_size: *mut usize,
    buf: *mut *mut c_void,
) -> usize {
    let r = catch_unwind(AssertUnwindSafe(|| -> FResult<usize> {
        // SAFETY: HDF5 passes cd_nelmts values, and nbytes valid bytes at *buf.
        let (cd, input) = unsafe {
            let cd = if cd_nelmts == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(cd_values, cd_nelmts)
            };
            (cd, std::slice::from_raw_parts(*buf as *const u8, nbytes))
        };
        match filter_impl(flags, cd, input)? {
            None => Ok(nbytes),
            Some(out) => {
                let p = malloc_copy(&out)?;
                // SAFETY: *buf was malloc'ed by HDF5, which takes the new buffer and its size.
                unsafe {
                    libc::free(*buf);
                    *buf = p;
                    *buf_size = out.len();
                }
                Ok(out.len())
            }
        }
    }));
    match r {
        Ok(Ok(n)) => n,
        Ok(Err((k, m))) => {
            // HDF5 1.14.5+ may pause the error stack around filters; print too.
            eprintln!("{m}");
            push_error(c"H5Z_filter_sz3", k, &m);
            0
        }
        Err(_) => {
            push_error(c"H5Z_filter_sz3", Kind::PlineCallback, "internal error");
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(b: &[u8]) -> Vec<c_uint> {
        b.chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                c_uint::from_le_bytes(w)
            })
            .collect()
    }

    #[test]
    fn cd_values_layouts() {
        let mut conf = Config::new(&[20, 30]);
        conf.abs_error_bound = 0.25;
        conf.data_type = 1;
        let mut b = Vec::new();
        conf.save(&mut b).unwrap();
        // 3.3.2: the Config alone
        let c = conf_from_cd(&words(&b)).unwrap();
        assert_eq!(
            (c.dims.clone(), c.abs_error_bound, c.data_type),
            (vec![20, 30], 0.25, 1)
        );
        // 3.4.0: the data version first
        let mut cd = vec![version_int(SZ3_DATA_VER)];
        cd.extend(words(&b));
        assert_eq!(conf_from_cd(&cd).unwrap().dims, vec![20, 30]);
        // other versions are refused
        cd[0] = version_int("3.9.0");
        assert!(conf_from_cd(&cd).is_err());
        assert!(conf_from_cd(&[SZ3_MAGIC_NUMBER, version_int("3.2.0")]).is_err());
    }
}
