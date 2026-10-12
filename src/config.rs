//! Compression configuration, its binary serialization (stored after every stream) and the INI
//! format the `sz3` tool reads.

use crate::bytes::{ByteWriter, Reader, bytes2vector, vector_bit_width, vector2bytes};
use crate::error::{Error, Result, corrupt};

pub const SZ3_MAGIC_NUMBER: u32 = 0xF342_F310;
pub const SZ3_VERSION: &str = "3.4.0";
/// Version of the compressed format this port writes.
pub const SZ3_DATA_VER: &str = "3.4.0";
/// The oldest data version this port reads.
pub const SZ3_DATA_VER_OLDEST: &str = "3.3.2";

pub fn version_int(v: &str) -> u32 {
    let mut it = v.split('.').map(|p| p.parse::<u32>().unwrap_or(0));
    let (a, b, c) = (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0));
    (a << 24) | (b << 16) | (c << 8)
}

pub fn version_str(v: u32) -> String {
    format!("{}.{}.{}", (v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff)
}

/// Error bound modes (`SZ3::EB`).
pub mod eb {
    pub const ABS: u8 = 0;
    pub const REL: u8 = 1;
    pub const PSNR: u8 = 2;
    pub const L2NORM: u8 = 3;
    pub const ABS_AND_REL: u8 = 4;
    pub const ABS_OR_REL: u8 = 5;
}

/// Compression algorithms (`SZ3::ALGO`).
pub mod algo {
    pub const LORENZO_REG: u8 = 0;
    pub const INTERP_LORENZO: u8 = 1;
    pub const INTERP: u8 = 2;
    pub const NOPRED: u8 = 3;
    pub const LOSSLESS: u8 = 4;
    pub const BIOMD: u8 = 5;
    pub const BIOMDXTC: u8 = 6;
}

/// Interpolation algorithms (`SZ3::INTERP_ALGO`).
pub mod interp {
    pub const LINEAR: u8 = 0;
    pub const CUBIC: u8 = 1;
}

pub const ALGO_MAP: &[(&str, u8)] = &[
    ("ALGO_BIOMD", algo::BIOMD),
    ("ALGO_BIOMDXTC", algo::BIOMDXTC),
    ("ALGO_INTERP", algo::INTERP),
    ("ALGO_INTERP_LORENZO", algo::INTERP_LORENZO),
    ("ALGO_LORENZO_REG", algo::LORENZO_REG),
    ("ALGO_LOSSLESS", algo::LOSSLESS),
    ("ALGO_NOPRED", algo::NOPRED),
];

pub const EB_MAP: &[(&str, u8)] = &[
    ("ABS", eb::ABS),
    ("ABS_AND_REL", eb::ABS_AND_REL),
    ("ABS_OR_REL", eb::ABS_OR_REL),
    ("NORM", eb::L2NORM),
    ("PSNR", eb::PSNR),
    ("REL", eb::REL),
];

pub const INTERP_ALGO_MAP: &[(&str, u8)] = &[
    ("INTERP_ALGO_CUBIC", interp::CUBIC),
    ("INTERP_ALGO_LINEAR", interp::LINEAR),
];

/// Case-insensitive lookup; leaves `out` unchanged when nothing matches, as SZ3 does.
pub fn match_enum(input: &str, table: &[(&str, u8)], out: &mut u8) {
    let lc = input.to_lowercase();
    for (k, v) in table {
        if k.to_lowercase() == lc {
            *out = *v;
        }
    }
}

pub fn enum_to_string(value: u8, table: &[(&'static str, u8)]) -> &'static str {
    table.iter().find(|e| e.1 == value).map(|e| e.0).unwrap_or("")
}

/// SZ3's `Config`: data shape, algorithm and error bound, and module settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub sz3_magic_number: u32,
    pub sz3_data_ver: u32,

    pub n: u8,
    pub dims: Vec<usize>,
    pub num: usize,

    pub cmpr_algo: u8,
    pub error_bound_mode: u8,
    pub abs_error_bound: f64,
    pub rel_error_bound: f64,
    pub psnr_error_bound: f64,
    pub l2norm_error_bound: f64,
    pub openmp: bool,

    pub quantbin_cnt: i32,
    pub block_size: i32,
    pub pred_dim: u8,
    pub data_type: u8,
    pub lorenzo: bool,
    pub lorenzo2: bool,
    pub regression: bool,
    pub regression2: bool,
    pub interp_algo: u8,
    pub interp_direction: u8,
    pub interp_anchor_stride: i32,
    pub interp_alpha: f64,
    pub interp_beta: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sz3_magic_number: SZ3_MAGIC_NUMBER,
            sz3_data_ver: version_int(SZ3_DATA_VER),
            n: 0,
            dims: Vec::new(),
            num: 0,
            cmpr_algo: algo::INTERP_LORENZO,
            error_bound_mode: eb::ABS,
            abs_error_bound: 1e-3,
            rel_error_bound: 0.0,
            psnr_error_bound: 0.0,
            l2norm_error_bound: 0.0,
            openmp: false,
            quantbin_cnt: 65536,
            block_size: 0,
            pred_dim: 0,
            data_type: 0,
            lorenzo: true,
            lorenzo2: false,
            regression: true,
            regression2: false,
            interp_algo: interp::CUBIC,
            interp_direction: 0,
            interp_anchor_stride: -1,
            interp_alpha: 1.25,
            interp_beta: 2.0,
        }
    }
}

impl Config {
    /// A configuration for data of the given shape, slowest dimension first.
    pub fn new(dims: &[usize]) -> Self {
        let mut c = Config::default();
        c.set_dims(dims);
        c
    }

    /// Sets the shape. Dimensions of size 1 are dropped; also resets `block_size` and `pred_dim`.
    pub fn set_dims(&mut self, dims: &[usize]) -> usize {
        self.dims = dims.iter().copied().filter(|&d| d > 1).collect();
        if self.dims.is_empty() {
            self.dims = vec![1];
        }
        self.n = self.dims.len() as u8;
        self.num = self.dims.iter().product();
        self.pred_dim = self.n;
        self.block_size = match self.n {
            1 => 128,
            2 => 16,
            _ => 6,
        };
        self.num
    }

    /// Appends the binary form; returns its size.
    pub fn save(&self, out: &mut Vec<u8>) -> Result<usize> {
        let start = out.len();
        out.put_u8(0); // size, filled in below
        out.put_u8(self.n);
        let bw = vector_bit_width(&self.dims);
        out.put_u8(bw);
        vector2bytes(&self.dims, bw, out);
        out.put_u64(self.num as u64);
        out.put_u8(self.cmpr_algo);
        out.put_u8(self.error_bound_mode);
        match self.error_bound_mode {
            eb::ABS => out.put_f64(self.abs_error_bound),
            eb::REL => out.put_f64(self.rel_error_bound),
            eb::PSNR => out.put_f64(self.psnr_error_bound),
            eb::L2NORM => out.put_f64(self.l2norm_error_bound),
            eb::ABS_OR_REL | eb::ABS_AND_REL => {
                out.put_f64(self.abs_error_bound);
                out.put_f64(self.rel_error_bound);
            }
            _ => return Err(Error::InvalidArgument("Config::save: unknown error bound mode".into())),
        }
        let boolvals = (self.lorenzo as u8) << 7
            | (self.lorenzo2 as u8) << 6
            | (self.regression as u8) << 5
            | (self.regression2 as u8) << 4
            | (self.openmp as u8) << 3;
        out.put_u8(boolvals);
        out.put_u8(self.data_type);
        out.put_i32(self.quantbin_cnt);
        out.put_i32(self.block_size);
        out.put_u8(self.pred_dim);
        let written = out.len() - start;
        if written > 255 {
            return Err(Error::InvalidArgument(format!(
                "Config::save: a serialized config is at most 255 bytes, this one is {written}"
            )));
        }
        out[start] = written as u8;
        Ok(written)
    }

    /// Size of the binary form.
    pub fn size_est(&self) -> usize {
        let mut v = Vec::with_capacity(64);
        self.save(&mut v).unwrap_or(255)
    }

    /// Reads the binary form, keeping fields absent from older streams at their current values.
    pub fn load(&mut self, r: &mut Reader) -> Result<()> {
        let start = r.position();
        let avail_end = start + r.remaining();
        let conf_size = r.u8()? as usize;
        let c1 = (start + conf_size).min(avail_end);

        self.n = r.u8()?;
        if self.n > 4 {
            return corrupt("Config::load: invalid number of dimensions");
        }
        let bw = r.u8()?;
        if bw > 64 {
            return corrupt("Config::load: invalid dimension bit width");
        }
        if bw as usize > usize::BITS as usize {
            return corrupt("Config::load: dimension wider than usize");
        }
        self.dims = bytes2vector(r, bw, self.n as usize)?;
        let num64 = r.u64()?;
        self.num = usize::try_from(num64).map_err(|_| Error::Corrupt("element count overflows".into()))?;
        let mut prod: usize = if self.dims.is_empty() { 0 } else { 1 };
        let mut ok = true;
        for &d in &self.dims {
            match prod.checked_mul(d) {
                Some(p) if d != 0 => prod = p,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || prod != self.num {
            return corrupt("Config::load: dimensions inconsistent with the element count");
        }
        self.cmpr_algo = r.u8()?;
        self.error_bound_mode = r.u8()?;
        match self.error_bound_mode {
            eb::ABS => self.abs_error_bound = r.f64()?,
            eb::REL => self.rel_error_bound = r.f64()?,
            eb::PSNR => self.psnr_error_bound = r.f64()?,
            eb::L2NORM => self.l2norm_error_bound = r.f64()?,
            eb::ABS_OR_REL | eb::ABS_AND_REL => {
                self.abs_error_bound = r.f64()?;
                self.rel_error_bound = r.f64()?;
            }
            _ => return Err(Error::Corrupt("Config::load: unknown error bound mode".into())),
        }
        if r.position() < c1 {
            let b = r.u8()?;
            self.lorenzo = (b >> 7) & 1 != 0;
            self.lorenzo2 = (b >> 6) & 1 != 0;
            self.regression = (b >> 5) & 1 != 0;
            self.regression2 = (b >> 4) & 1 != 0;
            self.openmp = (b >> 3) & 1 != 0;
        }
        if r.position() < c1 {
            self.data_type = r.u8()?;
        }
        if r.position() < c1 {
            self.quantbin_cnt = r.i32()?;
        }
        if r.position() < c1 {
            self.block_size = r.i32()?;
        }
        if r.position() < c1 {
            self.pred_dim = r.u8()?;
        }
        if r.position() > c1 {
            return corrupt("Config::load: the config is longer than its declared size");
        }
        // Skip fields a later version appends.
        r.skip(c1 - r.position())?;
        Ok(())
    }

    /// Reads settings from INI text (`[GlobalSettings]`, `[AlgoSettings]`).
    pub fn load_ini(&mut self, content: &str) -> Result<()> {
        let mut section = String::new();
        let parse_bool = |s: &str| matches!(s.to_lowercase().as_str(), "true" | "1" | "yes" | "on");
        let num_err = |k: &str, v: &str| Error::InvalidArgument(format!("bad value for {k}: {v}"));
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                section = rest.split(']').next().unwrap_or("").to_string();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim().to_lowercase(), value.trim());
            let f = |v: &str| parse_leading_f64(v).ok_or_else(|| num_err(&key, v));
            let i = |v: &str| parse_leading_i64(v).map(|x| x as i32).ok_or_else(|| num_err(&key, v));
            match (section.to_lowercase().as_str(), key.as_str()) {
                ("globalsettings", "cmpralgo") => match_enum(value, ALGO_MAP, &mut self.cmpr_algo),
                ("globalsettings", "errorboundmode") => match_enum(value, EB_MAP, &mut self.error_bound_mode),
                ("globalsettings", "abserrorbound") => self.abs_error_bound = f(value)?,
                ("globalsettings", "relerrorbound") => self.rel_error_bound = f(value)?,
                ("globalsettings", "psnrerrorbound") => self.psnr_error_bound = f(value)?,
                ("globalsettings", "l2normerrorbound") => self.l2norm_error_bound = f(value)?,
                ("globalsettings", "openmp") => self.openmp = parse_bool(value),
                ("algosettings", "lorenzo") => self.lorenzo = parse_bool(value),
                ("algosettings", "lorenzo2ndorder") => self.lorenzo2 = parse_bool(value),
                ("algosettings", "regression") => self.regression = parse_bool(value),
                ("algosettings", "regression2ndorder") => self.regression2 = parse_bool(value),
                ("algosettings", "interpolationalgo") => match_enum(value, INTERP_ALGO_MAP, &mut self.interp_algo),
                ("algosettings", "interpolationdirection") => self.interp_direction = i(value)? as u8,
                ("algosettings", "blocksize") => self.block_size = i(value)?,
                ("algosettings", "quantizationbintotal") => self.quantbin_cnt = i(value)?,
                ("algosettings", "interpolationanchorstride") => self.interp_anchor_stride = i(value)?,
                ("algosettings", "interpolationalpha") => self.interp_alpha = f(value)?,
                ("algosettings", "interpolationbeta") => self.interp_beta = f(value)?,
                _ => {}
            }
        }
        Ok(())
    }

    pub fn load_cfg(&mut self, path: &str) -> Result<()> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| Error::InvalidArgument(format!("Failed to open INI file: {path}: {e}")))?;
        self.load_ini(&s)
    }

    pub fn save_ini(&self) -> String {
        let b = |v: bool| if v { "true" } else { "false" };
        let mut s = String::new();
        s += "[GlobalSettings]\n";
        s += &format!("CmprAlgo = {}\n", enum_to_string(self.cmpr_algo, ALGO_MAP));
        s += &format!("ErrorBoundMode = {}\n", enum_to_string(self.error_bound_mode, EB_MAP));
        s += &format!("AbsErrorBound = {}\n", fmt_g(self.abs_error_bound));
        s += &format!("RelErrorBound = {}\n", fmt_g(self.rel_error_bound));
        s += &format!("PSNRErrorBound = {}\n", fmt_g(self.psnr_error_bound));
        s += &format!("L2NormErrorBound = {}\n", fmt_g(self.l2norm_error_bound));
        s += &format!("OpenMP = {}\n", b(self.openmp));
        s += "\n[AlgoSettings]\n";
        s += &format!("Lorenzo = {}\n", b(self.lorenzo));
        s += &format!("Lorenzo2ndOrder = {}\n", b(self.lorenzo2));
        s += &format!("Regression = {}\n", b(self.regression));
        s += &format!("Regression2ndOrder = {}\n", b(self.regression2));
        s += &format!("BlockSize = {}\n", self.block_size);
        s += &format!("QuantizationBinTotal = {}\n", self.quantbin_cnt);
        s += &format!(
            "InterpolationAlgo = {}\n",
            enum_to_string(self.interp_algo, INTERP_ALGO_MAP)
        );
        s += &format!("InterpolationDirection = {}\n", self.interp_direction);
        s += &format!("InterpolationAnchorStride = {}\n", self.interp_anchor_stride);
        s += &format!("InterpolationAlpha = {}\n", fmt_g(self.interp_alpha));
        s += &format!("InterpolationBeta = {}\n", fmt_g(self.interp_beta));
        s
    }

    pub fn print(&self) {
        println!("\nsz3MagicNumber = {}", self.sz3_magic_number);
        println!("sz3DataVer = {}", version_str(self.sz3_data_ver));
        print!("Dimensions =");
        for d in &self.dims {
            print!(" {d}");
        }
        println!();
        print!("{}", self.save_ini());
    }
}

/// `std::stod`-like: parses the longest numeric prefix.
fn parse_leading_f64(s: &str) -> Option<f64> {
    let s = s.trim_start();
    (1..=s.len())
        .rev()
        .filter(|&i| s.is_char_boundary(i))
        .find_map(|i| s[..i].parse::<f64>().ok())
}

fn parse_leading_i64(s: &str) -> Option<i64> {
    let s = s.trim_start();
    (1..=s.len())
        .rev()
        .filter(|&i| s.is_char_boundary(i))
        .find_map(|i| s[..i].parse::<i64>().ok())
}

/// C++ `ostream << double` (`%g` with 6 significant digits).
pub fn fmt_g(v: f64) -> String {
    fmt_g_prec(v, 6)
}

pub fn fmt_g_prec(v: f64, prec: usize) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".into()
        } else if v > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let p = prec.max(1);
    let e = format!("{:.*e}", p - 1, v);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if exp < -4 || exp >= p as i32 {
        format!("{}e{}{:02}", strip(mant), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        strip(&format!("{:.*}", (p as i32 - 1 - exp).max(0) as usize, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_round_trip() {
        let mut c = Config::new(&[100, 200, 300]);
        c.error_bound_mode = eb::ABS_AND_REL;
        c.rel_error_bound = 1e-4;
        let mut buf = Vec::new();
        let n = c.save(&mut buf).unwrap();
        assert_eq!(n, buf.len());
        let mut d = Config::default();
        d.load(&mut Reader::new(&buf)).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn set_dims_drops_ones() {
        let c = Config::new(&[1, 8, 1, 128]);
        assert_eq!(c.dims, vec![8, 128]);
        assert_eq!(c.n, 2);
        assert_eq!(c.block_size, 16);
    }

    #[test]
    fn g_format() {
        assert_eq!(fmt_g(1e-3), "0.001");
        assert_eq!(fmt_g(1.25), "1.25");
        assert_eq!(fmt_g(2.0), "2");
        assert_eq!(fmt_g(1e-7), "1e-07");
        assert_eq!(fmt_g(123456789.0), "1.23457e+08");
    }

    #[test]
    fn versions() {
        assert_eq!(version_int("3.4.0"), 0x0304_0000);
        assert_eq!(version_str(0x0303_0200), "3.3.2");
    }
}
