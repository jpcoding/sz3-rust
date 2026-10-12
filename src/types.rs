//! Element types SZ3 compresses.

use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::bytes::Pod;

/// Floating-point element types the prediction pipeline works on (`f32`, `f64`).
///
/// Arithmetic mirrors C++: an operation between two `T`s happens in `T`; an operation with a
/// `double` literal or variable promotes to `f64` (`to_f64`) and rounds back with `from_f64`.
pub trait SzFloat:
    Pod
    + PartialOrd
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    /// SZ3's `SZ_FLOAT`/`SZ_DOUBLE`.
    const DATA_TYPE: u8;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    /// `static_cast<T>(size_t)`.
    fn from_usize(v: usize) -> Self;
    /// `T(int)` for the small integer constants in the predictors.
    fn c(v: i32) -> Self;
    fn abs(self) -> Self;
    fn is_finite(self) -> bool;
}

impl SzFloat for f32 {
    const ZERO: Self = 0.0;
    const DATA_TYPE: u8 = 0;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as f32
    }
    #[inline(always)]
    fn c(v: i32) -> Self {
        v as f32
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl SzFloat for f64 {
    const ZERO: Self = 0.0;
    const DATA_TYPE: u8 = 1;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    #[inline(always)]
    fn c(v: i32) -> Self {
        v as f64
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

/// How an element type enters the pipeline.
pub enum Repr<'a, T> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    /// Integers are compressed as the float type that holds them.
    Int(&'a [T]),
}

/// Any element type the public API accepts: floats, and integers which SZ3 compresses as
/// `f32` (1- and 2-byte) or `f64` (4- and 8-byte) with an integer-rounded error bound.
pub trait SzData: Pod + Debug {
    const IS_FLOAT: bool;
    /// Size of the float type the data is compressed as.
    const FLOAT_SIZE: usize;
    /// SZ3's data type code (`SZ_FLOAT`, `SZ_INT32`, ...).
    const DATA_TYPE: u8;
    fn to_f64(self) -> f64;
    /// Round and clamp a decompressed value into the type, as SZ3 does.
    fn from_f64_rounded(v: f64) -> Self;
    fn as_f32_slice(s: &[Self]) -> Option<&[f32]>;
    fn as_f64_slice(s: &[Self]) -> Option<&[f64]>;
    fn from_f32_vec(v: Vec<f32>) -> Option<Vec<Self>>;
    fn from_f64_vec(v: Vec<f64>) -> Option<Vec<Self>>;
}

impl SzData for f32 {
    const IS_FLOAT: bool = true;
    const FLOAT_SIZE: usize = 4;
    const DATA_TYPE: u8 = 0;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64_rounded(v: f64) -> Self {
        v as f32
    }
    fn as_f32_slice(s: &[Self]) -> Option<&[f32]> {
        Some(s)
    }
    fn as_f64_slice(_: &[Self]) -> Option<&[f64]> {
        None
    }
    fn from_f32_vec(v: Vec<f32>) -> Option<Vec<Self>> {
        Some(v)
    }
    fn from_f64_vec(_: Vec<f64>) -> Option<Vec<Self>> {
        None
    }
}

impl SzData for f64 {
    const IS_FLOAT: bool = true;
    const FLOAT_SIZE: usize = 8;
    const DATA_TYPE: u8 = 1;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64_rounded(v: f64) -> Self {
        v
    }
    fn as_f32_slice(_: &[Self]) -> Option<&[f32]> {
        None
    }
    fn as_f64_slice(s: &[Self]) -> Option<&[f64]> {
        Some(s)
    }
    fn from_f32_vec(_: Vec<f32>) -> Option<Vec<Self>> {
        None
    }
    fn from_f64_vec(v: Vec<f64>) -> Option<Vec<Self>> {
        Some(v)
    }
}

macro_rules! impl_int {
    ($t:ty, $fs:expr, $code:expr) => {
        impl SzData for $t {
            const IS_FLOAT: bool = false;
            const FLOAT_SIZE: usize = $fs;
            const DATA_TYPE: u8 = $code;
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64_rounded(v: f64) -> Self {
                // Clamped first; a NaN from damaged data becomes the lowest value.
                let v = v.round();
                if !(v > <$t>::MIN as f64) {
                    <$t>::MIN
                } else if v >= <$t>::MAX as f64 {
                    <$t>::MAX
                } else {
                    v as $t
                }
            }
            fn as_f32_slice(_: &[Self]) -> Option<&[f32]> {
                None
            }
            fn as_f64_slice(_: &[Self]) -> Option<&[f64]> {
                None
            }
            fn from_f32_vec(_: Vec<f32>) -> Option<Vec<Self>> {
                None
            }
            fn from_f64_vec(_: Vec<f64>) -> Option<Vec<Self>> {
                None
            }
        }
    };
}
impl_int!(u8, 4, 2);
impl_int!(i8, 4, 3);
impl_int!(u16, 4, 4);
impl_int!(i16, 4, 5);
impl_int!(u32, 8, 6);
impl_int!(i32, 8, 7);
impl_int!(u64, 8, 8);
impl_int!(i64, 8, 9);
