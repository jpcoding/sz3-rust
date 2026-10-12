//! Error-bound conversion and quality metrics.

use crate::config::{Config, eb};
use crate::error::{Error, Result};
use crate::types::SzFloat;

/// max - min, computed in `T` as SZ3 does.
pub fn data_range<T: SzFloat>(data: &[T]) -> T {
    let mut max = data[0];
    let mut min = data[0];
    for &v in &data[1..] {
        if max < v {
            max = v;
        }
        if min > v {
            min = v;
        }
    }
    max - min
}

pub fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

pub fn abs_eb_from_psnr(psnr: f64, threshold: f64, value_range: f64) -> f64 {
    let v1 = psnr + 10.0 * (1.0 - 2.0 / 3.0 * threshold).log10();
    let v2 = v1 / -20.0;
    let v3 = 10f64.powf(v2);
    value_range * v3
}

/// Turns any error bound mode into an absolute bound (`calAbsErrorBound`). `range` > 0 skips
/// computing the value range.
pub fn cal_abs_error_bound<T: SzFloat>(conf: &mut Config, data: &[T], range: Option<T>) -> Result<()> {
    if conf.error_bound_mode == eb::ABS {
        return Ok(());
    }
    let range = || -> f64 {
        match range {
            Some(r) if r > T::ZERO => r.to_f64(),
            _ => data_range(&data[..conf.num]).to_f64(),
        }
    };
    conf.abs_error_bound = match conf.error_bound_mode {
        eb::REL => conf.rel_error_bound * range(),
        eb::PSNR => abs_eb_from_psnr(conf.psnr_error_bound, 0.99, range()),
        eb::L2NORM => (3.0 / conf.num as f64).sqrt() * conf.l2norm_error_bound,
        eb::ABS_AND_REL => conf.abs_error_bound.min(conf.rel_error_bound * range()),
        eb::ABS_OR_REL => conf.abs_error_bound.max(conf.rel_error_bound * range()),
        _ => return Err(Error::InvalidArgument("Error bound mode not supported".into())),
    };
    conf.error_bound_mode = eb::ABS;
    Ok(())
}

/// Distortion of `dec` against `ori`.
#[derive(Debug, Clone, Copy)]
pub struct Quality {
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub max_pw_rel_err: f64,
    pub psnr: f64,
    pub nrmse: f64,
    pub norm_err: f64,
    pub norm_err_norm: f64,
    pub ac_eff: f64,
}

pub fn verify(ori: &[f64], dec: &[f64]) -> Quality {
    let n = ori.len() as f64;
    let (mut max, mut min) = (ori[0], ori[0]);
    let mut max_diff = (dec[0] - ori[0]).abs();
    let mut maxpw = 0f64;
    let (mut sum1, mut sum2, mut l2sum) = (0f64, 0f64, 0f64);
    for i in 0..ori.len() {
        sum1 += ori[i];
        sum2 += dec[i];
        l2sum += dec[i] * dec[i];
    }
    let (mean1, mean2) = (sum1 / n, sum2 / n);
    let (mut sum3, mut sum4, mut sum, mut prod) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..ori.len() {
        max = max.max(ori[i]);
        min = min.min(ori[i]);
        let err = (dec[i] - ori[i]).abs();
        if ori[i] != 0.0 {
            maxpw = maxpw.max(err / ori[i].abs());
        }
        max_diff = max_diff.max(err);
        prod += (ori[i] - mean1) * (dec[i] - mean2);
        sum3 += (ori[i] - mean1) * (ori[i] - mean1);
        sum4 += (dec[i] - mean2) * (dec[i] - mean2);
        sum += err * err;
    }
    let ac_eff = (prod / n) / (sum3 / n).sqrt() / (sum4 / n).sqrt();
    let mse = sum / n;
    let range = max - min;
    Quality {
        min,
        max,
        range,
        max_abs_err: max_diff,
        max_rel_err: max_diff / range,
        max_pw_rel_err: maxpw,
        psnr: 20.0 * range.log10() - 10.0 * mse.log10(),
        nrmse: mse.sqrt() / range,
        norm_err: sum.sqrt(),
        norm_err_norm: sum.sqrt() / l2sum.sqrt(),
        ac_eff,
    }
}
