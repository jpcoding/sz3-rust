//! `ALGO_BIOMD`: molecular-dynamics coordinates `{frames, atoms, 3}` (nm, absolute bound) as
//! streams of integer symbols, one after the other as `[count, symbols]` (SZ3's
//! `SZBioMDDecomposition`). Every coordinate goes on a lattice, `q = round(x / step)`,
//! `|x - q step| <= eb`, and all prediction is integer arithmetic on that lattice, which the
//! decoder repeats exactly.
//!
//! * rigid water (O, H1, H2): H1 on the sphere `|H1 - O| = r`, H2 on the circle that `r` and the
//!   H-H distance leave
//! * other atoms: from their bond partner, one of the previous `MAX_BOND_OFFSET` atoms (intra a
//!   delta, else on the sphere of the previous frame's bond length), else a delta
//! * frame 0 intra; for later frames water O's from the previous frame, and per other atom group
//!   intra or from the previous frame, whichever costs less on a sample of frame 1
//!
//! The layout (which atoms are water, the bond of each other atom) is found on a chunk's first
//! frame and stored with it.

use std::marker::PhantomData;

use super::Decomposition;
use crate::bytes::{BitAppender, BitConsumer, ByteWriter, Reader, unzigzag, zigzag};
use crate::config::Config;
use crate::error::{Error, Result, corrupt};
use crate::types::SzFloat;

/// A bond partner is searched among the previous `MAX_BOND_OFFSET` atoms.
const MAX_BOND_OFFSET: usize = 4;
/// Deltas from here on (lattice units) are an escape symbol and a raw byte.
const ESCAPE: u32 = 2048;
/// `|q| <= 2^28`.
const MAX_LATTICE: f64 = (1u32 << 28) as f64;

const G_WATER_O: usize = 0;
const G_WATER_H: usize = 1;
const G_BONDED: usize = 2;
const G_UNBONDED: usize = 3;
const NUM_GROUPS: usize = 4;

const S_WATER_GAP: usize = 0;
const S_BOND_REF: usize = 1;
const S_WATER_O: usize = 2;
const S_WATER_H1_FACE: usize = 3;
const S_WATER_H1_KEPT: usize = 4;
const S_WATER_H1_RADIAL: usize = 5;
const S_WATER_H2_AXIS: usize = 6;
const S_WATER_H2_RESIDUAL: usize = 7;
const S_BOND_DELTA: usize = 8;
const S_BOND_KEPT: usize = 9;
const S_BOND_RADIAL: usize = 10;
const S_UNBONDED: usize = 11;
pub const NUM_STREAMS: usize = 12;

const K_WATER_O: u8 = 0;
const K_WATER_H: u8 = 1;
const K_OTHER: u8 = 2;

fn fallback(msg: &str) -> Error {
    Error::Fallback(msg.into())
}

#[inline(always)]
fn round_half_away(y: f64) -> i64 {
    (y + 0.5f64.copysign(y)) as i64
}

#[inline(always)]
fn isqrt_round(n: i64) -> i64 {
    if n <= 0 { 0 } else { ((n as f64).sqrt() + 0.5) as i64 }
}

/// `frexp`'s exponent: `v = m 2^e`, `m` in [0.5, 1); 0 for zero, infinity or NaN (as glibc).
fn frexp_exp(v: f64) -> i32 {
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0 {
        return frexp_exp(v * 2f64.powi(64)) - 64;
    }
    e - 1022
}

/// `ldexp`: `x 2^e`, rounded once.
fn ldexp(mut x: f64, mut e: i32) -> f64 {
    let p = |k: i32| f64::from_bits(((k + 1023) as u64) << 52);
    while e > 1000 {
        x *= p(1000);
        e -= 1000;
    }
    while e < -1000 {
        x *= p(-1000);
        e += 1000;
    }
    if e >= -1022 {
        x * p(e)
    } else {
        // 2^e is subnormal: two steps, the second the only rounding one
        x * p(e + 52) * p(-52)
    }
}

/// d = atom i - atom j
#[inline(always)]
fn displacement(q: &[i32], i: usize, j: usize) -> [i64; 3] {
    [0, 1, 2].map(|c| q[3 * i + c] as i64 - q[3 * j + c] as i64)
}

/// The largest axis of p, dropped by the sphere code, and the other two, kept.
#[inline(always)]
fn split_axes(p: &[i64; 3]) -> (usize, usize, usize) {
    let mut drop = if p[1].wrapping_abs() > p[0].wrapping_abs() {
        1
    } else {
        0
    };
    if p[2].wrapping_abs() > p[drop].wrapping_abs() {
        drop = 2;
    }
    (drop, (drop + 1) % 3, (drop + 2) % 3)
}

// ------------------------------------------------------------------------------------ layout

#[derive(Default)]
struct Layout {
    kind: Vec<u8>,
    bond: Vec<u8>,
    water_oh: f64,
    water_hh: f64,
}

#[inline(always)]
fn dist2<T: SzFloat>(x: &[T], a: usize, b: usize) -> f32 {
    let d = |c: usize| (x[3 * a + c] - x[3 * b + c]).to_f64() as f32;
    let (dx, dy, dz) = (d(0), d(1), d(2));
    dx * dx + dy * dy + dz * dz
}

/// Rigid water on frame x: kind, water_oh, water_hh.
fn detect_water<T: SzFloat>(x: &[T], atoms: usize, layout: &mut Layout) {
    layout.kind = vec![K_OTHER; atoms];
    const OH2_MIN: f32 = 0.0064;
    const OH2_MAX: f32 = 0.015625;
    const HH2_MIN: f32 = 0.0169;
    const HH2_MAX: f32 = 0.04;
    const PROBES: usize = 300;
    const MIN_COUNT: usize = 16;
    let mut cand_oh: Vec<f32> = Vec::new();
    let mut cand_hh: Vec<f32> = Vec::new();
    let probe = (atoms / PROBES).max(1);
    let mut i = 0;
    while i + 2 < atoms {
        let mut o = 0;
        while o < 5 && i + o + 2 < atoms {
            let k = i + o;
            let (a, b, c) = (dist2(x, k, k + 1), dist2(x, k, k + 2), dist2(x, k + 1, k + 2));
            if a > OH2_MIN && a < OH2_MAX && b > OH2_MIN && b < OH2_MAX && c > HH2_MIN && c < HH2_MAX {
                cand_hh.push(c.sqrt());
                cand_oh.push(a.sqrt());
                cand_oh.push(b.sqrt());
                break;
            }
            o += 1;
        }
        i += probe;
    }
    if cand_hh.len() < MIN_COUNT {
        return;
    }
    const TIGHT: f64 = 0.001;
    const TOL: f64 = 0.002;
    const LOOSE: f64 = 0.01;
    let peak = |mut v: Vec<f32>| -> f64 {
        const PEAK_WINDOW: f32 = 0.004;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mut best, mut n, mut lo) = (0usize, 0usize, 0usize);
        for hi in 0..v.len() {
            while v[hi] - v[lo] > PEAK_WINDOW {
                lo += 1;
            }
            if hi - lo + 1 > n {
                n = hi - lo + 1;
                best = (lo + hi) / 2;
            }
        }
        v[best] as f64
    };
    let hh = peak(cand_hh.clone());
    let mut oh_of_hh = Vec::new();
    for c in 0..cand_hh.len() {
        if (cand_hh[c] as f64 - hh).abs() < TOL {
            oh_of_hh.push(cand_oh[2 * c]);
            oh_of_hh.push(cand_oh[2 * c + 1]);
        }
    }
    let oh = peak(oh_of_hh);
    let (mut n_tight, mut n_loose) = (0usize, 0usize);
    for c in 0..cand_hh.len() {
        let dh = (cand_hh[c] as f64 - hh).abs();
        let da = (cand_oh[2 * c] as f64 - oh).abs();
        let db = (cand_oh[2 * c + 1] as f64 - oh).abs();
        if dh < LOOSE && da < LOOSE && db < LOOSE {
            n_loose += 1;
            n_tight += (dh < TIGHT && da < TIGHT && db < TIGHT) as usize;
        }
    }
    if n_loose < MIN_COUNT || n_tight * 10 < n_loose * 7 {
        return;
    }
    let oh2_lo = ((oh - TOL) * (oh - TOL)) as f32;
    let oh2_hi = ((oh + TOL) * (oh + TOL)) as f32;
    let hh2_lo = ((hh - TOL) * (hh - TOL)) as f32;
    let hh2_hi = ((hh + TOL) * (hh + TOL)) as f32;
    let mut waters = 0;
    let mut i = 0;
    while i + 2 < atoms {
        let a = dist2(x, i, i + 1);
        let is_water = a > oh2_lo && a < oh2_hi && {
            let b = dist2(x, i, i + 2);
            b > oh2_lo && b < oh2_hi && {
                let c = dist2(x, i + 1, i + 2);
                c > hh2_lo && c < hh2_hi
            }
        };
        if is_water {
            layout.kind[i] = K_WATER_O;
            layout.kind[i + 1] = K_WATER_H;
            layout.kind[i + 2] = K_WATER_H;
            waters += 1;
            i += 3;
        } else {
            i += 1;
        }
    }
    if waters < MIN_COUNT {
        return;
    }
    layout.water_oh = oh;
    layout.water_hh = hh;
}

/// Bonds of the other atoms on frame x: the nearest of the previous `MAX_BOND_OFFSET` atoms within
/// 0.01 .. 0.2 nm; none if under one probed atom in 16 is bonded.
fn detect_bonds<T: SzFloat>(x: &[T], atoms: usize, layout: &mut Layout) {
    let partner = |i: usize| -> u8 {
        let mut best = 1e30f32;
        let mut offset = 0u8;
        let mut o = 1;
        while o <= MAX_BOND_OFFSET && o <= i {
            let v = dist2(x, i, i - o);
            offset = if v < best { o as u8 } else { offset };
            best = if v < best { v } else { best };
            o += 1;
        }
        if best > 0.0001 && best < 0.04 { offset } else { 0 }
    };
    layout.bond = vec![0; atoms];
    let (mut probed, mut bonded) = (0usize, 0usize);
    let step = (atoms / 256).max(1);
    let mut i = 1;
    while i < atoms {
        if layout.kind[i] == K_OTHER {
            probed += 1;
            bonded += (partner(i) != 0) as usize;
        }
        i += step;
    }
    if bonded * 16 < probed {
        return;
    }
    for i in 1..atoms {
        if layout.kind[i] == K_OTHER {
            layout.bond[i] = partner(i);
        }
    }
}

// ------------------------------------------------------------------------------------ symbols

trait Sink {
    fn put(&mut self, s: usize, v: u32);
    fn put_bits(&mut self, v: u64, bits: u32);
}

struct SymbolWriter<'a> {
    streams: Vec<Vec<i32>>,
    raw: BitAppender<'a>,
}

impl Sink for SymbolWriter<'_> {
    #[inline(always)]
    fn put(&mut self, s: usize, v: u32) {
        self.streams[s].push(v as i32);
    }
    #[inline(always)]
    fn put_bits(&mut self, v: u64, bits: u32) {
        self.raw.put(v, bits);
    }
}

/// Bits of a sample, for the predictor choice.
#[derive(Default)]
struct CostEstimate {
    bits: f64,
}

impl Sink for CostEstimate {
    #[inline(always)]
    fn put(&mut self, _: usize, v: u32) {
        self.bits += (v as f64 + 1.0).log2() + 1.0;
    }
    #[inline(always)]
    fn put_bits(&mut self, _: u64, n: u32) {
        self.bits += n as f64;
    }
}

struct SymbolReader<'a> {
    streams: Vec<&'a [i32]>,
    raw: BitConsumer<'a>,
}

impl SymbolReader<'_> {
    #[inline(always)]
    fn get(&mut self, s: usize) -> Result<u32> {
        let Some((&v, rest)) = self.streams[s].split_first() else {
            return corrupt("BioMD: corrupt stream");
        };
        self.streams[s] = rest;
        Ok(v as u32)
    }
    #[inline(always)]
    fn get_bits(&mut self, bits: u32) -> Result<u64> {
        self.raw.get(bits)
    }
}

#[inline(always)]
fn put_escaped(out: &mut impl Sink, s: usize, v: u32) {
    if v < ESCAPE {
        return out.put(s, v);
    }
    out.put(s, ESCAPE + (v >> 8));
    out.put_bits((v & 255) as u64, 8);
}

#[inline(always)]
fn get_escaped(r: &mut SymbolReader, s: usize) -> Result<u32> {
    let v = r.get(s)?;
    Ok(if v < ESCAPE {
        v
    } else {
        (v.wrapping_sub(ESCAPE) << 8) | r.get_bits(8)? as u32
    })
}

struct Frame {
    cur: Vec<i32>,
    prev: Vec<i32>,
    water_oh2: i64,
    water_hh2: i64,
    box_min: [i32; 3],
    box_size: [u64; 3],
    box_bits: i32,
    side_bits: [u32; 3],
    unbonded_box: bool,
}

/// Bit width of n - 1.
fn bits_below(n: u64) -> u32 {
    64 - (n - 1).leading_zeros()
}

impl Frame {
    fn set_water_box(&mut self, size: &[u32; 3]) {
        let r = size.map(|v| v as u64);
        self.box_size = r;
        self.box_bits = if r[0] * r[1] < (1u64 << 56) / r[2] {
            bits_below(r[0] * r[1] * r[2]) as i32
        } else {
            -1
        };
        self.side_bits = r.map(bits_below);
    }

    /// A point in the box, as one number or per coordinate.
    #[inline(always)]
    fn put_box_point(&self, p: &[i32], out: &mut impl Sink) {
        let r = &self.box_size;
        let u = [0, 1, 2].map(|c| (p[c] as i64 - self.box_min[c] as i64) as u64);
        if self.box_bits >= 0 {
            return out.put_bits(u[0] + r[0] * (u[1] + r[1] * u[2]), self.box_bits as u32);
        }
        for c in 0..3 {
            out.put_bits(u[c], self.side_bits[c]);
        }
    }

    #[inline(always)]
    fn get_box_point(&mut self, i: usize, r: &mut SymbolReader) -> Result<()> {
        let rs = self.box_size;
        let u = if self.box_bits >= 0 {
            let v = r.get_bits(self.box_bits as u32)?;
            let yz = v / rs[0];
            [v - yz * rs[0], yz % rs[1], yz / rs[1]]
        } else {
            [
                r.get_bits(self.side_bits[0])?,
                r.get_bits(self.side_bits[1])?,
                r.get_bits(self.side_bits[2])?,
            ]
        };
        for c in 0..3 {
            self.cur[3 * i + c] = (self.box_min[c] as i64).wrapping_add(u[c] as i64) as i32;
        }
        Ok(())
    }

    #[inline(always)]
    fn predict_unbonded(&self, mode: usize, i: usize, c: usize) -> i64 {
        if mode != 0 {
            self.prev[3 * i + c] as i64
        } else if i > 0 {
            self.cur[3 * (i - 1) + c] as i64
        } else {
            0
        }
    }

    /// Atom i on the sphere `|d|^2 = r2` around atom j (`r2 = 0`: the previous frame's distance).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn put_sphere(
        &self,
        mode: usize,
        i: usize,
        j: usize,
        mut r2: i64,
        s_kept: usize,
        s_radial: usize,
        out: &mut impl Sink,
    ) {
        let d = displacement(&self.cur, i, j);
        let p = if mode != 0 {
            displacement(&self.prev, i, j)
        } else {
            [0; 3]
        };
        if r2 == 0 {
            r2 = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).min(1 << 28);
        }
        let (drop, k0, k1) = split_axes(if mode != 0 { &p } else { &d });
        let neg = (if mode != 0 { p[drop] } else { d[drop] }) < 0;
        let radial =
            zigzag((if neg { -d[drop] } else { d[drop] }) - isqrt_round(r2 - d[k0] * d[k0] - d[k1] * d[k1])) as u32;
        if mode == 0 {
            out.put(S_WATER_H1_FACE, (2 * drop as u32 + neg as u32) + 6 * radial.min(15));
        }
        if mode != 0 || radial >= 15 {
            out.put(s_radial, if mode == 0 { radial - 15 } else { radial });
        }
        out.put(s_kept, zigzag(d[k0] - p[k0]) as u32);
        out.put(s_kept, zigzag(d[k1] - p[k1]) as u32);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn get_sphere(
        &mut self,
        mode: usize,
        i: usize,
        j: usize,
        mut r2: i64,
        s_kept: usize,
        s_radial: usize,
        r: &mut SymbolReader,
    ) -> Result<()> {
        let mut d = [0i64; 3];
        let mut p = [0i64; 3];
        let (drop, k0, k1, neg, radial);
        if mode == 0 {
            let v = r.get(S_WATER_H1_FACE)?;
            drop = (v % 6 / 2) as usize;
            k0 = (drop + 1) % 3;
            k1 = (drop + 2) % 3;
            neg = v & 1 == 1;
            let mut rad = v / 6;
            if rad == 15 {
                rad = rad.wrapping_add(r.get(s_radial)?);
            }
            radial = rad;
        } else {
            p = displacement(&self.prev, i, j);
            if r2 == 0 {
                r2 = (p[0]
                    .wrapping_mul(p[0])
                    .wrapping_add(p[1].wrapping_mul(p[1]))
                    .wrapping_add(p[2].wrapping_mul(p[2])))
                .min(1 << 28);
            }
            (drop, k0, k1) = split_axes(&p);
            neg = p[drop] < 0;
            radial = r.get(s_radial)?;
        }
        d[k0] = p[k0].wrapping_add(unzigzag(r.get(s_kept)? as u64));
        d[k1] = p[k1].wrapping_add(unzigzag(r.get(s_kept)? as u64));
        // unsigned: a corrupt stream's squares may pass 2^63
        let rest = (r2 as u64)
            .wrapping_sub((d[k0] as u64).wrapping_mul(d[k0] as u64))
            .wrapping_sub((d[k1] as u64).wrapping_mul(d[k1] as u64));
        let rr = isqrt_round(rest as i64).wrapping_add(unzigzag(radial as u64));
        d[drop] = if neg { rr.wrapping_neg() } else { rr };
        for c in 0..3 {
            self.cur[3 * i + c] = (self.cur[3 * j + c] as i64).wrapping_add(d[c]) as i32;
        }
        Ok(())
    }

    /// H2 - O of the previous frame, or 0 beyond the lattice sizes of a water.
    #[inline(always)]
    fn previous_h2(&self, o: usize) -> [i64; 3] {
        let p = displacement(&self.prev, o + 2, o);
        if p.iter().any(|v| v.wrapping_abs() >= 16384) {
            [0; 3]
        } else {
            p
        }
    }

    #[inline(always)]
    fn put_circle(&self, mode: usize, o: usize, out: &mut impl Sink) {
        let h1 = displacement(&self.cur, o + 1, o);
        let d = displacement(&self.cur, o + 2, o);
        let axis = circle_axis(&h1);
        let side = circle_side(&h1, axis, &d);
        if mode != 0 {
            let p = self.previous_h2(o);
            out.put(
                S_WATER_H2_AXIS,
                (zigzag(d[axis] - p[axis]).wrapping_mul(2) + (side != circle_side(&h1, axis, &p)) as u64) as u32,
            );
        } else {
            out.put(S_WATER_H2_AXIS, (zigzag(d[axis]).wrapping_mul(2) + side as u64) as u32);
        }
        let point = circle_point(&h1, self.water_oh2, self.water_hh2, axis, d[axis], side);
        for c in 0..3 {
            if c != axis {
                out.put(S_WATER_H2_RESIDUAL, zigzag(d[c] - point[c]) as u32);
            }
        }
    }

    #[inline(always)]
    fn get_circle(&mut self, mode: usize, o: usize, r: &mut SymbolReader) -> Result<()> {
        let h1 = displacement(&self.cur, o + 1, o);
        let axis = circle_axis(&h1);
        let p = if mode != 0 { self.previous_h2(o) } else { [0; 3] };
        let v = r.get(S_WATER_H2_AXIS)?;
        let side = (v & 1 == 1) ^ (mode != 0 && circle_side(&h1, axis, &p));
        let point = circle_point(
            &h1,
            self.water_oh2,
            self.water_hh2,
            axis,
            p[axis].wrapping_add(unzigzag((v >> 1) as u64)),
            side,
        );
        for c in 0..3 {
            let delta = if c == axis {
                point[c]
            } else {
                point[c].wrapping_add(unzigzag(r.get(S_WATER_H2_RESIDUAL)? as u64))
            };
            self.cur[3 * (o + 2) + c] = (self.cur[3 * o + c] as i64).wrapping_add(delta) as i32;
        }
        Ok(())
    }

    fn put_atom(&self, layout: &Layout, mode: &[usize; NUM_GROUPS], i: usize, out: &mut impl Sink) {
        let q = &self.cur;
        if layout.kind[i] == K_WATER_O {
            if mode[G_WATER_O] != 0 {
                for c in 0..3 {
                    out.put(
                        S_WATER_O,
                        zigzag(q[3 * i + c] as i64 - self.prev[3 * i + c] as i64) as u32,
                    );
                }
            } else {
                self.put_box_point(&q[3 * i..], out);
            }
            self.put_sphere(
                mode[G_WATER_H],
                i + 1,
                i,
                self.water_oh2,
                S_WATER_H1_KEPT,
                S_WATER_H1_RADIAL,
                out,
            );
            self.put_circle(mode[G_WATER_H], i, out);
        } else if layout.kind[i] == K_OTHER {
            let j = i - layout.bond[i] as usize;
            if j != i && mode[G_BONDED] != 0 {
                self.put_sphere(1, i, j, 0, S_BOND_KEPT, S_BOND_RADIAL, out);
            } else if j != i {
                for c in 0..3 {
                    put_escaped(
                        out,
                        S_BOND_DELTA,
                        zigzag(q[3 * i + c] as i64 - q[3 * j + c] as i64) as u32,
                    );
                }
            } else if self.unbonded_box {
                self.put_box_point(&q[3 * i..], out);
            } else {
                for c in 0..3 {
                    let pred = self.predict_unbonded(mode[G_UNBONDED], i, c);
                    put_escaped(out, S_UNBONDED, zigzag(q[3 * i + c] as i64 - pred) as u32);
                }
            }
        }
    }

    fn get_atom(&mut self, layout: &Layout, mode: &[usize; NUM_GROUPS], i: usize, r: &mut SymbolReader) -> Result<()> {
        if layout.kind[i] == K_WATER_O {
            if mode[G_WATER_O] != 0 {
                for c in 0..3 {
                    let v = unzigzag(r.get(S_WATER_O)? as u64);
                    self.cur[3 * i + c] = (self.prev[3 * i + c] as i64).wrapping_add(v) as i32;
                }
            } else {
                self.get_box_point(i, r)?;
            }
            self.get_sphere(
                mode[G_WATER_H],
                i + 1,
                i,
                self.water_oh2,
                S_WATER_H1_KEPT,
                S_WATER_H1_RADIAL,
                r,
            )?;
            self.get_circle(mode[G_WATER_H], i, r)?;
        } else if layout.kind[i] == K_OTHER {
            let j = i - layout.bond[i] as usize;
            if j != i && mode[G_BONDED] != 0 {
                self.get_sphere(1, i, j, 0, S_BOND_KEPT, S_BOND_RADIAL, r)?;
            } else if j != i {
                for c in 0..3 {
                    let v = unzigzag(get_escaped(r, S_BOND_DELTA)? as u64);
                    self.cur[3 * i + c] = (self.cur[3 * j + c] as i64).wrapping_add(v) as i32;
                }
            } else if self.unbonded_box {
                self.get_box_point(i, r)?;
            } else {
                for c in 0..3 {
                    let pred = self.predict_unbonded(mode[G_UNBONDED], i, c);
                    self.cur[3 * i + c] = pred.wrapping_add(unzigzag(get_escaped(r, S_UNBONDED)? as u64)) as i32;
                }
            }
        }
        Ok(())
    }
}

#[inline(always)]
fn circle_axis(h1: &[i64; 3]) -> usize {
    let axis = if h1[1].wrapping_abs() < h1[0].wrapping_abs() {
        1
    } else {
        0
    };
    if h1[2].wrapping_abs() < h1[axis].wrapping_abs() {
        2
    } else {
        axis
    }
}

/// 1 when (-h1[j], h1[i]) . (v[i], v[j]) < 0, for the other two axes i, j.
#[inline(always)]
fn circle_side(h1: &[i64; 3], axis: usize, v: &[i64; 3]) -> bool {
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    h1[i].wrapping_mul(v[j]).wrapping_sub(h1[j].wrapping_mul(v[i])) < 0
}

#[inline(always)]
fn circle_point(h1: &[i64; 3], oh2: i64, hh2: i64, axis: usize, coord: i64, side: bool) -> [i64; 3] {
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    let (u, w, lim) = (h1[i], h1[j], 16384i64);
    let ok = u.wrapping_abs() < lim
        && w.wrapping_abs() < lim
        && h1[axis].wrapping_abs() < lim
        && coord.wrapping_abs() < lim
        && u * u + w * w > 0;
    let mut point = [0i64; 3];
    point[axis] = coord;
    if ok {
        let g2 = u * u + w * w;
        let m = oh2 + g2 + h1[axis] * h1[axis] - hh2 - 2 * h1[axis] * coord;
        let s = (if side { -1 } else { 1 })
            * isqrt_round(
                (4 * g2)
                    .wrapping_mul(oh2 - coord * coord)
                    .wrapping_sub(m.wrapping_mul(m)),
            );
        point[i] = round_half_away((m.wrapping_mul(u).wrapping_sub(s.wrapping_mul(w))) as f64 / (2 * g2) as f64);
        point[j] = round_half_away((m.wrapping_mul(w).wrapping_add(s.wrapping_mul(u))) as f64 / (2 * g2) as f64);
    }
    point
}

fn water_box(q: &[i32], atoms: &[u32]) -> ([i32; 3], [i32; 3]) {
    let mut lo = [0i32; 3];
    let mut hi = [0i32; 3];
    if let Some(&a) = atoms.first() {
        for c in 0..3 {
            lo[c] = q[3 * a as usize + c];
            hi[c] = lo[c];
        }
    }
    for &a in atoms {
        for c in 0..3 {
            lo[c] = lo[c].min(q[3 * a as usize + c]);
            hi[c] = hi[c].max(q[3 * a as usize + c]);
        }
    }
    (lo, hi)
}

// ------------------------------------------------------------------------------------ decomposition

pub struct BioMdDecomposition<T: SzFloat> {
    frames: usize,
    atoms: usize,
    coded_frames: usize,
    fill_value: T,
    error_bound: f64,
    step: f64,
    water_oh2: i64,
    water_hh2: i64,
    num_waters: usize,
    modes: [usize; NUM_GROUPS],
    unbonded_box: bool,
    box_min: [i32; 3],
    box_size: [u32; 3],
    raw_bits: Vec<u8>,
    _t: PhantomData<T>,
}

impl<T: SzFloat> BioMdDecomposition<T> {
    pub fn new(conf: &Config) -> Result<Self> {
        let n = conf.dims.len();
        if n > 3 || conf.dims[n - 1] != 3 {
            return Err(Error::InvalidArgument("BioMD: data must be {frames, atoms, 3}".into()));
        }
        Ok(BioMdDecomposition {
            frames: if n == 3 { conf.dims[0] } else { 1 },
            atoms: if n >= 2 { conf.dims[n - 2] } else { 1 },
            coded_frames: 1,
            fill_value: T::ZERO,
            error_bound: conf.abs_error_bound,
            step: 0.0,
            water_oh2: 0,
            water_hh2: 0,
            num_waters: 0,
            modes: [1, 0, 0, 0],
            unbonded_box: false,
            box_min: [0; 3],
            box_size: [1; 3],
            raw_bits: Vec::new(),
            _t: PhantomData,
        })
    }

    fn has_box(&self) -> bool {
        self.unbonded_box || self.num_waters > 0
    }

    fn frame(&self) -> Frame {
        Frame {
            cur: vec![0; self.atoms * 3],
            prev: vec![0; self.atoms * 3],
            water_oh2: self.water_oh2,
            water_hh2: self.water_hh2,
            box_min: [0; 3],
            box_size: [1; 3],
            box_bits: 0,
            side_bits: [0; 3],
            unbonded_box: self.unbonded_box,
        }
    }

    /// Trailing frames all of one value are stored as that value.
    fn find_fill_frames(&mut self, data: &[T]) {
        let fv = self.atoms * 3;
        self.fill_value = data[(self.frames - 1) * fv];
        let fill = self.fill_value.bits();
        self.coded_frames = self.frames;
        while self.coded_frames > 0
            && data[(self.coded_frames - 1) * fv..self.coded_frames * fv]
                .iter()
                .all(|v| v.bits() == fill)
        {
            self.coded_frames -= 1;
        }
    }

    /// Lattice step `2 eb - m ulp` for the binade below the lattice span (see SZ3).
    fn choose_step(&mut self, data: &[T]) -> Result<()> {
        if !(self.error_bound > 0.0) || !self.error_bound.is_finite() {
            return Err(Error::InvalidArgument(
                "BioMD: the error bound must be positive and finite".into(),
            ));
        }
        let abs_mask = if T::SIZE == 4 { 0x7fff_ffffu64 } else { u64::MAX >> 1 };
        let max_bits = data[..self.coded_frames * self.atoms * 3]
            .iter()
            .map(|v| v.bits() & abs_mask)
            .max()
            .unwrap_or(0);
        if max_bits >= T::INF_BITS {
            return Err(fallback("BioMD: NaN or Inf in a frame that is not trailing fill"));
        }
        let max_abs = T::from_bits_u64(max_bits).to_f64();
        let span_bits = if T::SIZE == 4 { 21 } else { 27 };
        let exponent = frexp_exp(ldexp(self.error_bound, span_bits));
        let span = ldexp(1.0, exponent);
        let m = if T::SIZE == 4 { 1.0 + 1.0 / 1024.0 } else { 8.0 };
        let margin = m * T::DENORM_MIN_F64.max(ldexp(1.0, exponent - T::DIGITS));
        self.step = 2.0 * self.error_bound - margin;
        if !(margin < self.error_bound)
            || !(1.0 / self.step).is_finite()
            || !(span <= T::MAX_F64)
            || !(span / self.step < MAX_LATTICE)
            || !(max_abs < span + self.step)
            || !((round_half_away(max_abs * (1.0 / self.step)) as f64).abs() * self.step < span)
        {
            return Err(fallback("BioMD: coordinates beyond the lattice the error bound allows"));
        }
        Ok(())
    }

    fn detect_layout(&mut self, data: &[T], layout: &mut Layout) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        detect_water(data, self.atoms, layout);
        let oh = layout.water_oh / self.step;
        let hh = layout.water_hh / self.step;
        self.water_oh2 = if oh < 16384.0 { round_half_away(oh * oh) } else { 0 };
        self.water_hh2 = if hh < 32768.0 { round_half_away(hh * hh) } else { 0 };
        if !(self.water_oh2 > 0 && self.water_oh2 < (1 << 28) && self.water_hh2 > 0 && self.water_hh2 < (1 << 30)) {
            self.water_oh2 = 0;
            self.water_hh2 = 0;
            layout.kind.fill(K_OTHER);
        }
        detect_bonds(data, self.atoms, layout);
        let (mut waters, mut bonded, mut unbonded) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..self.atoms {
            if layout.kind[i] == K_WATER_O {
                waters.push(i as u32);
            }
            if layout.kind[i] == K_OTHER {
                if layout.bond[i] != 0 {
                    &mut bonded
                } else {
                    &mut unbonded
                }
                .push(i as u32);
            }
        }
        self.num_waters = waters.len();
        (waters, bonded, unbonded)
    }

    fn read_layout(&self, r: &mut SymbolReader) -> Result<Layout> {
        let mut layout = Layout {
            kind: vec![K_OTHER; self.atoms],
            bond: vec![0; self.atoms],
            ..Default::default()
        };
        let mut o = 0usize;
        for k in 0..self.num_waters {
            let gap = r.get(S_WATER_GAP)? as usize;
            if (k > 0 && gap < 3) || gap > self.atoms || o + gap + 2 >= self.atoms {
                return corrupt("BioMD: corrupt stream");
            }
            o += gap;
            layout.kind[o] = K_WATER_O;
            layout.kind[o + 1] = K_WATER_H;
            layout.kind[o + 2] = K_WATER_H;
        }
        for i in 0..self.atoms {
            if layout.kind[i] == K_OTHER {
                let bond = r.get(S_BOND_REF)? as usize;
                if bond > MAX_BOND_OFFSET || bond > i {
                    return corrupt("BioMD: corrupt stream");
                }
                layout.bond[i] = bond as u8;
            }
        }
        Ok(layout)
    }
}

/// Per group but the water O's, the mode (0 intra, 1 previous frame) that costs fewer bits on a
/// sample of about 256 of its atoms.
fn choose_modes(frame: &Frame, layout: &Layout, groups: [&[u32]; NUM_GROUPS], mode: &mut [usize; NUM_GROUPS]) {
    for g in G_WATER_H..NUM_GROUPS {
        let sample = groups[g];
        let mut best = 1e300;
        for m in 0..2 {
            if sample.is_empty() {
                break;
            }
            let mut cost = CostEstimate::default();
            let mut trial = *mode;
            trial[g] = m;
            let stride = (sample.len() / 256).max(1);
            let mut k = 0;
            while k < sample.len() {
                frame.put_atom(layout, &trial, sample[k] as usize, &mut cost);
                k += stride;
            }
            if cost.bits < best {
                best = cost.bits;
                mode[g] = m;
            }
        }
    }
}

impl<T: SzFloat> Decomposition<T> for BioMdDecomposition<T> {
    fn compress(&mut self, conf: &Config, data: &mut [T]) -> Result<Vec<i32>> {
        let data = &data[..conf.num];
        let fv = self.atoms * 3;
        self.find_fill_frames(data);
        self.choose_step(data)?;
        let mut layout = Layout::default();
        let (waters, bonded, unbonded) = self.detect_layout(data, &mut layout);
        self.unbonded_box = self.coded_frames == 1 && waters.is_empty() && bonded.len() * 4 < self.atoms;

        let groups: [&[u32]; NUM_GROUPS] = [&waters, &waters, &bonded, &unbonded];
        let mut raw = Vec::new();
        let mut w = SymbolWriter {
            streams: vec![Vec::new(); NUM_STREAMS],
            raw: BitAppender::new(&mut raw),
        };
        for k in 0..waters.len() {
            w.put(S_WATER_GAP, waters[k] - if k > 0 { waters[k - 1] } else { 0 });
        }
        for i in 0..self.atoms {
            if layout.kind[i] == K_OTHER {
                w.put(S_BOND_REF, layout.bond[i] as u32);
            }
        }
        let mut frame = self.frame();
        let intra = [0usize; NUM_GROUPS];
        self.modes[G_WATER_H..].fill(0);
        let inv_step = 1.0 / self.step;
        for t in 0..self.coded_frames {
            for (q, &x) in frame.cur.iter_mut().zip(&data[t * fv..(t + 1) * fv]) {
                let y = x.to_f64() * inv_step;
                *q = (y + 0.5f64.copysign(y)) as i32;
            }
            if t == 0 {
                let (lo, hi) = water_box(&frame.cur, if self.unbonded_box { &unbonded } else { &waters });
                self.box_min = lo;
                frame.box_min = lo;
                self.box_size = [0, 1, 2].map(|c| (hi[c] - lo[c] + 1) as u32);
                frame.set_water_box(&self.box_size);
            }
            if t == 1 {
                choose_modes(&frame, &layout, groups, &mut self.modes);
            }
            let mode = if t > 0 { self.modes } else { intra };
            for i in 0..self.atoms {
                frame.put_atom(&layout, &mode, i, &mut w);
            }
            std::mem::swap(&mut frame.cur, &mut frame.prev);
        }
        let SymbolWriter { streams, raw: appender } = w;
        appender.flush();
        self.raw_bits = raw;

        let mut bins = Vec::with_capacity(NUM_STREAMS + streams.iter().map(|s| s.len()).sum::<usize>());
        for s in &streams {
            if s.len() > i32::MAX as usize {
                return Err(Error::InvalidArgument(
                    "BioMD: a stream of more symbols than an int counts".into(),
                ));
            }
            bins.push(s.len() as i32);
            bins.extend_from_slice(s);
        }
        Ok(bins)
    }

    fn decompress(&mut self, conf: &Config, bins: &[i32], out: &mut [T]) -> Result<()> {
        let fv = self.atoms * 3;
        let mut streams = Vec::with_capacity(NUM_STREAMS);
        let mut k = 0usize;
        for _ in 0..NUM_STREAMS {
            let Some(&n) = bins.get(k) else {
                return corrupt("BioMD: corrupt stream");
            };
            if n < 0 || n as usize > bins.len() - k - 1 {
                return corrupt("BioMD: corrupt stream");
            }
            streams.push(&bins[k + 1..k + 1 + n as usize]);
            k += n as usize + 1;
        }
        let mut r = SymbolReader {
            streams,
            raw: BitConsumer::new(&self.raw_bits),
        };
        let layout = self.read_layout(&mut r)?;
        let mut frame = self.frame();
        frame.box_min = self.box_min;
        frame.set_water_box(&self.box_size);
        let intra = [0usize; NUM_GROUPS];
        for t in 0..self.coded_frames {
            let mode = if t > 0 { self.modes } else { intra };
            let mut i = 0;
            while i < self.atoms {
                frame.get_atom(&layout, &mode, i, &mut r)?;
                i += if layout.kind[i] == K_WATER_O { 3 } else { 1 };
            }
            for (x, &q) in out[t * fv..(t + 1) * fv].iter_mut().zip(&frame.cur) {
                *x = T::from_f64(q as f64 * self.step);
            }
            std::mem::swap(&mut frame.cur, &mut frame.prev);
        }
        if r.streams.iter().any(|s| !s.is_empty()) || !r.raw.at_end() {
            return corrupt("BioMD: corrupt stream");
        }
        out[self.coded_frames * fv..conf.num.min(self.frames * fv)].fill(self.fill_value);
        Ok(())
    }

    fn save(&mut self, out: &mut Vec<u8>) -> Result<()> {
        out.put_u64(self.coded_frames as u64);
        out.put(self.fill_value);
        out.put_f64(self.step);
        out.put(self.water_oh2);
        out.put(self.water_hh2);
        out.put_u64(self.num_waters as u64);
        for &m in &self.modes {
            out.put_u8(m as u8);
        }
        out.put_u8(self.unbonded_box as u8);
        if self.has_box() {
            out.put_slice(&self.box_min);
            out.put_slice(&self.box_size);
        }
        out.put_u64(self.raw_bits.len() as u64);
        out.extend_from_slice(&self.raw_bits);
        Ok(())
    }

    fn load(&mut self, r: &mut Reader, _: &Config) -> Result<()> {
        let coded = r.u64()?;
        if coded > self.frames as u64 {
            return corrupt("BioMD: corrupt stream");
        }
        self.coded_frames = coded as usize;
        self.fill_value = r.get()?;
        self.step = r.f64()?;
        self.water_oh2 = r.get()?;
        self.water_hh2 = r.get()?;
        let waters = r.u64()?;
        self.num_waters = waters.min(self.atoms as u64) as usize;
        for m in &mut self.modes {
            let v = r.u8()?;
            if v > 1 {
                return corrupt("BioMD: corrupt stream");
            }
            *m = v as usize;
        }
        let ub = r.u8()?;
        if ub > 1 {
            return corrupt("BioMD: corrupt stream");
        }
        self.unbonded_box = ub == 1 && self.coded_frames == 1;
        if self.has_box() {
            for c in 0..3 {
                self.box_min[c] = r.get()?;
            }
            for c in 0..3 {
                self.box_size[c] = r.get()?;
            }
        }
        let raw = r.u64()?;
        if raw > r.remaining() as u64 {
            return corrupt("BioMD: corrupt stream");
        }
        self.raw_bits = r.take(raw as usize)?.to_vec();
        let mut ok = self.step.is_finite()
            && self.step > 0.0
            && waters <= self.atoms as u64
            && self.water_oh2 >= 0
            && self.water_oh2 < (1 << 28)
            && self.water_hh2 >= 0
            && self.water_hh2 < (1 << 30);
        for &side in &self.box_size {
            ok = ok && (1..=(1 << 30)).contains(&side);
        }
        if !ok {
            return corrupt("BioMD: corrupt stream");
        }
        Ok(())
    }

    fn out_range(&self) -> (i32, i32) {
        (0, i32::MAX)
    }

    fn size_est(&self) -> usize {
        96 + self.raw_bits.len()
    }

    fn max_bins(&self, conf: &Config) -> usize {
        // two symbols per value at most, the layout streams one per atom each, and the counts
        2 * conf.num + 2 * self.atoms + NUM_STREAMS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_ldexp() {
        for v in [1.0f64, 0.75, 3.0, 1e-300, 5e-324, 1e300] {
            let e = frexp_exp(v);
            let m = v / ldexp(1.0, e);
            assert!((0.5..1.0).contains(&m), "{v}: {e} {m}");
        }
        assert_eq!(ldexp(1.0, -1074), 5e-324);
        assert_eq!(ldexp(3.0, 4), 48.0);
    }
}
