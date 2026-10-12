//! Random inputs for the ports of SZ3's module tests: the engines `std::mt19937` and
//! `std::mt19937_64`, and the distributions those tests draw from (as libstdc++ computes them).

#![allow(dead_code)]

/// `std::mt19937`.
pub struct Mt32 {
    s: [u32; 624],
    i: usize,
}

impl Mt32 {
    pub fn new(seed: u32) -> Self {
        let mut s = [0u32; 624];
        s[0] = seed;
        for i in 1..624 {
            s[i] = 1812433253u32
                .wrapping_mul(s[i - 1] ^ (s[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt32 { s, i: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.s[k] & 0x8000_0000) | (self.s[(k + 1) % 624] & 0x7fff_ffff);
                self.s[k] = self.s[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.i = 0;
        }
        let mut y = self.s[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `uniform_real_distribution<float>(a, b)`.
    pub fn uniform_f32(&mut self, a: f32, b: f32) -> f32 {
        let mut u = self.next_u32() as f32 / 4294967296.0f32;
        if u >= 1.0 {
            u = 1.0 - f32::EPSILON / 2.0;
        }
        a + (b - a) * u
    }

    /// `generate_canonical<double, 53>`: two draws.
    pub fn canonical(&mut self) -> f64 {
        let lo = self.next_u32() as f64;
        let hi = self.next_u32() as f64;
        let u = (lo + hi * 4294967296.0) / 18446744073709551616.0;
        if u >= 1.0 { 1.0 - f64::EPSILON / 2.0 } else { u }
    }

    /// `uniform_real_distribution<double>(a, b)`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.canonical()
    }
}

/// `std::mt19937_64`.
pub struct Mt64 {
    s: [u64; 312],
    i: usize,
}

impl Mt64 {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 312];
        s[0] = seed;
        for i in 1..312 {
            s[i] = 6364136223846793005u64
                .wrapping_mul(s[i - 1] ^ (s[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Mt64 { s, i: 312 }
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.i >= 312 {
            for k in 0..312 {
                let y = (self.s[k] & 0xffff_ffff_8000_0000) | (self.s[(k + 1) % 312] & 0x7fff_ffff);
                self.s[k] = self.s[(k + 156) % 312] ^ (y >> 1) ^ if y & 1 != 0 { 0xb502_6f5a_a966_19e9 } else { 0 };
            }
            self.i = 0;
        }
        let mut y = self.s[self.i];
        self.i += 1;
        y ^= (y >> 29) & 0x5555_5555_5555_5555;
        y ^= (y << 17) & 0x71d6_7fff_eda6_0000;
        y ^= (y << 37) & 0xfff7_eee0_0000_0000;
        y ^ (y >> 43)
    }

    /// `generate_canonical<double, 53>`: one draw.
    pub fn canonical(&mut self) -> f64 {
        let u = self.next_u64() as f64 / 18446744073709551616.0;
        if u >= 1.0 { 1.0 - f64::EPSILON / 2.0 } else { u }
    }

    /// `exponential_distribution<double>(lambda)`.
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -(1.0 - self.canonical()).ln() / lambda
    }

    /// `geometric_distribution<int>(p)`.
    pub fn geometric(&mut self, p: f64) -> usize {
        let log_1_p = (1.0 - p).ln();
        loop {
            let c = ((1.0 - self.canonical()).ln() / log_1_p).floor();
            if c < i32::MAX as f64 {
                return c as usize;
            }
        }
    }

    /// Fisher-Yates.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next_u64() % (i as u64 + 1)) as usize);
        }
    }
}

/// `normal_distribution<double>(0, 1)` over `std::mt19937` (Marsaglia's polar method, which yields
/// its values in pairs).
#[derive(Default)]
pub struct Normal {
    saved: Option<f64>,
}

impl Normal {
    pub fn sample(&mut self, g: &mut Mt32) -> f64 {
        if let Some(v) = self.saved.take() {
            return v;
        }
        loop {
            let x = 2.0 * g.canonical() - 1.0;
            let y = 2.0 * g.canonical() - 1.0;
            let r2 = x * x + y * y;
            if r2 <= 1.0 && r2 != 0.0 {
                let m = (-2.0 * r2.ln() / r2).sqrt();
                self.saved = Some(x * m);
                return y * m;
            }
        }
    }
}

pub fn max_abs_error<T: Copy + Into<f64>>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x.into() - y.into()).abs())
        .fold(0.0, f64::max)
}
