//! Optimal 1-D k-means by dynamic programming with SMAWK (Grønlund et al., 2017), used by MDZ to
//! find the discrete levels of solid-state MD data (SZ3's KmeansUtil.hpp).

/// Row minima of an implicitly defined totally monotone `rows x cols` matrix.
fn smawk(num_rows: usize, num_cols: usize, lookup: &dyn Fn(usize, usize) -> f32) -> Vec<usize> {
    let mut result = vec![0usize; num_rows];
    let rows: Vec<usize> = (0..num_rows).collect();
    let cols: Vec<usize> = (0..num_cols).collect();
    smawk_rec(&rows, &cols, lookup, &mut result);
    result
}

fn smawk_rec(rows: &[usize], cols: &[usize], lookup: &dyn Fn(usize, usize) -> f32, result: &mut [usize]) {
    if rows.is_empty() {
        return;
    }
    // REDUCE: the surviving columns
    let mut kept: Vec<usize> = Vec::with_capacity(rows.len());
    for &col in cols {
        while let Some(&last) = kept.last() {
            let row = rows[kept.len() - 1];
            if lookup(row, col) >= lookup(row, last) {
                break;
            }
            kept.pop();
        }
        if kept.len() < rows.len() {
            kept.push(col);
        }
    }
    let odd: Vec<usize> = rows.iter().skip(1).step_by(2).copied().collect();
    smawk_rec(&odd, &kept, lookup, result);
    // INTERPOLATE the even rows between their odd neighbours' minima
    let index: std::collections::HashMap<usize, usize> = kept.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let index_of = |c: usize| index.get(&c).copied().unwrap_or(0);
    let mut start = 0;
    let mut r = 0;
    while r < rows.len() {
        let row = rows[r];
        let stop = if r < rows.len() - 1 {
            index_of(result[rows[r + 1]])
        } else {
            kept.len() - 1
        };
        let mut argmin = kept[start];
        let mut min = lookup(row, argmin);
        for c in start + 1..=stop {
            let v = lookup(row, kept[c]);
            if v < min {
                argmin = kept[c];
                min = v;
            }
        }
        result[row] = argmin;
        start = stop;
        r += 2;
    }
}

/// Cluster costs in O(1) from prefix sums.
struct CostCalculator {
    cumsum: Vec<f64>,
    cumsum2: Vec<f64>,
}

impl CostCalculator {
    fn new(v: &[f32]) -> Self {
        let mut cumsum = vec![0.0];
        let mut cumsum2 = vec![0.0];
        for (i, &x) in v.iter().enumerate() {
            let x = x as f64;
            cumsum.push(x + cumsum[i]);
            cumsum2.push(x * x + cumsum2[i]);
        }
        CostCalculator { cumsum, cumsum2 }
    }

    fn calc(&self, i: usize, j: usize) -> f64 {
        if j < i {
            return 0.0;
        }
        let n = (j - i + 1) as f64;
        let mu = (self.cumsum[j + 1] - self.cumsum[i]) / n;
        let mut r = self.cumsum2[j + 1] - self.cumsum2[i];
        r += n * (mu * mu);
        r -= (2.0 * mu) * (self.cumsum[j + 1] - self.cumsum[i]);
        r
    }
}

/// Clusters `values` into at most `k` groups and picks the count where the cost drops sharply.
/// Returns the centroids, or `None` when no such count exists (SZ3's `cluster`).
pub fn cluster(values: &[f32], k: usize) -> Option<Vec<f32>> {
    let n = values.len();
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cost = CostCalculator::new(&sorted);
    let mut d = vec![0f32; k * n];
    let mut t = vec![0usize; k * n];
    for i in 0..n {
        d[i] = cost.calc(0, i) as f32;
    }
    let mut ratio_avg = 0f64;
    let mut found = false;
    let mut best_k = 0;
    for k_ in 1..k {
        let prev = &d[(k_ - 1) * n..k_ * n];
        let c = |i: usize, j: usize| -> f32 {
            let col = if i < j.wrapping_sub(1) { i } else { j.wrapping_sub(1) };
            (prev[col] as f64 + cost.calc(j, i)) as f32
        };
        let argmins = smawk(n, n, &c);
        let mins: Vec<f32> = argmins.iter().enumerate().map(|(i, &a)| c(i, a)).collect();
        d[k_ * n..(k_ + 1) * n].copy_from_slice(&mins);
        t[k_ * n..(k_ + 1) * n].copy_from_slice(&argmins);
        let ratio = d[(k_ - 1) * n + n - 1] / d[k_ * n + n - 1];
        ratio_avg = (ratio_avg * (k_ - 1) as f64 + ratio as f64) / k_ as f64;
        if ratio as f64 / ratio_avg > 1.5 {
            best_k = k_ + 1;
            found = true;
        } else if found {
            break;
        }
    }
    if !found {
        return None;
    }
    let k = best_k;
    let mut centroids = vec![0f32; k];
    let mut tt = n;
    let mut k_ = k - 1;
    let mut n_ = n - 1;
    loop {
        let t_ = tt;
        tt = t[k_ * n + n_];
        let mut centroid = 0f32;
        for i in tt..t_ {
            centroid += (sorted[i] - centroid) / (i - tt + 1) as f32;
        }
        centroids[k_] = centroid;
        if tt == 0 {
            break;
        }
        k_ -= 1;
        n_ = tt - 1;
    }
    Some(centroids)
}

/// Evenly spaced levels of the data: (start, offset, count), count 0 when there are none
/// (SZ3's `get_cluster`). With `sample_num < data.len()`, a fixed-seed sample is clustered (SZ3
/// seeds from `random_device`, so its sample differs run to run).
pub fn get_cluster(data: &[f32], sample_num: usize) -> (f32, f32, i32) {
    let max = data.iter().copied().fold(data[0], |a, b| if a < b { b } else { a });
    let sample: Vec<f32> = if data.len() == sample_num {
        data.to_vec()
    } else {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        (0..sample_num)
            .map(|_| {
                seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = seed;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                data[((z ^ (z >> 31)) % data.len() as u64) as usize]
            })
            .collect()
    };
    let Some(cents) = cluster(&sample, 150) else {
        return (0.0, 0.0, 0);
    };
    let k = cents.len();
    let offset = (cents[k - 1] - cents[0]) / (k - 1) as f32;
    let mut start = cents[0];
    for (i, &c) in cents.iter().enumerate().skip(1) {
        start += c - i as f32 * offset;
    }
    start /= k as f32;
    let level_num = ((max as f64 - start as f64) / offset as f64).round() as i32 + 1;
    (start, offset, level_num)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_lattice_levels() {
        // values near 10 evenly spaced levels
        let mut seed: u64 = 1;
        let v: Vec<f32> = (0..3000)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (i % 10) as f32 * 2.5 + 1.0 + ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.02
            })
            .collect();
        let (start, offset, num) = get_cluster(&v, v.len());
        assert!((offset - 2.5).abs() < 0.01, "{start} {offset} {num}");
        assert!((start - 1.0).abs() < 0.01);
        assert_eq!(num, 10);
    }
}
