//! Block sampling for auto-tuning (SZ3's Sample.hpp).

use crate::types::SzFloat;

/// Calls `f` on every index `i` with `i[k]` in `(0..limit[k]).step_by(step[k])`, row-major.
fn for_each_index<const N: usize>(
    limit: [usize; N],
    step: [usize; N],
    inclusive: bool,
    mut f: impl FnMut(&[usize; N]),
) {
    let within = |v: usize, k: usize| if inclusive { v <= limit[k] } else { v < limit[k] };
    if (0..N).any(|k| !within(0, k)) {
        return;
    }
    let mut idx = [0usize; N];
    loop {
        f(&idx);
        let mut d = N;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += step[d];
            if within(idx[d], d) {
                break;
            }
            idx[d] = 0;
        }
    }
}

fn linear<const N: usize>(idx: &[usize; N], dims: &[usize]) -> usize {
    let mut p = 0;
    for k in 0..N {
        p = p * dims[k] + idx[k];
    }
    p
}

/// Starts of the `block_size` blocks whose sampled values vary by more than `abseb`.
pub fn profiling_block<T: SzFloat, const N: usize>(
    data: &[T],
    dims: &[usize],
    block_size: usize,
    abseb: f64,
    stride: usize,
) -> Vec<[usize; N]> {
    let stride = if stride == 0 { block_size } else { stride };
    let mut starts = Vec::new();
    if dims.iter().any(|&d| d < block_size) {
        return starts;
    }
    let mut limit = [0usize; N];
    for k in 0..N {
        limit[k] = dims[k] - block_size;
    }
    for_each_index(limit, [block_size; N], false, |start| {
        let s = linear(start, dims);
        let mut min = data[s];
        let mut max = data[s];
        for_each_index([block_size; N], [stride; N], true, |off| {
            let mut idx = *start;
            for k in 0..N {
                idx[k] += off[k];
            }
            let v = data[linear(&idx, dims)];
            if v < min {
                min = v;
            } else if v > max {
                max = v;
            }
        });
        if (max - min).to_f64() > abseb {
            starts.push(*start);
        }
    });
    starts
}

/// Copies the `block_size`^N block at `start`.
fn sample_block<T: SzFloat, const N: usize>(
    data: &[T],
    dims: &[usize],
    start: &[usize; N],
    block_size: usize,
) -> Vec<T> {
    if dims.iter().any(|&d| d < block_size) {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(block_size.pow(N as u32));
    for_each_index([block_size; N], [1; N], false, |off| {
        let mut idx = *start;
        for k in 0..N {
            idx[k] += off[k];
        }
        out.push(data[linear(&idx, dims)]);
    });
    out
}

/// Samples `(sample_block_size + 1)^N` blocks (`sampleBlocks`).
pub fn sample_blocks<T: SzFloat, const N: usize>(
    data: &[T],
    dims: &[usize],
    sbs: usize,
    sample_rate: f64,
    profiling: bool,
    starts: &[[usize; N]],
) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    if dims.iter().any(|&d| d < sbs) {
        return out;
    }
    let mut total: usize = 1;
    for &d in dims {
        total *= ((d - 1) / sbs) as i32 as usize;
    }
    if profiling {
        let mut stride = (starts.len() as f64 / (total as f64 * sample_rate)) as usize;
        if stride == 0 {
            stride = 1;
        }
        let mut i = 0;
        while i < starts.len() {
            out.push(sample_block(data, dims, &starts[i], sbs + 1));
            i += stride;
        }
    } else {
        let mut stride = (1.0 / sample_rate) as usize;
        if stride == 0 {
            stride = 1;
        }
        let mut limit = [0usize; N];
        for k in 0..N {
            limit[k] = dims[k] - sbs;
        }
        let mut idx = 0usize;
        for_each_index(limit, [sbs; N], false, |start| {
            if idx.is_multiple_of(stride) {
                out.push(sample_block(data, dims, start, sbs + 1));
            }
            idx += 1;
        });
    }
    out
}
