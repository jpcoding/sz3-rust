#!/usr/bin/env python3
"""Synthetic datasets for crosscheck.py: smooth fields with noise (1-4 D; f32, f64, i32, i64) and
molecular-dynamics trajectories for ALGO_BIOMD.

usage: gen_data.py <out dir>
"""
import os
import sys

import numpy as np

FIELDS = {"d1": (200000,), "d2": (600, 700), "d3": (64, 128, 160), "d4": (20, 30, 40, 50),
          "small3": (8, 8, 128), "tiny": (37,)}
OH, HH = 0.09572, 0.15139  # rigid TIP3P water (nm)


def field(rng, shape):
    grids = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    f = np.zeros(shape)
    for k, g in enumerate(grids):
        f += np.sin((3 + 2 * k) * np.pi * g + k) * (1 + k * 0.3)
    f += 0.5 * np.exp(-sum((g - 0.4) ** 2 for g in grids) * 20)
    return f + rng.normal(0, 0.01, shape)


def rotation(rng):
    a, b, c, d = (q := rng.normal(size=4)) / np.linalg.norm(q)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def md_system(rng, nchain, nwater, box):
    """A bonded chain with side atoms, then rigid waters (O, H1, H2)."""
    ang = np.arccos(1 - HH**2 / (2 * OH**2))
    h1, h2 = np.array([OH, 0, 0]), np.array([OH * np.cos(ang), OH * np.sin(ang), 0])
    atoms, p = [], rng.uniform(1, box - 1, 3)
    for i in range(nchain):
        p = p + rotation(rng) @ np.array([0.15, 0, 0])
        atoms.append(p.copy())
        if i % 3 == 0:
            atoms.append(p + rotation(rng) @ np.array([0.109, 0, 0]))
    nchain_atoms = len(atoms)
    for _ in range(nwater):
        o, r = rng.uniform(0, box, 3), rotation(rng)
        atoms += [o, o + r @ h1, o + r @ h2]
    return np.array(atoms), nchain_atoms


def md_trajectory(rng, x0, frames, nchain_atoms):
    out, x = [x0], x0.copy()
    for _ in range(1, frames):
        x = x.copy()
        x[:nchain_atoms] += rng.normal(0, 0.002, (nchain_atoms, 3))
        for k in range(nchain_atoms, len(x) - 2, 3):
            r = rotation(rng) if rng.random() < 0.1 else np.eye(3)
            o = x[k] + rng.normal(0, 0.01, 3)
            x[k + 1], x[k + 2], x[k] = o + r @ (x[k + 1] - x[k]), o + r @ (x[k + 2] - x[k]), o
        out.append(x)
    return np.array(out)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    for name, shape in FIELDS.items():
        f = field(rng, shape)
        f.astype("<f4").tofile(f"{out}/{name}.f32")
        f.astype("<f8").tofile(f"{out}/{name}.f64")
        (f * 1000).round().astype("<i4").tofile(f"{out}/{name}.i32")
        (f * 1e6).round().astype("<i8").tofile(f"{out}/{name}.i64")
    x0, nchain_atoms = md_system(rng, 300, 3000, 5.0)
    t = md_trajectory(rng, x0, 6, nchain_atoms)
    t.astype("<f4").tofile(f"{out}/md.f32")
    t.astype("<f8").tofile(f"{out}/md.f64")
    fill = t.astype("<f4")
    fill[4:] = np.nan
    fill.tofile(f"{out}/mdfill.f32")
    rng.uniform(0, 20, (5000, 3)).astype("<f4").tofile(f"{out}/cg.f32")
    nan = t.astype("<f4")
    nan[2, 10, 1] = np.nan
    nan.tofile(f"{out}/mdnan.f32")
    # for mdz: a solid on a lattice (levels; with hops between sites), and a liquid
    frames, atoms = 60, 3000
    lat = np.stack(np.meshgrid(np.arange(15), np.arange(20), np.arange(10), indexing="ij"), -1).reshape(-1, 3)
    lat = lat[:atoms] * 2.55 + 0.3
    solid, pos = np.empty((frames, atoms, 3)), lat + rng.normal(0, 0.02, (atoms, 3))
    for t in range(frames):
        pos = lat + 0.7 * (pos - lat) + rng.normal(0, 0.02, (atoms, 3))
        if t % 7 == 3:
            lat[rng.integers(0, atoms, 5), 0] += 2.55
        solid[t] = pos
    solid.astype("<f4").tofile(f"{out}/solid.f32")
    solid[:, :, 0].astype("<f4").tofile(f"{out}/solid2.f32")
    liquid = np.cumsum(rng.normal(0, 0.01, (frames, atoms)), 0) + rng.uniform(0, 30, atoms)
    liquid.astype("<f4").tofile(f"{out}/liquid2.f32")


if __name__ == "__main__":
    main()
