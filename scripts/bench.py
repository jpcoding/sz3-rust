#!/usr/bin/env python3
"""Benchmark the Rust sz3 against the C++ sz3: best-of-N compression and decompression time.

usage: bench.py <cpp sz3> <rust sz3> <work dir> [--runs N] <file:d0xd1x...:type>...
type is f32 or f64; dimensions slowest first. Also checks the two streams are identical.
"""
import argparse
import os
import re
import subprocess

ALGOS = ["ALGO_INTERP_LORENZO", "ALGO_LORENZO_REG"]
BOUNDS = ["1e-2", "1e-4"]


def timed(cmd, pattern):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(re.search(pattern, out).group(1)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp")
    ap.add_argument("rust")
    ap.add_argument("work")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    print(f"{'file':28} {'algo':20} {'rel':6} {'ratio':>9} {'C++ comp':>9} {'Rust comp':>9} {'speedup':>7}"
          f" {'C++ dec':>8} {'Rust dec':>8} {'speedup':>7} same")
    totals = {"c": 0.0, "r": 0.0, "cd": 0.0, "rd": 0.0}
    for spec in a.specs:
        path, shape, ty = spec.rsplit(":", 2)
        dims = [int(x) for x in shape.split("x")]
        dflag = [f"-{len(dims)}"] + [str(d) for d in reversed(dims)]
        tflag = "-f" if ty == "f32" else "-d"
        for algo in ALGOS:
            cfg = os.path.join(a.work, f"{algo}.cfg")
            with open(cfg, "w") as f:
                f.write(f"[GlobalSettings]\nCmprAlgo = {algo}\n")
            for rel in BOUNDS:
                best = {}
                ratio = 0.0
                for who, exe in (("c", a.cpp), ("r", a.rust)):
                    sz = os.path.join(a.work, f"{who}.sz")
                    ct, dt = [], []
                    for _ in range(a.runs):
                        t, out = timed([exe, tflag] + dflag + ["-c", cfg, "-M", "REL", rel, "-i", path, "-z", sz],
                                       r"compression time = ([\d.]+)")
                        ratio = float(re.search(r"compression ratio = ([\d.]+)", out).group(1))
                        ct.append(t)
                        t, _ = timed([exe, tflag, "-s", sz, "-o", os.path.join(a.work, f"{who}.out")],
                                     r"decompression time = ([\d.]+)")
                        dt.append(t)
                    best[who] = (min(ct), min(dt))
                same = open(os.path.join(a.work, "c.sz"), "rb").read() == open(os.path.join(a.work, "r.sz"), "rb").read()
                (cc, cd), (rc, rd) = best["c"], best["r"]
                totals["c"] += cc
                totals["r"] += rc
                totals["cd"] += cd
                totals["rd"] += rd
                print(f"{os.path.basename(path)[:28]:28} {algo:20} {rel:6} {ratio:9.1f} {cc:9.3f} {rc:9.3f} {cc / rc:7.2f}"
                      f" {cd:8.3f} {rd:8.3f} {cd / rd:7.2f} {same}", flush=True)
    print(f"{'total':57} {totals['c']:9.3f} {totals['r']:9.3f} {totals['c'] / totals['r']:7.2f}"
          f" {totals['cd']:8.3f} {totals['rd']:8.3f} {totals['cd'] / totals['rd']:7.2f}")


if __name__ == "__main__":
    main()
