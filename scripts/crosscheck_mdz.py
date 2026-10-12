#!/usr/bin/env python3
"""Compare the Rust and C++ mdz: compressor chosen per batch, compression ratio and max error.

usage: crosscheck_mdz.py <cpp mdz> <rust mdz> <data dir>   (data from gen_data.py)
Frames of at most 5000 atoms, where C++ mdz samples deterministically.
"""
import subprocess
import sys

CASES = [
    "solid2.f32 -2 60 3000 -r 1e-3",
    "solid2.f32 -2 60 3000 -a 0.01 10",
    "solid.f32 -3 60 3000 3 -r 1e-4 20",
    "liquid2.f32 -2 60 3000 -r 1e-3 10",
    "liquid2.f32 -2 60 3000 -r 1e-3 0 4",
    "solid2.f32 -2 60 3000 -r 1e-3 0 0",
    "solid2.f32 -2 60 3000 -r 1e-3 0 1",
    "solid2.f32 -2 60 3000 -r 1e-3 15 3",
]


def summary(exe, args):
    out = subprocess.run([exe] + args, capture_output=True, text=True, check=True).stdout
    return sorted(l for l in out.splitlines() if l.startswith(("From", "Compression ratio", "Max absolute")))


def main():
    cpp, rust, data = sys.argv[1:4]
    ok = 0
    for case in CASES:
        f, *args = case.split()
        args = [f"{data}/{f}"] + args
        same = summary(cpp, args) == summary(rust, args)
        ok += same
        print("same" if same else "DIFF", case)
    print(f"{ok}/{len(CASES)} mdz cases match")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
