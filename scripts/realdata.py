#!/usr/bin/env python3
"""Compare the Rust and C++ sz3 on real datasets: identical streams, cross-decoding, ratio, speed.

usage: realdata.py <cpp sz3> <rust sz3> <work dir> <file:d0xd1x...>...
Dimensions are slowest first (as SDRBench names them).
"""
import os
import re
import subprocess
import sys

ALGOS = ["ALGO_INTERP_LORENZO", "ALGO_LORENZO_REG"]
BOUNDS = ["1e-2", "1e-4"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def num(pat, s):
    return float(re.search(pat, s).group(1))


def main():
    cpp, rust, work = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    print(f"{'file':34} {'algo':20} {'rel':6} {'ratio':>8} {'cpp c/d s':>13} {'rust c/d s':>13}  same  cross")
    bad = 0
    for spec in sys.argv[4:]:
        path, shape = spec.rsplit(":", 1)
        dims = [int(x) for x in shape.split("x")]
        dflag = [f"-{len(dims)}"] + [str(d) for d in reversed(dims)]
        for algo in ALGOS:
            cfg = os.path.join(work, f"{algo}.cfg")
            with open(cfg, "w") as f:
                f.write(f"[GlobalSettings]\nCmprAlgo = {algo}\n")
            for rel in BOUNDS:
                t = {}
                for who, exe in (("c", cpp), ("r", rust)):
                    out = run([exe, "-f"] + dflag + ["-c", cfg, "-M", "REL", rel, "-i", path, "-z", f"{work}/{who}.sz"])
                    ratio = num(r"compression ratio = ([\d.]+)", out)
                    ct = num(r"compression time = ([\d.]+)", out)
                    out = run([exe, "-f", "-s", f"{work}/{who}.sz", "-o", f"{work}/{who}.out"])
                    t[who] = (ratio, ct, num(r"decompression time = ([\d.]+)", out))
                with open(f"{work}/c.sz", "rb") as a, open(f"{work}/r.sz", "rb") as b:
                    same = a.read() == b.read()
                # each decodes the other's stream
                run([cpp, "-f", "-s", f"{work}/r.sz", "-o", f"{work}/r.byc.out"])
                run([rust, "-f", "-s", f"{work}/c.sz", "-o", f"{work}/c.byr.out"])
                rd = lambda p: open(p, "rb").read()
                cross = rd(f"{work}/r.byc.out") == rd(f"{work}/r.out") and rd(f"{work}/c.byr.out") == rd(f"{work}/c.out")
                bad += not (same and cross)
                print(
                    f"{os.path.basename(path)[:34]:34} {algo:20} {rel:6} {t['r'][0]:8.2f} "
                    f"{t['c'][1]:6.2f}/{t['c'][2]:<6.2f} {t['r'][1]:6.2f}/{t['r'][2]:<6.2f}  {same!s:5} {cross}",
                    flush=True,
                )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
