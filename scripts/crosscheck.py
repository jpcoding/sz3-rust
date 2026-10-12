#!/usr/bin/env python3
"""Cross-check the Rust sz3 against the C++ sz3.

For each dataset, type, algorithm and error bound:
  * both compress the same input; the streams should be byte-identical,
  * each decompresses the other's stream; outputs should equal the stream author's own output.

usage: crosscheck.py <cpp sz3> <rust sz3> <data dir> <work dir>
Data files are named <name>.<f32|f64|i32|i64>; shapes are listed in SHAPES.
"""
import itertools
import os
import subprocess
import sys

SHAPES = {
    "tiny": (37,),
    "d1": (200000,),
    "d2": (600, 700),
    "small3": (8, 8, 128),
    "d3": (64, 128, 160),
    "d4": (20, 30, 40, 50),
}
TYPES = {"f32": ["-f"], "f64": ["-d"], "i32": ["-I", "32"], "i64": ["-I", "64"]}
ALGOS = ["ALGO_INTERP_LORENZO", "ALGO_INTERP", "ALGO_LORENZO_REG", "ALGO_NOPRED", "ALGO_LOSSLESS"]
BOUNDS = [["-M", "ABS", "1e-3"], ["-M", "REL", "1e-4"], ["-M", "ABS_AND_REL", "-A", "1e-2", "-R", "1e-5"]]
EXTRA_CFG = {
    "ALGO_LORENZO_REG/l2": "Lorenzo = true\nLorenzo2ndOrder = true\nRegression = true\n",
    "ALGO_LORENZO_REG/lonly": "Lorenzo = true\nLorenzo2ndOrder = false\nRegression = false\n",
    "ALGO_LORENZO_REG/ronly": "Lorenzo = false\nLorenzo2ndOrder = false\nRegression = true\n",
    "ALGO_INTERP/linear": "InterpolationAlgo = INTERP_ALGO_LINEAR\n",
    # OpenMP layout: chunked along the first dimension, OMP_NUM_THREADS chunks
    "ALGO_INTERP_LORENZO/omp": "[GlobalSettings]\nOpenMP = true\n",
    "ALGO_LORENZO_REG/omp": "[GlobalSettings]\nOpenMP = true\n",
    "ALGO_LOSSLESS/omp": "[GlobalSettings]\nOpenMP = true\n",
}
# Both tools pick the OpenMP chunk count from here; 3 gives uneven chunks.
os.environ["OMP_NUM_THREADS"] = "3"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def read(p):
    with open(p, "rb") as f:
        return f.read()


def main():
    cpp, rust, data, work = sys.argv[1:5]
    only = sys.argv[5] if len(sys.argv) > 5 else ""
    os.makedirs(work, exist_ok=True)
    cases = [(a, None) for a in ALGOS] + [(k.split("/")[0], k) for k in EXTRA_CFG]
    total = ok = 0
    failures = []
    for (name, shape), (ty, tflag), (algo, extra), bound in itertools.product(
        SHAPES.items(), TYPES.items(), cases, BOUNDS
    ):
        tag = f"{name}.{ty}.{(extra or algo).replace('/', '-')}.{'_'.join(bound[1:3])}"
        if only and only not in tag:
            continue
        if extra and not extra.endswith("/omp") and bound is not BOUNDS[0]:
            continue
        src = os.path.join(data, f"{name}.{ty}")
        cfg = os.path.join(work, f"{tag}.cfg")
        with open(cfg, "w") as f:
            f.write(f"[GlobalSettings]\nCmprAlgo = {algo}\n[AlgoSettings]\n{EXTRA_CFG.get(extra, '')}")
        dims = [f"-{len(shape)}"] + [str(d) for d in reversed(shape)]
        base = tflag + dims + ["-c", cfg] + bound
        total += 1
        try:
            outs = {}
            for who, exe in (("c", cpp), ("r", rust)):
                run([exe] + base + ["-i", src, "-z", f"{work}/{tag}.{who}.sz"])
            same = read(f"{work}/{tag}.c.sz") == read(f"{work}/{tag}.r.sz")
            for who, exe in (("c", cpp), ("r", rust)):
                for stream in ("c", "r"):
                    out = f"{work}/{tag}.{stream}.by{who}.out"
                    run([exe] + tflag + ["-s", f"{work}/{tag}.{stream}.sz", "-o", out])
                    outs[(who, stream)] = read(out)
            cross = outs[("c", "r")] == outs[("r", "r")] and outs[("r", "c")] == outs[("c", "c")]
            if same and cross:
                ok += 1
            else:
                failures.append(f"{tag}: identical_stream={same} cross_decode={cross}")
        except RuntimeError as e:
            failures.append(f"{tag}: error {str(e)[:400]}")
    for f in failures:
        print("FAIL", f)
    print(f"{ok}/{total} cases byte-identical and cross-decodable")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
