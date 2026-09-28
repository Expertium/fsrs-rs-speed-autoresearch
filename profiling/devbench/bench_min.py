"""DEV ranking bench (NOT the official protocol): single process pinned to 2 logical CPUs at
HIGH priority, times compute_parameters for the 50 users via the bit-for-bit
compute_parameters_raw twin (cached arrays, so no Python preprocessing per run), min of K reps.
Timing noise is one-sided, so the min rejects interference: champ-vs-champ ~0.4% even while
another heavy job loads the machine. Usage (repo root, installed .pyd = the one timed):
    python profiling/devbench/bench_min.py out.json 7 [cpu=30] [path/to/some.pyd]
(with a .pyd path it loads that binary directly instead of the installed package; this is also the
n=1 check of the protocol: one user at a time, alone)
Compare runs with cmp_min.py. Run cache_raw.py once first.
"""
import hashlib
import json
import os
import pickle
import sys

import psutil

out, reps = sys.argv[1], int(sys.argv[2])
cpu = int(sys.argv[3]) if len(sys.argv) > 3 else 30
p = psutil.Process()
p.nice(psutil.HIGH_PRIORITY_CLASS)
p.cpu_affinity([cpu, cpu + 1])  # a 2-CPU block, like compute_parameters.py's workers
if len(sys.argv) > 4:
    import importlib.util
    import shutil
    import tempfile

    # Windows loads the extension only under its real file name: copy it into a temp dir first.
    pyd = os.path.join(tempfile.mkdtemp(prefix="bench_min_"), "fsrs_rs_python.cp312-win_amd64.pyd")
    shutil.copy2(sys.argv[4], pyd)
    spec = importlib.util.spec_from_file_location("fsrs_rs_python", pyd)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FSRS = mod.FSRS
else:
    sys.path.insert(0, os.getcwd())
    from fsrs_rs_python import FSRS  # noqa: E402

raw = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw50.pkl"), "rb"))
b = FSRS(parameters=[])
res = {}
for u, (d, r, th, off) in raw.items():
    ts = []
    for _ in range(reps):
        prm, s = b.compute_parameters_raw(d, r, th, off)
        ts.append(s * 1000)
    res[u] = {"ms": min(ts), "size": sum(1 for x in th if x),
              "h": hashlib.md5(repr(prm).encode()).hexdigest()[:8]}
json.dump(res, open(out, "w"))
print("done", out, flush=True)
