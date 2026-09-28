"""Compare bench_min.py outputs; comma-join several runs of one build to take the per-user min.
    python profiling/devbench/cmp_min.py champ1.json,champ2.json cand1.json,cand2.json
Prints the median/mean per-user speedup and how many users' parameter hashes differ
(0/50 = bit-for-bit). Use Windows paths inside comma lists (Git Bash won't convert them)."""
import json
import statistics as st
import sys


def load(arg):
    runs = [json.load(open(f)) for f in arg.split(",")]
    return {u: {"ms": min(r[u]["ms"] for r in runs), "h": runs[0][u]["h"]} for u in runs[0]}


a, b = load(sys.argv[1]), load(sys.argv[2])
rs = [a[u]["ms"] / b[u]["ms"] for u in a]
diff = sum(a[u]["h"] != b[u]["h"] for u in a)
print(f"median ratio {st.median(rs):.4f}  mean {st.mean(rs):.4f}  param-hash diffs {diff}/{len(a)}  "
      f"median ms base {st.median(x['ms'] for x in a.values()):.2f} cand {st.median(x['ms'] for x in b.values()):.2f}")
