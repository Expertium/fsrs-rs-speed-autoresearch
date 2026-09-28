# Per-phase profile helper: FSRS_PROFILE=1 PYTHONPATH=. UIDS=13,17 python profiling/devbench/prof_phase.py 2>&1 | grep -E "PROFILE|USER"
import sys, os, subprocess, json
sys.argv=['benchmark.py','--algo','FSRS-rs','--short','--secs','--recency','--processes','1','--max-user-id','100000']
import benchmark as bm
from fsrs_rs_python import FSRS
from data_loader import UserDataLoader
loader=UserDataLoader(bm.config)
uids=[int(x) for x in os.environ.get('UIDS','22,3,13,1,8,6,17').split(',')]
b=FSRS(parameters=[])
for u in uids:
    ds=loader.load_user_data(u); items,cids=bm.convert_to_items(ds)
    best=None
    for _ in range(3):
        p,s=b.compute_parameters(items,cids); best=s if best is None else min(best,s)
    import hashlib; print(f"USER {u} items={len(items)} total_ms={best*1000:.2f} ph={hashlib.md5(repr(p).encode()).hexdigest()[:8]}", flush=True)
