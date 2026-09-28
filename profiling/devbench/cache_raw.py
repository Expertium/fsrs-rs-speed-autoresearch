"""One-time: cache the 50 users' compact-raw arrays (benchmark.convert_to_raw) to
profiling/devbench/raw50.pkl for bench_min.py. Run from the repo root:
    python profiling/devbench/cache_raw.py
"""
import os
import pickle
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw50.pkl")
sys.path.insert(0, REPO)
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import benchmark as bm  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402


def one(u):
    return u, [a.tolist() for a in bm.convert_to_raw(UserDataLoader(bm.config).load_user_data(u))]


if __name__ == "__main__":
    with Pool(8) as p:
        res = dict(p.map(one, range(1, 51)))
    pickle.dump(res, open(OUT, "wb"))
    print("cached", len(res), "->", OUT)
