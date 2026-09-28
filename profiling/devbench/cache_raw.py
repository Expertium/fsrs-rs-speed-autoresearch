"""One-time: cache users 1..N's compact-raw arrays (benchmark.convert_to_raw) for bench_min.py and
measure_paired.py. Run from the repo root:
    python profiling/devbench/cache_raw.py            # N=50 -> profiling/devbench/raw50.pkl
    python profiling/devbench/cache_raw.py 1000       # -> profiling/devbench/raw1000.pkl
"""
import os
import pickle
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# (Pool children re-run this module with the replaced argv below, so only accept a number.)
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"raw{N_USERS}.pkl")
sys.path.insert(0, REPO)
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import benchmark as bm  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402


def one(u):
    return u, [a.tolist() for a in bm.convert_to_raw(UserDataLoader(bm.config).load_user_data(u))]


if __name__ == "__main__":
    with Pool(8) as p:
        res = dict(p.map(one, range(1, N_USERS + 1), chunksize=4))
    pickle.dump(res, open(OUT, "wb"))
    print("cached", len(res), "->", OUT)
