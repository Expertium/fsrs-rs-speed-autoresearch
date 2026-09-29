"""Split a raw cache (e.g. raw1000.pkl) into one pickle per user plus index.pkl (user -> item count), so
measure_paired.py workers can load one user at a time instead of the whole cache (~6 GB per process
for raw1000). Usage: python profiling/devbench/split_raw.py raw1000.pkl raw1000_split"""
import os
import pickle
import sys

here = os.path.dirname(os.path.abspath(__file__))
src, dst = os.path.join(here, sys.argv[1]), os.path.join(here, sys.argv[2])
raw = pickle.load(open(src, "rb"))
os.makedirs(dst, exist_ok=True)
index = {}
for user, arrays in raw.items():
    with open(os.path.join(dst, f"u{user}.pkl"), "wb") as f:
        pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)
    index[user] = len(arrays[0])
pickle.dump(index, open(os.path.join(dst, "index.pkl"), "wb"))
print(f"{len(index)} users -> {dst}")
