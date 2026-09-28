"""PAIRED A/B timing (proposed protocol, Andrew 2026-09-28): the candidate and the champion run the
SAME user AT THE SAME TIME on neighbouring CPU blocks, so background load that changes over time
hits both equally and cancels in the ratio.

  * 5 pair slots x 2 binaries = 10 worker processes of 2 logical CPUs each (the same total load as
    the 10-process protocol), off CPUs 0-3, HIGH priority.
  * A slot takes the next user (largest first) and runs it 3 times on both binaries at once; the
    next user starts only when both finish (so neither binary gets a quieter machine at the end).
    A and B swap CPU blocks between repetitions, so neither always gets the better cores.
  * Per-user time = min of the 3 reps; speed_ratio = median over users of t_A / t_B.
  * Timed region = the Rust compute_parameters call (via the bit-for-bit compute_parameters_raw
    twin on cached arrays; run profiling/devbench/cache_raw.py once). Untimed afterwards: each
    user's log loss under the frozen evaluate() (evaluate_raw), for the band check.
  * A and B also alternate which one is started first (a fixed order leaned ~0.3% toward A).

Usage:  python profiling/measure_paired.py <A.pyd> <B.pyd> [out.json]
        (A = champion/base, B = candidate; an A/A test passes the same .pyd twice)
"""
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import statistics as st
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "profiling", "devbench", "raw50.pkl")
N_SLOTS, REPS, FIRST_CPU = 5, 3, 4


def _worker(pyd, conn):
    import importlib.util

    import psutil

    spec = importlib.util.spec_from_file_location("fsrs_rs_python", pyd)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fsrs = mod.FSRS(parameters=[])
    raw = pickle.load(open(RAW, "rb"))
    proc = psutil.Process()
    proc.nice(psutil.HIGH_PRIORITY_CLASS)
    conn.send("ready")
    last = {}
    while True:
        msg = conn.recv()
        if msg is None:
            return
        user, cpus = msg
        if cpus is None:  # untimed: log loss of this user's trained parameters (frozen evaluate)
            d, r, th, off = raw[user]
            conn.send(mod.FSRS(parameters=last[user]).evaluate_raw(d, r, th, off))
            continue
        proc.cpu_affinity(cpus)
        d, r, th, off = raw[user]
        prm, secs = fsrs.compute_parameters_raw(d, r, th, off)
        last[user] = prm
        conn.send((secs * 1000.0, hashlib.md5(repr(prm).encode()).hexdigest()[:8]))


def _slot(k, pyd_a, pyd_b, queue, results):
    base = FIRST_CPU + 4 * k
    blocks = ([base, base + 1], [base + 2, base + 3])
    conns, procs = [], []
    for pyd in (pyd_a, pyd_b):
        parent, child = mp.Pipe()
        p = mp.Process(target=_worker, args=(pyd, child))
        p.start()
        conns.append(parent)
        procs.append(p)
    for c in conns:
        assert c.recv() == "ready"
    while True:
        user = queue.get()
        if user is None:
            break
        times = ([], [])
        hashes = [None, None]
        for rep in range(REPS):
            swap = rep % 2
            # Alternate which binary is started first as well as which CPU block it gets.
            for j in ((0, 1), (1, 0))[(rep + k) % 2]:
                conns[j].send((user, blocks[j ^ swap]))
            for j, c in enumerate(conns):
                ms, h = c.recv()
                times[j].append(ms)
                hashes[j] = h
        for c in conns:
            c.send((user, None))
        ll = [c.recv() for c in conns]
        results[user] = {"a": min(times[0]), "b": min(times[1]), "a_runs": times[0],
                         "b_runs": times[1], "ha": hashes[0], "hb": hashes[1],
                         "ll_a": ll[0], "ll_b": ll[1]}
    for c in conns:
        c.send(None)
    for p in procs:
        p.join()


def main():
    pyd_a, pyd_b = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    tmp = tempfile.mkdtemp(prefix="paired_")
    # Separate file paths so each process maps its own copy (both are fsrs_rs_python.*.pyd).
    paths = []
    for tag, src in (("A", pyd_a), ("B", pyd_b)):
        os.makedirs(os.path.join(tmp, tag))
        dst = os.path.join(tmp, tag, "fsrs_rs_python.cp312-win_amd64.pyd")
        shutil.copy2(src, dst)
        paths.append(dst)
    raw = pickle.load(open(RAW, "rb"))
    order = sorted(raw, key=lambda u: -len(raw[u][0]))  # largest first
    del raw
    mgr = mp.Manager()
    queue, results = mgr.Queue(), mgr.dict()
    for u in order:
        queue.put(u)
    for _ in range(N_SLOTS):
        queue.put(None)
    slots = [mp.Process(target=_slot, args=(k, paths[0], paths[1], queue, results))
             for k in range(N_SLOTS)]
    for s in slots:
        s.start()
    for s in slots:
        s.join()
    res = dict(results)
    ratios = [res[u]["a"] / res[u]["b"] for u in res]
    diff = sum(res[u]["ha"] != res[u]["hb"] for u in res)
    summary = {"users": len(res), "speed_ratio": st.median(ratios), "mean_speedup": st.mean(ratios),
               "median_ms_a": st.median(r["a"] for r in res.values()),
               "median_ms_b": st.median(r["b"] for r in res.values()), "param_hash_diffs": diff,
               "mean_logloss_a": st.mean(r["ll_a"] for r in res.values()),
               "mean_logloss_b": st.mean(r["ll_b"] for r in res.values())}
    # Correctness bar (CLAUDE.md 3b): mean LogLoss in 0.3098 +- 0.0015, anchored to iter 0.
    summary["logloss_in_band"] = 0.3083 <= summary["mean_logloss_b"] <= 0.3113
    print(json.dumps(summary))
    if out:
        json.dump({"summary": summary, "per_user": res}, open(out, "w"))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
