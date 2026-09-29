"""Speed-campaign measurement harness (profiling-only).

NOT timed, NOT counted by complexity.py (lives in profiling/), and never writes a
champion file in result/. It only orchestrates runs of compute_parameters.py and
compares their per-user min-of-3 times with the constraint-12 metric.

Subcommands
-----------
  build
        Rebuild the Rust extension (cargo build --release + refresh the .pyd so the
        fresh .dll is actually what Python imports).

  run LABEL [--max-user-id N] [--processes P]
        Delete the champion result file, run compute_parameters.py fresh, and snapshot
        its per-user records to profiling/measure/LABEL.jsonl. Parent process is pinned
        to HIGH priority + affinity 0xFFFFFFF0 (CPUs >=4), matching the documented
        `start /high /affinity 0xFFFFFFF0` launch; workers re-pin themselves.

  compare CHAMP CAND
        Read two snapshots and report:
          * speed_ratio  = median over users of (t_champ / t_cand)   <- THE accept metric (>=1.05)
          * mean speedup = mean of the same ratios                    (informational)
          * param/log-loss drift (for the constraint-3a bit-for-bit check)

  ab LABEL_CHAMP LABEL_CAND [--max-user-id N] [--processes P]
        Convenience: run champion build first (must already be built), snapshot; then
        prompts you to rebuild candidate. (Usually you'll drive build/run/compare by hand.)

Snapshots are plain copies of the compute_parameters output JSONL, one record per user:
  {"user", "time_ms" (=min of 3), "time_ms_runs", "parameters", "metrics":{"LogLoss"}, "size"}
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_RESULT = _REPO / "result" / "compute_parameters-FSRS-rs-short-secs-recency.jsonl"
_SNAP_DIR = _REPO / "profiling" / "measure"
_CRATE = _REPO / "fsrs_rs_python"
_MANIFEST = _CRATE / "Cargo.toml"


def _set_parent_high_priority_and_affinity() -> None:
    """Match `start /high /affinity 0xFFFFFFF0`: HIGH priority, off logical CPUs 0-3.
    Children (compute_parameters.py + its pool workers) inherit the affinity; the
    workers re-apply HIGH priority and pin themselves to disjoint 2-CPU blocks."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        h = k32.GetCurrentProcess()
        k32.SetPriorityClass(h, 0x00000080)  # HIGH_PRIORITY_CLASS
        ncpu = os.cpu_count() or 0
        if ncpu > 4:
            mask = 0
            for cpu in range(4, ncpu):
                mask |= 1 << cpu
            k32.SetProcessAffinityMask(h, mask)
    except Exception:
        pass


def cmd_build() -> int:
    print("[build] cargo build --release ...", flush=True)
    proc = subprocess.run(
        ["cargo", "build", "--release",
         "--manifest-path", str(_MANIFEST),
         "--features", "pyo3/extension-module"],
        cwd=_REPO,
    )
    if proc.returncode != 0:
        print("[build] FAILED", flush=True)
        return proc.returncode
    # Refresh the .pyd copies from the freshly built .dll. The package __init__ only
    # copies .dll -> .pyd when the .pyd is ABSENT, so a stale .pyd would shadow the
    # new build. Overwrite every .pyd next to a same-named .dll.
    n = 0
    for sub in ("release", "release/deps"):
        d = _CRATE / "target" / sub
        if not d.exists():
            continue
        for dll in d.glob("*.dll"):
            if dll.stem != "fsrs_rs_python":
                continue
            for pyd in d.glob(dll.stem + ".*"):
                if pyd.suffix == ".pyd" or pyd.name.endswith(".pyd"):
                    try:
                        shutil.copy2(dll, pyd)
                        n += 1
                    except PermissionError:
                        print(f"[build] WARN could not overwrite {pyd} (in use)", flush=True)
    print(f"[build] refreshed {n} .pyd file(s)", flush=True)
    return 0


def cmd_run(label: str, max_user_id: int, processes: int) -> int:
    _SNAP_DIR.mkdir(parents=True, exist_ok=True)
    if _RESULT.exists():
        _RESULT.unlink()
    _set_parent_high_priority_and_affinity()
    cmd = [sys.executable, "compute_parameters.py", "--algo", "FSRS-rs",
           "--short", "--secs", "--recency",
           "--processes", str(processes), "--max-user-id", str(max_user_id)]
    print(f"[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=_REPO)
    if proc.returncode != 0:
        print("[run] compute_parameters.py FAILED", flush=True)
        return proc.returncode
    snap = _SNAP_DIR / f"{label}.jsonl"
    shutil.copy2(_RESULT, snap)
    rows = [json.loads(l) for l in snap.read_text(encoding="utf-8").splitlines() if l.strip()]
    med = statistics.median(r["time_ms"] for r in rows)
    print(f"[run] snapshot -> {snap}  users={len(rows)} median_time={med:.1f}ms", flush=True)
    return 0


def _load(label: str) -> dict:
    p = _SNAP_DIR / f"{label}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["user"]: r for r in rows}


def cmd_compare(champ: str, cand: str) -> int:
    a, b = _load(champ), _load(cand)
    users = sorted(set(a) & set(b))
    if not users:
        print("[compare] no overlapping users!", flush=True)
        return 1
    ratios = []          # s_u = t_champ / t_cand  (>1 = candidate faster)
    ll_deltas = []       # candidate LogLoss - champion LogLoss
    n_param_diff = 0
    max_param_absdiff = 0.0
    for u in users:
        tc, td = a[u]["time_ms"], b[u]["time_ms"]
        if td > 0:
            ratios.append(tc / td)
        lc = a[u]["metrics"]["LogLoss"]
        ld = b[u]["metrics"]["LogLoss"]
        ll_deltas.append(ld - lc)
        pc, pd = a[u].get("parameters", []), b[u].get("parameters", [])
        if pc != pd:
            n_param_diff += 1
            if len(pc) == len(pd):
                max_param_absdiff = max(max_param_absdiff,
                                        max((abs(x - y) for x, y in zip(pc, pd)), default=0.0))
            else:
                max_param_absdiff = float("inf")

    speed_ratio = statistics.median(ratios)
    mean_speedup = statistics.mean(ratios)
    champ_med_t = statistics.median(a[u]["time_ms"] for u in users)
    cand_med_t = statistics.median(b[u]["time_ms"] for u in users)
    mean_ll_champ = statistics.mean(a[u]["metrics"]["LogLoss"] for u in users)
    mean_ll_cand = statistics.mean(b[u]["metrics"]["LogLoss"] for u in users)
    avg_ll_delta = mean_ll_cand - mean_ll_champ           # THE correctness bar (per ruling)
    max_abs_ll = max(abs(d) for d in ll_deltas)           # per-user max: only a gross-bug tell

    # Correctness gate (2026-06-02 ruling): an ABSOLUTE band on the candidate's aggregate mean
    # LogLoss, anchored to the ORIGINAL (iter-0) baseline — NOT a per-step delta vs the champion.
    # This bounds the cumulative accuracy cost of all compounding precision-trades together.
    ORIG_LL, BAND = 0.3098, 0.0015          # compute_parameters.py reference -> [0.3078, 0.3113]
    lo, hi = ORIG_LL - BAND - 0.0005, ORIG_LL + BAND   # lower end widened by 0.0005 (Andrew 2026-09-29)
    bit_for_bit = (n_param_diff == 0)
    speed_ok = speed_ratio >= 1.05
    corr_ok = lo <= mean_ll_cand <= hi

    print(f"  users compared      : {len(users)}", flush=True)
    print(f"  median time champ   : {champ_med_t:.1f} ms", flush=True)
    print(f"  median time cand    : {cand_med_t:.1f} ms", flush=True)
    print(f"  SPEED_RATIO (median): {speed_ratio:.4f}   <- accept if >= 1.05   [{'OK' if speed_ok else 'NO'}]", flush=True)
    print(f"  mean speedup        : {mean_speedup:.4f}   (informational)", flush=True)
    print(f"  mean LogLoss champ  : {mean_ll_champ:.6f}", flush=True)
    print(f"  mean LogLoss cand   : {mean_ll_cand:.6f}", flush=True)
    print(f"  d_AVG LogLoss       : {avg_ll_delta:+.6f}  (vs champ; informational now that the gate is absolute)", flush=True)
    print(f"  abs mean LogLoss    : {mean_ll_cand:.6f}  <- CORRECTNESS BAR: in [{lo:.4f}, {hi:.4f}] (orig {ORIG_LL} +-{BAND})  [{'OK' if corr_ok else 'FAIL'}]", flush=True)
    print(f"  max |per-user dLL|  : {max_abs_ll:.6f}   (diagnostic only: gross-bug tell, NOT the bar)", flush=True)
    print(f"  users w/ param diff : {n_param_diff}/{len(users)}   ({'bit-for-bit' if bit_for_bit else 'reordered (expected for graph changes)'})", flush=True)
    print(f"  max |param diff|    : {max_param_absdiff:g}   (diagnostic only)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    pr = sub.add_parser("run")
    pr.add_argument("label")
    pr.add_argument("--max-user-id", type=int, default=50)
    pr.add_argument("--processes", type=int, default=10)
    pc = sub.add_parser("compare")
    pc.add_argument("champ")
    pc.add_argument("cand")
    args = ap.parse_args()

    if args.cmd == "build":
        return cmd_build()
    if args.cmd == "run":
        return cmd_run(args.label, args.max_user_id, args.processes)
    if args.cmd == "compare":
        return cmd_compare(args.champ, args.cand)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
