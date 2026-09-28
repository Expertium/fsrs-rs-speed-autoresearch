#!/usr/bin/env python3
"""
Visualize the FSRS-rs speed autoresearch iteration history.

Reads ``result/history.jsonl`` and draws TWO stacked views vs iteration
(-> ``result/history_plot.png``):

1. Cumulative speedup — the running product of the *accepted* iterations' median
   ``speed_ratio`` (the accept metric, constraint 12, compounded). This is the
   real "progress" curve, and it is drift-immune: every ``speed_ratio`` is a
   within-session *paired* ratio, so cross-session machine drift cancels.
   Rejected variants are drawn where they would have landed
   (current cumulative x their ``speed_ratio``). Only the TOP-N biggest wins
   (highest per-iteration ``speed_ratio``) are labelled, with the iteration number only
   (full summaries overlapped), so the panel stays readable (``--label-top``, default 5).
2. Median per-user time (ms) — intuitive, but MACHINE-SPECIFIC and measured per
   session, so it is NOT the accept/reject metric. Informational only.

Each panel: green dots + green step line = champions (best-so-far frontier; the
iter-0 baseline is "accepted" too); grey dots = rejected variants.

Expected per-row fields: ``iteration``, ``status`` ("accepted"/"rejected"),
``summary``, ``speed_ratio`` (median per-user speedup vs the then-current
champion; 1.0 for the iter-0 baseline), and ``time_after`` (the candidate's
median per-user time, ms).

    python plot_history.py                       # -> result/history_plot.png
    python plot_history.py --no-summaries
    python plot_history.py --history alt.jsonl --out alt.png
    python plot_history.py --benchmark           # Phase-2 benchmark() history
                                                 #   result/history_benchmark.jsonl
                                                 #   -> result/history_benchmark_plot.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to a file, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO = Path(__file__).resolve().parent
HISTORY = REPO / "result" / "history.jsonl"
DEFAULT_OUT = REPO / "result" / "history_plot.png"
# Phase-2 (benchmark()) history file pair, selected by --benchmark.
HISTORY_BENCH = REPO / "result" / "history_benchmark.jsonl"
DEFAULT_OUT_BENCH = REPO / "result" / "history_benchmark_plot.png"

# The best-so-far frontier is the set of accepted records (the iter-0 baseline
# is "accepted" too). "champion" is not a status value.
CHAMPION = {"accepted"}
GREEN = "#2ca02c"
GREEN_DARK = "#176117"
GREY = "0.70"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no history at {path}")
    rows = [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    rows = [r for r in rows if r.get("time_after") is not None]
    rows.sort(key=lambda r: r["iteration"])
    return rows


def cumulative(rows: list[dict]):
    """Walk in iteration order; cumulative = running product of *accepted*
    speed_ratios (rejects do not change the champion). Returns
    (champ_pts, reject_pts), each a list of (iteration, y, row)."""
    cum = 1.0
    champ_pts, reject_pts = [], []
    for r in rows:
        sr = r.get("speed_ratio")
        sr = 1.0 if sr is None else float(sr)
        if r.get("status") in CHAMPION:
            cum *= sr
            champ_pts.append((r["iteration"], cum, r))
        else:  # place a reject where it *would* have landed, vs the live champion
            reject_pts.append((r["iteration"], cum * sr, r))
    return champ_pts, reject_pts


def _frontier(ax, cx, cy, xmax):
    ax.step(cx + [xmax], cy + [cy[-1]], where="post", color=GREEN, lw=2.3, zorder=3)
    ax.scatter(
        cx,
        cy,
        s=62,
        color=GREEN,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="champion",
    )


def nice_tick_step(max_iter: int) -> int:
    """
    Aim for around 10 x-axis ticks, while keeping integer iteration ticks.

    Examples:
      max_iter <= 10  -> step 1
      max_iter 25     -> step 3
      max_iter 80     -> step 8
    """
    return max(1, math.ceil(max_iter / 10))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--history", type=Path, default=HISTORY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="plot the Phase-2 benchmark() history (result/history_benchmark.jsonl -> "
        "result/history_benchmark_plot.png) unless --history/--out are given explicitly",
    )
    ap.add_argument(
        "--no-summaries",
        action="store_true",
        help="hide the champion iteration labels on the speedup panel",
    )
    ap.add_argument(
        "--label-top",
        type=int,
        default=5,
        help="label only the N biggest wins (highest speed_ratio) on the speedup panel (default 5)",
    )
    ap.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="summary label angle, degrees (default 40)",
    )
    args = ap.parse_args()
    # --benchmark selects the Phase-2 file pair, but only for whichever of --history/--out the
    # user left at its default (so an explicit override always wins).
    if args.benchmark:
        if args.history == HISTORY:
            args.history = HISTORY_BENCH
        if args.out == DEFAULT_OUT:
            args.out = DEFAULT_OUT_BENCH

    rows = load(args.history)
    champs = [r for r in rows if r.get("status") in CHAMPION]
    rejects = [r for r in rows if r.get("status") == "rejected"]
    if not champs:
        raise SystemExit("no champion records to plot")

    xs = [r["iteration"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    xspan = max(1, xmax - xmin)

    # Enough padding for labels, without making small histories look absurdly wide.
    xpad_lo = max(0.6, 0.04 * xspan)
    xpad_hi = max(1.0, 0.10 * xspan)

    fig, (ax_sp, ax_t) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # ---- Top panel: cumulative speedup (the accept metric, compounded) ----
    champ_pts, reject_pts = cumulative(rows)
    cx = [p[0] for p in champ_pts]
    cy = [p[1] for p in champ_pts]

    ax_sp.axhline(1.0, color="0.6", lw=1.0, ls="--", zorder=1)  # baseline

    if reject_pts:
        ax_sp.scatter(
            [p[0] for p in reject_pts],
            [p[1] for p in reject_pts],
            s=28,
            c=GREY,
            zorder=2,
            label="rejected",
        )

    _frontier(ax_sp, cx, cy, xmax)

    if not args.no_summaries:
        # Only label the TOP-N biggest wins (highest per-iteration speed_ratio among the accepted
        # champions). Annotating every champion cluttered the panel as the campaign grew; the
        # baseline (iter 0, no change) and rejects are never labelled. Each label is prefixed with
        # its iteration number only (full summaries overlapped; the history .md has them).
        labellable = [(it, y, r) for it, y, r in champ_pts if r["iteration"] != 0]
        biggest = sorted(labellable, key=lambda p: float(p[2].get("speed_ratio") or 1.0), reverse=True)
        top_iters = {p[0] for p in biggest[: max(0, args.label_top)]}
        for it, y, r in champ_pts:
            if r["iteration"] == 0 or it not in top_iters:
                continue
            ax_sp.annotate(
                f"#{it}",
                (it, y),
                textcoords="offset points",
                xytext=(0, 6),
                rotation=args.rotation,
                rotation_mode="anchor",
                ha="center",
                va="bottom",
                fontsize=9,
                color=GREEN_DARK,
                annotation_clip=False,
                clip_on=False,
            )

    ax_sp.set_ylabel("Cumulative speedup vs baseline (×)\nhigher=better", fontsize=14)
    _phase = "benchmark() " if args.benchmark else ""
    ax_sp.set_title(
        f"FSRS-rs speed autoresearch — {_phase}{len(champs) - 1} accepted speedups, "
        f"{len(rejects)} rejected (cumulative ×{cy[-1]:.1f})",
        fontsize=17,
    )
    ax_sp.grid(True, alpha=0.25)
    ax_sp.legend(loc="upper right", framealpha=0.9)

    ylo, yhi = min(cy + [1.0]), max(cy + [1.0])
    yr = (yhi - ylo) or 1.0
    # Room above the highest point so its label stays clear of the upper-right legend.
    pad_top = (0.35 if not args.no_summaries else 0.25) * yr
    ax_sp.set_ylim(ylo - 0.05 * yr - 0.01, yhi + pad_top + 0.01)

    # ---- Bottom panel: median time (machine-specific; NOT the accept metric) ----
    ty = [r["time_after"] for r in rows]
    tmin, tmax = min(ty), max(ty)
    tr = (tmax - tmin) or 1.0

    if rejects:
        ax_t.scatter(
            [r["iteration"] for r in rejects],
            [r["time_after"] for r in rejects],
            s=28,
            c=GREY,
            zorder=2,
            label="rejected",
        )

    _frontier(ax_t, [r["iteration"] for r in champs], [r["time_after"] for r in champs], xmax)

    ax_t.set_ylabel("Median per-user time (ms)\nlower=better", fontsize=14)
    ax_t.set_xlabel("Iteration", fontsize=14)
    ax_t.set_title(
        "Median per-user time — machine-specific & per-session; "
        "NOT the accept/reject metric (see README)",
        fontsize=11,
        color="0.40",
    )
    ax_t.grid(True, alpha=0.25)
    ax_t.legend(loc="upper right", framealpha=0.9)
    ax_t.set_ylim(0, tmax + 0.10 * tr)

    max_iter = max(xs)
    x_step = nice_tick_step(max_iter)
    ax_t.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax_t.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax_t.set_xlim(xmin - xpad_lo, xmax + xpad_hi)

    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.95))
    fig.savefig(args.out, dpi=130, bbox_inches="tight", pad_inches=0.25)

    print(
        f"wrote {args.out}  ({len(champs)} champions, {len(rejects)} rejected, "
        f"latest iter {xmax}, cumulative ×{cy[-1]:.1f})"
    )


if __name__ == "__main__":
    main()