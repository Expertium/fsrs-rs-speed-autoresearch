# FSRS-rs speed autoresearch

You are the autoresearcher. Your job is to make FSRS parameter optimization **faster** (FSRS-7 in this fork for Phases 1–2; **FSRS-6 in the upstream repo for the current Phase 3**), so real Anki users spend less time staring at the optimizer's progress bar and more time doing reviews. Making the number go down only counts if it actually helps users:

- A speedup that only works on AVX-512 CPUs and Void Linux doesn't count.
- A speedup that makes FSRS less accurate doesn't count (that's what the correctness bars are for).
- A speedup that only works when processing multiple users in parallel doesn't count.

You pick the directions, run the experiments, and report. Balancing broad exploration against squeezing one idea dry is your call. Budget: ~150–300 iterations over ~2 weeks — plenty of room for bold structural bets, so don't fear "wasting" iterations or declare convergence early. The user is a Python/PyTorch person, not a Rust dev: ask only for high-level feedback, never Rust technicalities.

**The slow era (as of ~iter23, ~×100 product / ~×96 anchored).** The big and medium levers are spent — autodiff removal, the O(N) expanding window, SIMD across cards, the hand-rolled optimizer. (The minimax accuracy bands had been ~exhausted by iter23's windowed-minimax trade, but that trade was later REVERTED on cost/benefit review — see the Phase-3 status block — restoring ~2e-4 of cp-band headroom.) From here, **accepted speedups will be rare**: most candidates will reject on the 5% median floor or a correctness band. **That is expected, not failure — do NOT get discouraged if your success rate (accepted / (accepted + rejected)) drops sharply, and do NOT declare the campaign "converged" or over just because the accept rate falls.** A run of rejections is normal in this regime and is still useful signal (it maps the wall). Keep probing: bit-for-bit structural/setup trims, creative math-preserving reformulations, and stacking several sub-5% bit-for-bit wins into one logged iteration (the "stack to 5%" ruling). Record every attempt (including rejects) in the history as usual.

## ⭐ PHASE 3 — port the wins to UPSTREAM fsrs-rs (FSRS-6) and PR them

**✅ STATUS (2026-06-05): MERGED UPSTREAM — Phase 3 DONE.** My PR #417 (3 bit-for-bit scalar commits) was CLOSED and **superseded by maintainer asukaminato0721's [PR #419](https://github.com/open-spaced-repetition/fsrs-rs/pull/419) "Speed up compute_parameters without SIMD", MERGED 2026-06-05** — it landed this repo's scalar windowed-`compute_parameters` wins (hoist loop-invariant work; skip dead padding + inactive stability branches in `step_forward`; reuse train batches for validation; avoid redundant cloning/card-id realignment in outlier filtering; reuse scratch buffers; in-place clip), on top of the earlier merged [PR #411](https://github.com/open-spaced-repetition/fsrs-rs/pull/411) (analytic gradient + host Adam + `card_ids` window). **SIMD/FMV stayed OUT** (x86-only, didn't replicate on ARM — auto-vec minimax ~neutral on FSRS-6's already-fast analytic gradient). Full detail + the f32-backward experiment (local branch `f32-backward-experiment`) in memory `fsrs-phase3-upstream-port`. The directive that produced this:

**Phases 1 & 2 are DONE in THIS fork (FSRS-7), validated + pushed.** Phase 1 (`compute_parameters()`): **×98.2** 1000-user portable / ~×188 AVX2 (was ×103.8 pre-revert — see the iter23-revert note below). Phase 2 (`benchmark()`): **×1.23** (iter1 +16% + iter2 +6%, both bit-for-bit; champion `.pyd` 3d1147ed; history in `result/history_benchmark.{jsonl,md}`).

**⚠ iter23 REVERTED (2026-06-05) on cost/benefit review.** iter23's cruder windowed minimax (exp8 deg3→deg2 + ln8 deg2→deg1) was undone: it had cost **+0.00035 cp log loss** for what re-measured — *after* iter24's exp-fusion thinned the per-timestep exp work — as only **~5–6% speed**, not the +9.6% it originally claimed. The windowed `compute_parameters` path now uses the accurate deg-3 `exp8` / deg-2 `ln8` again (the 6 `::<true>` call sites in `analytic.rs` flipped to `::<false>`). Champion cp mean LogLoss **0.311000** (was 0.311245, still in band); 1000-user speedup **×98.2** (was ×103.8). The history plot/`.jsonl` are **KEPT** as the campaign record (the revert is NOT logged as an iteration, per Andrew). **Epoch-buyback diagnostic:** running **~20 epochs** (vs the shipped 8) brings cp loss back to **0.3098** (the iter0 baseline) at ~2.26× the 8-epoch time → still **~×43** vs iter0. The shipped default stays 8 epochs (constraint 4); the `num_epochs` arg added to `compute_parameters` (default 8) is for the `profiling/epoch_sweep.py` diagnostic. The banked "division-free ln" accuracy buyback is SUBSUMED by this revert (the full revert reaches 0.311000, more accurate than its 0.311108) — not committed. The fsrs-rs maintainers **MERGED a subset upstream in PR #411** — analytic gradient + host Adam + `card_ids` window, **scalar**, crediting this repo → ~9.3× on their `compute_parameters`. The SIMD did **not** replicate on Apple Silicon (it's x86-effective; ARM gets the algorithmic core ≈ ×32, see README portability section).

**New target: take the REMAINING optimizations to the REAL upstream [`open-spaced-repetition/fsrs-rs`](https://github.com/open-spaced-repetition/fsrs-rs) and PR them. ⚠ Upstream is FSRS-6, NOT FSRS-7 → the formulas differ** (so the analytic gradient + SIMD kernel must be RE-DERIVED/adapted for FSRS-6; you can't copy this fork's `analytic.rs` verbatim). Andrew's directive (≈3× more headroom there, it seems):
1. **Download the upstream fsrs-rs repo.** ⚠ Make sure this fork's local `fsrs-rs/` crate does NOT shadow/override imports of anything that uses the real fsrs-rs (path-dep collision) — keep the upstream clone in a SEPARATE dir and point the harness at it explicitly.
2. **Plug their `compute_parameters` into the same pipeline we have here** (the `compute_parameters.py` / `profiling/measure.py` timing harness). Adapt the Python glue to their FSRS-6 API/types.
3. **Locate their benchmark suite** (they have `cargo bench --bench parameters`) and use it to measure time.
4. **Optimize their `compute_parameters`** — port the SIMD `f32x8` forward+gradient, minimax `exp`/`ln`, build-once batching, hoists/dedup/skip-last that PR #411 left out, re-derived for FSRS-6. **Keep checking log loss** so accuracy doesn't regress.
5. **Make a PR to fsrs-rs.**

**Carries over / templates:** ALL of this fork's optimizations (`fsrs-rs/src/analytic.rs` SIMD kernels + `training.rs` Adam/penalty/batching) and the Phase-1 history (`result/history.jsonl`) are the playbook. Constraints carry over (portable per constraint 7 — but note SIMD is x86-effective, doesn't help wide-OoO ARM; ≤2 threads/user; don't change epochs/seed/hyperparams; **accuracy must not regress**) + the min-of-3 measurement discipline. The accept bar shifts from the strict 5% campaign floor to **"meaningful speedup + no log-loss regression + clean code upstream will accept."** Most of the REST of this CLAUDE.md (Code layout, band numbers 0.3098/0.313943, etc.) describes the FSRS-7 fork — use it as reference, but Phase-3 specifics (FSRS-6 formulas, their file layout, their accuracy baseline) are discovered against the upstream clone.

## Host machine

- Windows 10 Pro 22H2 (build 19045), 64 GB RAM, Ryzen 9 5950x

## Running it

```
start /high /affinity 0xFFFFFFF0 python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50
```

- **`compute_parameters.py`** is the timing harness: it trains each user with FSRS-rs `compute_parameters()` (the function you optimize) and scores log loss with Rust `evaluate()`, using train set == test set. Per-user time (ms) is written to `result/compute_parameters-FSRS-rs-short-secs-recency.jsonl`.
- **`benchmark.py`** is the reference harness (`benchmark()` + 5-fold TimeSeriesSplit + Python forgetting curve). Use it together with **`evaluate.py`** for the bit-for-bit correctness check (constraint 3a).
- `/high /affinity 0xFFFFFFF0` plus the per-worker CPU pinning built into `compute_parameters.py` keeps the workers off logical CPUs 0–3, at HIGH priority, each on its own pair of cores — so cross-worker contention is constant rather than random. (Intended to cut measurement noise; empirically it mainly lowers/steadies the compute *level*, not the run-to-run variance — see *What the guards buy*.)
- **Big users first:** `compute_parameters.py` dispatches users largest-collection-first → smallest-last (hardcoded `USERS_BY_SIZE_DESC`). Measured benefit (3×10 experiment, 2026-06-01) is **scheduling, not noise**: dispatching the long jobs first balances the 10 workers so each run finishes ~**10% sooner** (less tail idle); run-to-run noise is unchanged. It changes only the scheduling order, not the recorded numbers (the output `.jsonl` is re-sorted by user).
- Already applied on this machine to pin the clock speed and remove turbo/thermal noise:
  ```
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 99
  powercfg /setactive SCHEME_CURRENT
  ```

## Measuring a candidate — OFFICIAL from iter34: the PAIRED protocol (Andrew 2026-09-28)

**RESOURCE LIMITS (Andrew 2026-09-29): at most 4 threads in total, and keep at least 10 GB of RAM free.** A 10-worker
paired run on the 1000-user cache (~6 GB per worker) made his PC unusable. `measure_paired.py` therefore defaults to
`FSRS_PAIRED_SLOTS=1` (2 workers x 2 CPUs = 4 threads, ~1 min per run); the 5-slot description below is the old load.

**`python profiling/measure_paired.py <champion.pyd> <candidate.pyd> [out.json]`** is the official timing for every iteration from iter34 on. It prints `speed_ratio` (median of per-user t_champion / t_candidate), `mean_speedup`, `param_hash_diffs` (0 = bit-for-bit), both mean LogLosses and `logloss_in_band` (the 3b bar). ~12 s per run.
- **Why (Andrew's idea):** the champion and the candidate run the SAME user AT THE SAME TIME, so background load that changes over time (other projects' jobs, thermal drift) hits both equally and cancels in the ratio. The old back-to-back runs were ~5 minutes apart and absorbed the load difference.
- **Design:** 5 pair slots × 2 binaries = 10 worker processes of 2 logical CPUs each (same total load as `--processes 10`), off CPUs 0-3, HIGH priority. A slot takes the next user (largest first) and runs it 3 times on both binaries at once; the next user starts only when both finish (so the faster binary never leaves the slower one a quieter machine). Between reps the two binaries swap CPU blocks AND start order (a fixed start order leaned ~0.3% toward the first). Per-user time = min of 3. Each binary is loaded from its own temp copy of the `.pyd`, so no `.pyd` swap into `target/` is needed.
- **Timed region:** the same Rust `compute_parameters()` call with the same input, fed through the bit-for-bit `compute_parameters_raw` twin from cached arrays (`profiling/devbench/raw50.pkl`; regenerate with `python profiling/devbench/cache_raw.py`, which asserts nothing — check `sum(size)` if the dataset changes). Untimed afterwards: each user's LogLoss under the frozen `evaluate()` (`evaluate_raw`).
- **Validation (2026-09-28, with a ~7-core RWKV job loading the box):** A/A ×10 (champion vs itself) medians 0.9952–1.0075 (worst 0.75%), no side favoured; the old protocol showed ~±5% under the same load. A real difference (iter30 → iter33) read ×1.114/1.111/1.115/1.116. LogLoss 0.308320 = the official harness exactly.
- **n=1 rule (Andrew 2026-09-28): a speedup must also be a gain for ONE user optimized alone.** Anything that only pays off when several users run in parallel (e.g. less contention between the 10 workers, shared caches, cross-user amortization) is NOT allowed (see also constraint 11). Check every candidate before logging it as accepted: run `python profiling/devbench/bench_min.py` (one process, pinned to 2 logical CPUs = one core, min of K reps) for the champion and the candidate, 3 alternating rounds, and compare with `cmp_min.py`. The n=1 median speedup must be clearly above 1 (beyond its ~0.5% noise); record it in the history comment. A candidate with no n=1 gain is rejected even if the paired run passes.
- Snapshot every candidate `.pyd` (e.g. `cp …/target/release/fsrs_rs_python.cp312-win_amd64.pyd <scratch>/candN.pyd`) and keep the champion's snapshot current on every accept. **Gotcha:** a running Python process that imported `fsrs_rs_python` locks `target/release/fsrs_rs_python.cp312-win_amd64.pyd`, so `measure.py build` refreshes only the unlocked copy (it prints "refreshed 1 .pyd file(s)" instead of 2). Snapshot from `target/release/fsrs_rs_python.dll` (always the fresh build) and check the sha.
- The old back-to-back `.pyd`-swap recipe below remains for reference and for anything the paired script does not cover (it is how iters 0–33 were timed).

## (Legacy, iters 0–33) Measuring a candidate: the `.pyd`-swap recipe

`profiling/measure.py` drives every timing run. "Back-to-back in the same session" means **swap pre-built `.pyd` files**, NOT rebuild the champion from git twice (slow, and the swap is byte-exact). Python imports `fsrs_rs_python/target/release/fsrs_rs_python.cp312-win_amd64.pyd` (release `""` is tried before `deps/` before `debug/` — `fsrs_rs_python/__init__.py` loads the first hit), so **that one file decides which binary runs.**

1. Candidate source in the tree → `python profiling/measure.py build` (cargo `--release` + refreshes the `.pyd`). Snapshot it so you can restore it: `cp …/target/release/fsrs_rs_python.cp312-win_amd64.pyd /tmp/cand.pyd`.
2. `python profiling/measure.py run iterN_cand` → per-user min-of-3 → `profiling/measure/iterN_cand.jsonl`.
3. Swap the champion in: `cp /tmp/champion.pyd` over **both** `…/target/release/` and `…/target/release/deps/` `fsrs_rs_python.cp312-win_amd64.pyd`.
4. `python profiling/measure.py run iterN_champ`.
5. `python profiling/measure.py compare iterN_champ iterN_cand` → `speed_ratio` (median of per-user ratios; accept ≥1.05), mean-LogLoss band check, and param-diff count (`0/50` = bit-for-bit).

**Gotchas (each one cost a confused detour at least once):**
- `run` NEVER builds; `__init__.py` copies `.dll→.pyd` only when the `.pyd` is **absent**. So after a swap the champion `.pyd` persists even though the candidate `.dll` still sits right next to it (it's ignored). **`sha256sum` the installed `.pyd` immediately before every run** so you're certain which binary you're timing — the single easiest way to silently measure the wrong thing.
- Keep `/tmp/champion.pyd` == the current champion; re-snapshot it on every accept. The fixed iter-0 baseline for the periodic unbiased re-anchor is `/tmp/anchor_iter0.pyd` (+ `profiling/measure/anchor_iter0.jsonl`).
- A clean-tree `build` produces the **candidate** (current source). To rebuild the *champion* from source instead, `git stash` first — but prefer swapping the `/tmp` snapshot (one 35 s compile saved).
- For a **bit-for-bit** change, `compare` must print `0/50` param diff and `d_AVG LogLoss +0.000000`. Any param drift on a change you believed was math-unchanged = a bug, not noise (setup-only restructuring cannot move a single bit).

## Environment quick-facts (append new gotchas here as you hit them)

Whenever you catch yourself going "wait — wrong path / wrong command / that's not how this box works," jot the resolution here so the next you (post-compaction) skips the detour. Verified on this machine:
- **No virtualenv.** Just run `python` — it's system **Python 3.12** (`C:\Users\Andrew\AppData\Local\Programs\Python\Python312\python.exe`). Don't hunt for a `.venv/`; there isn't one.
- **The Bash tool has Unix coreutils here** (Git Bash on Windows): `cp`, `ls`, `find`, `sha256sum`, `grep`, `/tmp/…` all work, and `/tmp` is a real writable dir — it's where the `.pyd` snapshots (`champion.pyd`, `anchor_iter0.pyd`, …) live. Forward-slash paths in Bash; backslash only inside `cmd`/`start`.
- **The champion record** Python reads/writes is `result/compute_parameters-FSRS-rs-short-secs-recency.jsonl`; `measure.py run` deletes+regenerates it each run, then copies it to `profiling/measure/<label>.jsonl`. It does NOT persist meaningfully between runs — always work from the `profiling/measure/*.jsonl` snapshots.
- **Incremental cargo `--release` build ≈ 35 s** (fsrs lib + binding). If a "build" finishes near-instantly, nothing recompiled — confirm you actually edited a `.rs` under `fsrs-rs/src/` or `fsrs_rs_python/src/` (editing `.py`/`.md` needs no rebuild).
- **`/tmp` is NOT the same place in Bash and in Python.** The Bash tool is Git Bash, whose `/tmp` is a real dir (where the `.pyd` snapshots live). A *Windows* Python process resolves `/tmp/foo` to `C:\tmp\foo` (drive-relative) — a different location — so handing a Bash-made `/tmp/...` path to `python` fails with FileNotFoundError. Keep `/tmp` strictly inside Bash (cp/sha256sum/diff); for anything Python must read, use a **repo-relative** path or just `git diff` a tracked file.
- **`start /high /affinity ...` is cmd.exe-only** — the documented launch line is CMD syntax. In the Bash tool `start` isn't a command (Git Bash reads `/affinity` as a path and exits 1 without launching anything). For a *correctness* run, priority/affinity don't matter → just `python benchmark.py ...`. For a *timed* run, use `python profiling/measure.py run` (it applies HIGH priority + affinity 0xFFFFFFF0 via ctypes), or run the cmd line from a real `cmd`/PowerShell.

## Code layout

```
compute_parameters.py   # ★ PHASE-1 speed harness (champion FROZEN): times Rust compute_parameters(), scores with Rust evaluate(), train==test, min-of-3
benchmark.py            # ★ PHASE-2 target + reference harness: 5-fold TimeSeriesSplit + Python forgetting curve; also the 3a bit-for-bit anchor (timed via profiling/measure_benchmark.py)
evaluate.py             # Python forgetting-curve scorer used by benchmark.py (read-only; skipped by complexity.py)
config.py               # CLI args + Config (seed=42, max_seq_len, --data path, --processes)
data_loader.py          # per-user revlog loading from ../anki-revlogs-10k/revlogs/user_id=*
utils.py                # sort_jsonl(), catch_exceptions(), evaluate(), shared helpers
complexity.py           # constraint-13 score over .py/.rs (skips target/ __pycache__/ profiling/ + benchmark/evaluate/plot_history/log_history/complexity)
log_history.py          # appends one iteration to result/history{,_benchmark}.{jsonl,md} (--benchmark = Phase 2); read-only tooling, skipped by complexity.py
plot_history.py         # plots the history -> result/history{,_benchmark}_plot.png (--benchmark = Phase 2); read-only tooling, skipped by complexity.py
features/               # preprocessing: emits N-1 expanding-window prefix-items/card (card_id-tagged so Rust folds them into ONE O(N) pass — iter18); + __init__.py
  base.py               #   builds t_history/r_history; caps each card at 2× max_seq_len reviews
  fsrs_engineer.py      #   FSRS (t_history, rating) feature engineering
  create_features.py    #   feature-builder dispatch
fsrs-rs/src/            # the Rust FSRS crate (path dep of the binding) = the main mutation surface (3a/6); Cargo.toml alongside
  analytic.rs           #   ★ HOT PATH: hand-written SIMD f32x8 forward+gradient — card_*_simd = O(N) windowed per-card recurrence (compute_parameters); batch_loss_*simd/loss_and_grad_range_simd = O(N²) per-prefix (benchmark / Phase-2 target)
  training.rs           #   ★ Rust compute_parameters() AND benchmark(); train loop, Adam, epochs/batching, card grouping + outlier filter (count-based), Dual35 penalty gradient
  model.rs              #   burn-tensor forgetting curve + per-timestep recurrence forward()/update_state() (now only the frozen evaluate()/inference path)
  inference.rs          #   evaluate() = the scorer; do NOT modify (constraint 11 anchor)
  lib.rs, error.rs      #   crate exports, DEFAULT_PARAMETERS, error types
fsrs_rs_python/         # PyO3 binding -> built (maturin) into the installed fsrs_rs_python extension; Cargo.toml alongside
  src/lib.rs            #   Python entry points; compute_parameters() + benchmark_timed() time their Rust region with a monotonic clock (PyO3 conversion excluded)
  __init__.py           #   loads the first fsrs_rs_python.*.pyd hit (release > release/deps > debug); copies .dll->.pyd only when the .pyd is ABSENT — the .pyd-swap recipe relies on this
result/                 # per-user .jsonl (compute_parameters-* = champion record; FSRS-rs-* = benchmark output; {iter0,champ,avx2}-Nu = 1000-user validation; *(old) = kept baselines) + campaign history: history{,_benchmark}.{jsonl,md} + *_plot.png
profiling/              # profiling-only tools (skipped by complexity.py): measure.py (compute_parameters .pyd-swap timing) + measure_benchmark.py (Phase-2 benchmark() timing) + profile_structure/_profile_user/profile_ram/minimax_coeffs + bench_*.rs microbenches; see "Where the time goes"
evaluation/  raw/       # benchmark.py detailed-eval / --raw prediction dumps
```
Build artifacts (`*/target/`, `__pycache__/`) and the external dataset (`../anki-revlogs-10k/`) are omitted.

## Measurement protocol (how every timing number is produced)

- Time each of the 50 users 3 times; the official per-user time is the **MIN** of the 3. Timing noise is one-sided (interference only ever slows a run), so the fastest of the three is the cleanest estimate of the true compute floor, and it also rejects the cold first run. (Empirically, min ~halves per-user run-to-run noise vs. the median, for free.)
- Always time under `--processes 10` (parallel across users). (Paired protocol from iter34: 10 workers in total = 5 per binary; see "Measuring a candidate".)
- **Timed region = `compute_parameters()` (Rust) only**: exclude data loading / I/O and the `evaluate()` call; use a monotonic clock. Python stays untimed.
- Time the champion and the candidate **in the same session** — from iter34 **simultaneously, paired per user** (`profiling/measure_paired.py`); iters 0–33 ran them back-to-back (re-baseline every comparison). Never reuse a baseline from an earlier session — the machine drifts (thermal/turbo/background load) over a 150–300 iteration campaign.
- **Noise floor (measured: 3×10 identical-build runs, 2026-06-01).** Run-to-run noise is ~**1%**, *not* the ~0.35–0.5% earlier *single* back-to-back pairs implied (those were lucky-tight draws). Across 10 identical runs the median-over-50 time has CV ≈ **0.95–1.1%**; the accept metric itself — the median per-user speedup between two runs (constraint 12) — lands within **~0.7%** of 1.0 typically and up to **~2%** in the worst of 45 pairs. So the 5% bar clears *typical* noise ~7× but **worst-case paired noise only ~2.5×** — discount sub-~2% median "speedups" as likely noise. (The median does **not** average out per-user noise — it's one order statistic, not a mean — so its CV ≈ the ~1.2% per-user CV; the median is used for robustness to the big collections, not variance reduction.)
- **What the guards buy (same experiment).** Priority+affinity and big-first did **not** measurably change the noise floor — run-to-run CV was ~1.0% / 1.1% / 0.95% for no-guards / +prio+affinity / +big-first (indistinguishable at n=10), plausibly because min-of-3 already discards contention-driven slow runs. Their real payoff is elsewhere: CPU-pinning lowered the median compute *level* ~**2.6%** (less core-migration/cache thrash) and big-first cut *wall-clock* ~**10%**. Keep them for a lower/faster/steadier floor — just don't expect them to tighten run-to-run variance.
- **Per-user `.jsonl` record structure** (one JSON object per line in `result/compute_parameters-*.jsonl` and its `profiling/measure/<label>.jsonl` snapshots; one line per user, re-sorted by `user`):
  ```json
  {"user": 1, "time_ms": 2833, "time_ms_runs": [2910, 2936, 2833],
   "parameters": [<36 floats>], "metrics": {"LogLoss": 0.462613}, "size": 19676}
  ```
  - `user` — user_id. `time_ms` — the official per-user time = **MIN** of `time_ms_runs` (the 3 reps); this is the only timing field the accept metric uses. `time_ms_runs` — the individual rep times (min-of-3 evidence).
  - `parameters` — the 36 trained FSRS-7 weights. `metrics.LogLoss` — per-user log loss from the frozen `evaluate()` (the band metric is the **unweighted mean** of these across users — `statistics.mean`).
  - `size` — the per-user **item count** (windowed prefixes; `sum(size)` over users == the asserted `total_items`, e.g. 1,897,936 @50 users / 60,360,449 @1000). ≈ reviews (N reviews → N−1 items/card), so **reviews/s ≈ `size / (time_ms/1000)`** — report the median of that per-user throughput alongside median ms/user.

## Constraints

1. No CUDA, no GPU.
2. At most **2 threads per user**. (Multiple users are already processed in parallel across processes; don't use more than 2 threads on a *single* user's data.)
3. Two correctness bars, by change type:
   - **(a) Math-unchanged changes** (constraint 6 — fusing, caching, reordering, etc.): if the change does **not** restructure the autodiff graph (forward-only fusions, dead-code removal, pure caching that doesn't move taped ops), per-user results should reproduce the champion **bit-for-bit**. Verify with **both** harnesses (diff the result `.jsonl`; sanity-check aggregates with `evaluate.py`) — each catches what the other can't:
     - **benchmark.py** records per-user **params + log loss** → check both. Reference: 50 users, **1,581,505 reviews**, log loss **0.313943** (benchmark.py now trains via `compute_parameters()`'s windowed O(N) path — the algorithm real users run — so this is the honest accuracy of what ships).
     - **compute_parameters.py** records per-user **params + log loss** (and time) → diff the params and log loss; its recorded time naturally varies run-to-run (see the Noise floor note), so ignore time for the bit-for-bit check. Reference: 50 users, **1,897,936 items**, log loss **0.3098** (iter-66 dual-trace champion; was 0.3085 pre-port). (Its params differ from benchmark.py's — different train/test split — so treat each harness as its own within-harness bit-for-bit reference.)

     **Reassociation caveat (ruling 2026-06-01):** changes that restructure the *backward* graph (hoisting/fusing/caching **taped** ops — e.g. pre-slicing loop-invariant weights) reorder floating-point gradient accumulation under burn's `Autodiff<NdArray<f32>>`, so they will **not** be bit-for-bit even though the math is identical — Adam then wanders over the 8 epochs to slightly different but equally-good params. These are still valid; judge them by the **(b) average-log-loss band** below, not bit-for-bit. (A *forward* identical to the champion is a good sanity check that the math really is unchanged.)
   - **(b) Precision-trading (float32/float16) OR graph-reordering changes**: log loss may drift, but the **AVERAGE log loss** — the single aggregate `evaluate.py` reports (benchmark.py's and compute_parameters.py's mean LogLoss), **NOT** the max per-user change — must stay within an **absolute ±0.0015 band anchored to the ORIGINAL (iter-0) baseline** (updated 2026-06-02: was ±0.0010 vs the *current* champion). Anchoring to the fixed original bounds the **cumulative** accuracy cost of all compounding precision-trades as a whole, rather than letting per-step ±0.0010 moves compound without limit. Concretely:
     - **compute_parameters.py mean LogLoss must stay in [0.3078, 0.3113]** (0.3098 - 0.0020 / + 0.0015; Andrew 2026-09-29 widened the LOWER end by 0.0005 when the champion sat 1.8e-5 above 0.3083, so a small accuracy GAIN no longer fails the band).
     - benchmark.py's mean LogLoss correspondingly in **[0.31244, 0.31544]** (0.313943 ± 0.0015).

     Individual users' params and per-user log loss may move more; only the *aggregate average* is gated. A regression or improvement inside that band is fine; landing outside means either the cumulative accuracy cost is now too high to justify, or something broke (a real bug usually moves the average far more than reassociation does — iter-1 hoist moved it ~1e-6, so a single step that jumps a large fraction of the band is still a bug-tell worth investigating even when it stays in-band).

     **Wall-clock-equalized accuracy check — REQUIRED for any inexact (non-bit-for-bit) change whose |Δ average log loss| ≥ 0.0001 (Andrew, 2026-06-02).** Staying in the band above is necessary but NOT sufficient to call a precision/reordering trade *good*. When such a change moves the average log loss by ≥0.0001 in either harness, also run the "equalize wall-clock" analysis: give the FASTER side the extra **epochs** its speedup buys so its wall-clock matches the slower side, then compare accuracy at that matched time. The trade is genuinely worthwhile only if, at equal wall-clock, the faster version is **at least as accurate** — Pareto-better on the accuracy-vs-wall-clock frontier, not just trading accuracy away for speed. **Why:** a user waiting on the optimizer cares about accuracy-*per-second*, not accuracy-*per-epoch* — a kernel slightly worse per epoch that converges further in the same wall-clock is strictly better for them. Worked example that prompted this rule (2026-06-02): iter17 O(N²)@8ep (0.310070 / 143.5 ms) vs iter18 O(N)@24ep (0.309620 / 117.5 ms) — O(N) was less accurate per epoch (+0.00084 at 8ep) but, given the 3× epochs its 3.12× speed bought, reached *lower* loss in *less* time, so the trade was real. (Cumulative form = the 800-epoch experiment: champion@~800ep beats iter0@8ep on both axes.) This is a DIAGNOSTIC: run it via compute_parameters (train==test → read as "optimization progress per second," not proven generalization) and REPORT it — it *characterizes* the trade, it does not replace the band or the 5% speed gate, and the epoch bump is only for this measurement (the shipped candidate still uses the fixed 8 epochs per constraint 4). Tiny reassociations (|Δ| < 0.0001, e.g. most fusions/hoists) skip it.
4. Don't change batch size, number of epochs, or any other SGD hyperparameter. No early stopping. Don't change `max_seq_len`.
5. Number of users must stay at **50**, and the dataset size must stay fixed — **benchmark.py: 1,581,505 reviews; compute_parameters.py: 1,897,936 items**. Assert this so you can't silently break review preprocessing.
6. Don't change FSRS-7's math or logic. Fusing operations, caching, float32/float16, etc. are all fair game **as long as the underlying math is identical**.
7. No CPU- or OS-specific tricks. Optimizations must work on any CPU/OS you can reasonably expect in 2026, **including smartphones**.
8. If Rust isn't low-level enough, custom SIMD / LLVM vectorization / other clever comp-sci is welcome — as long as it doesn't violate constraint 7 (stay portable; don't depend on AVX-512 etc.).
9. Any non-deterministic operation must be seeded; choose the seed once and never change it.
10. Feel free to build or import your own profiling tools and do profiling-only runs, but never modify the value written to the final .jsonl, and keep Python untimed. Also, profiling-only runs shouldn't be added to history.
11. **Mutation surface:** you may modify `compute_parameters()` and any func/struct it uses, plus FSRS-7 itself (as long as the math stays the same, per 3a). You may **not** modify `evaluate()` — the Rust scorer in `fsrs-rs/src/inference.rs`, invoked as `evaluate(items, |_| true)` by the binding (`fsrs_rs_python/src/lib.rs`) and as `predictor.evaluate(items)` in `compute_parameters.py`. It's the anti-cheating anchor.
    - Don't shift timed Rust work into untimed Python/preprocessing to beat the timer — the work has to actually disappear, not move off-clock. (For the same reason `complexity.py` counts both Python and Rust, so you can't move Rust into Python and call it a "simplification." `evaluate.py` and a few other diagnostics-only files are excluded from the complexity score.)
    - Optimizations must speed up **one user in isolation**. Anything that only pays off by batching 50 users (cross-user caching/amortization) is an artifact — a real Anki user optimizes one collection at a time.
12. **Acceptance — speed:** for each of the 50 users the per-user speedup is `s_u = t_champion_u / t_candidate_u` (each time = the min of its 3 runs). The single speed metric is **`speed_ratio` = the median of the 50 `s_u`** (the typical user's speedup). **Accept only if `speed_ratio ≥ 1.05`** (the median user is ≥5% faster). The median — not the mean of times — is used so a few slow/large collections can't carry the result and the ~1.5% warm-up drift can't fake a 5% median. Report the mean speedup too, but only as an informational "is one user dominating?" tell. (A paired Wilcoxon test was dropped: with per-user noise this low, the ~1-1.5% drift between two *identical* runs already drives p ≈ 0, so it fired on drift alone.)
13. **Acceptance — complexity:** the speedup must out-run added complexity — `speed_ratio ≥ complexity_ratio^2.5`, using the same median `speed_ratio` from constraint 12, with `complexity_ratio = c_candidate / c_champion` (>1 = more complex; see `complexity.py`). Examples: +1% complexity needs ≥ +2.52% speed; +5% needs ≥ +12.97%; +50% needs ≥ 2.76×. If complexity is flat or drops, only the 5% floor (constraint 12) binds. (Self-consistent under compounding: N changes each sitting on the limit keep cumulative speed vs. complexity on the same curve.)
Complexity score weights were chosen so that LOC contributes ~25% to the final score, AST node count and cyclomatic complexity contribute ~37.5% each. However, these proportions may have drifted after many iterations.

## Champion & compounding

The champion is the current fastest accepted version. Every candidate is measured against the **current** champion (not the original), so accepted speedups compound multiplicatively; on accept, the candidate becomes the new champion. If an idea just barely misses (e.g. +4.5% median when +5% is needed), it's your call whether to retry with a tweak or move on.

**Final validation:** before declaring victory / committing to GitHub, re-validate the final champion on **1000 users** (not just 50) to confirm the speedup generalizes across the review-count distribution. CPU-specific optimizations may be added as a bonus at that point, but they don't officially count for the loop.

## History bookkeeping

Keep a `.jsonl` log and a human-readable `.md`. Each entry records:
1. Iteration (0 = baseline, 1 = first change)
2. Timestamp
3. Median time (50 users) before the change
4. Median time (50 users) after
5. **Median speed ratio** (see constraint 12 above, it is not the ratio of medians, but the **median of ratios**)
6. Complexity score before
7. Complexity score after (see `complexity.py`)
8. **Complexity ratio**
9. True/False for "did all checks (log loss within bands, same number of reviews, same parameters (for changes that don't trade precision), etc.) pass?"
10. "accepted"/"rejected" status. "rejected" could mean either "improvement in speed wasn't sufficient" or "checks didn't pass"
11. A summary of the change **written before timing it** (≤15 words; one number = one word)
12. (`.jsonl` only) A private comment to your future self (e.g. notes to survive a compaction), but it must not be displayed in the .md file

**Progress plots (`plot_history.py`): two stacked views vs iteration.**
1. **Cumulative `speed_ratio`** (the headline) — the running product of the *accepted* iterations' median `speed_ratio` (item 5; rejects ×1). Read as "the median user is now X× faster than the iter-0 baseline." This *is* the accept metric (constraint 12) compounded, and it's **drift-immune**: each `speed_ratio` is a *within-session paired* ratio, so cross-session machine drift (~1%) cancels.
2. **Median per-user time (ms)** — intuitive, but **machine-specific and per-session, so it is NOT the accept/reject metric** (it's the ratio *of medians*, not the median *of ratios* that gates accepts, and it jitters with cross-session drift). Informational only; items 3–4 are its log evidence.
- **Caveat — the product is upward-biased:** you accept only when a noisy `speed_ratio` clears 1.05, so accepted ratios are selected high (winner's curse) and the ~1–2% paired noise compounds. **Anchor it** — every ~10–20 iters, re-measure the current champion vs the iter-0 baseline back-to-back in one session and plot that as a point: a direct, unbiased cumulative speedup whose gap to the product line is the accumulated bias. The final **1000-user** re-validation (see *Champion & compounding*) is the official total.
- **Regenerate the plot EVERY iteration.** Right after `log_history.py add`, run **`python plot_history.py`** and `git add result/history_plot.png result/history_plot_log.png` into that iteration's commit (linear and log-y versions, Andrew 2026-09-28). `README.md` embeds `result/history_plot.png`, so it must always show the latest iter. `plot_history.py` is Andrew-owned tooling — **run it, don't edit its code** unless he asks; if `git status` shows `plot_history.py` modified (his pending cosmetic edits), commit it alongside the plot so the committed script reproduces the committed plot.

**Compaction:** run `/compact` every 5 iterations, unconditionally (regardless of how many were accepted vs rejected) — over a 150–300-iteration campaign this keeps the context window fresh so per-iteration reasoning doesn't degrade. The history `.jsonl`/`.md` above is the durable record across compactions, so nothing important is lost.

## Notes

- **The expanding window is now O(N) (iter18 — DONE, the biggest single win, +212% / 3.12×).** A card with N reviews still becomes N−1 FSRSItems in the Python preprocessing (so the dataset row count is unchanged — constraint 5), but the Rust `compute_parameters()` training no longer re-runs the recurrence over every prefix (the old O(N²)). The prefix-items are tagged with their card id, regrouped per card, and each card runs ONE recurrence pass that reads off a loss at every timestep — the curve it computes for review t's stability update IS exactly the prediction R_t (same input state, same delta_t). The original author warned it would hurt log loss; in practice it landed inside the (then-widened) absolute ±0.0015 band. The `benchmark()` reference path deliberately keeps the O(N²) per-prefix forward as a bit-for-bit anchor.
- **`result/*(old)*.jsonl`** — files in `result/` with `(old)` in their names are kept on purpose as comparison baselines (e.g. the pre-dual-trace single-trace numbers). They are not stale junk; leave them in place.

## Where the time goes (profiling)

Profiling lives in `profiling/` — **excluded from `complexity.py`** (never counts against constraint 13) and never imported by the timed path; profiling-only per constraint 10 (doesn't touch `result/`).
- **`profiling/profile_structure.py`** (`python profiling/profile_structure.py`): reproduces the 50-user preprocessing and regresses their *already-recorded* times against structural work proxies; caches `profiling/structure.json`.
- **`profiling/_profile_user.py`** (`python profiling/_profile_user.py <user_id> <reps>`): loads one user and loops `compute_parameters()` so a sampler can attach.

The training loop and optimizer are fixed, so these hold across FSRS-7 variants (different hyperparams / a heavier forgetting curve only *strengthen* the per-timestep pathways below):
- **Cost model (the O(N) window changed the slope, not the shape):** time ≈ linear in **sequence-work** + a fixed per-user floor (~0.18 s) that dominates tiny collections. The OLD O(N²) work was Σ over *prefix-items* of prefix-length; the O(N) window (iter18) cut it to **Σ over *cards* of full-sequence-length** — one recurrence pass per card — so the per-user floor (setup/penalty/opt) is now a bigger share (~⅓ of the largest users' time after iter19 built the batches once). `n_cards`/`n_items` are weak predictors; summed sequence length is what counts. Per-step cost still dominates per card — **shrinking per-step work is the main lever**, the floor the secondary one.
- **Hot path (now):** the hand-written SIMD forward+gradient in `fsrs-rs/src/analytic.rs` — `card_loss_and_grad_simd` (training gradient) + `card_loss_simd` (per-epoch validation): a **sequential per-timestep recurrence, 8 cards/lane (`wide::f32x8`)**, run for the fixed 8 epochs. Burn `Autodiff` is **gone** from the timed path (replaced first by the analytic gradient, then by the O(N) window). Grad+valid ≈ **78%** of the median user. Backend is **f32**; transcendentals are minimax `exp8`/`ln8`. "Use f32" and "remove the tape" are spent levers — remaining knobs are per-step op count (math-preserving hoists / 3b reassociation) and f16 (risky per 3b).
- **Native profiling caveat:** py-spy `--native` returns 0 samples on the stripped release `.pyd` (no `.pdb`), and the work runs entirely inside the native Rust call, so the non-native sampler sees nothing either. For function-level Rust hotspots, rebuild with symbols (`fsrs_rs_python/`: add `[profile.release] debug = true` to `Cargo.toml`, then `maturin develop --release`) and sample via `_profile_user.py`; otherwise just read the (small) Rust source.

Candidate pathways — most of the original best-first list is now **DONE** (all subject to the correctness bars):
1. ~~Hoist loop-invariant weight ops + kill per-step allocations~~ **DONE** (iter8–10: `wconsts` + cached lns lift the weight-only transcendentals out of the per-timestep loop; bit-for-bit 3a).
2. ~~Analytic / dual-number gradient for the main loss~~ **DONE** (burn Autodiff for the BCE loss replaced by the hand-written `analytic.rs` gradient — the biggest structural win before the window; 3b band).
3. ~~O(N²)→O(N) expanding window~~ **DONE** (iter18, +212%; see the Notes bullet). The author's "it hurts log loss" warning held only weakly — it landed inside the ±0.0015 band.
4. **Trim per-iteration / per-epoch overhead** (still partly open). iter19 built the host batches ONCE across epochs (+14.5%). Remaining floor work: dedupe the per-batch card regrouping in `build_batch_host_windowed`, plus other one-time setup. Helps small collections / the long tail.

**Remaining frontier — the "strategic wall":** the SIMD windowed forward (`card_*_simd`, ≈78% of the median user) is already O(N), `f32x8` (the portable max; ARM NEON is 128-bit so f32x8 already = 2×), and minimax-transcendental. The safe levers are nearly spent. What's left: per-step op-count cuts (math-preserving hoists / 3b reassociation), lower-degree `exp8`/`ln8` minimax (band headroom only ~0.0004 — risky), or an f16 forward (bold, ~2× potential, high band + portability risk).
