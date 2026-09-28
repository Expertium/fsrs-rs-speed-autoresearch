# fsrs-rs-speed-autoresearch

An **autoresearch loop** that made [FSRS-rs](https://github.com/open-spaced-repetition/fsrs-rs) parameter optimization **over 100x faster** (its portable core is being upstreamed into FSRS-rs itself — [PR #411](https://github.com/open-spaced-repetition/fsrs-rs/pull/411) and [#419](https://github.com/open-spaced-repetition/fsrs-rs/pull/419)) — so real Anki users spend less time staring at the optimizer's progress bar and more time doing reviews — **while only making it mildly less accurate**. An AI agent (Claude) proposes a change, measures it under a strict protocol, and keeps it only if it clears both the speed and the correctness bars. Inspired by AlphaEvolve and [Andrej Karpathy's "autoresearch" repo](https://github.com/karpathy/autoresearch). Also check out [my other autoresearch repo](https://github.com/Expertium/fsrs-autoresearch).

[![Campaign progress: cumulative median speedup (top) and median per-user optimizer time (bottom) vs iteration](result/history_plot.png)](result/history_plot.png)

[![The same two panels with logarithmic y axes](result/history_plot_log.png)](result/history_plot_log.png)

*Each point is an accepted change: the agent measures a candidate against the current champion and keeps it only if the typical (median) user's optimization gets faster with minor loss of accuracy. The top panel is the compounding median speedup vs the original baseline; see [**Tracking progress**](#tracking-progress) for how to read it.*

> **🤖 Working on this repo (human or AI)? Read [`CLAUDE.md`](CLAUDE.md) first.**
> It is the authoritative spec — the goal, the exact measurement protocol, the hard constraints and acceptance bars, an annotated map of the code, and current profiling findings. This README is just the human-facing quickstart.

## The idea

The function being optimized is the Rust `compute_parameters()` in [`fsrs-rs/`](fsrs-rs) (exposed to Python via [`fsrs_rs_python/`](fsrs_rs_python)) — the same routine Anki runs when a user optimizes their FSRS parameters. Two harnesses drive it:

| Harness | Role |
| --- | --- |
| **`compute_parameters.py`** | The **speed** harness the campaign optimizes: times `compute_parameters()` per user (min of 3 runs) and scores it with Rust `evaluate()`, train set == test set. |
| **`benchmark.py`** | The **reference** harness (5-fold TimeSeriesSplit + Python forgetting curve): it trains via the same windowed `compute_parameters()` real users run, so it reports the honest accuracy of the shipped optimizer. |

A candidate is **accepted only if** the median user gets **≥5% faster** *and* accuracy stays within the correctness bars (identical results for math-preserving changes; for precision-trading ones the aggregate log loss must stay within an absolute **±0.0015 of the original baseline** — i.e. `compute_parameters.py` mean LogLoss in **[0.3083, 0.3113]**), and the speedup out-runs any added code complexity (`complexity.py`). The precise definitions live in [`CLAUDE.md`](CLAUDE.md).

## Setup

1. **Dataset** — [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k). Put it next to this repo at `../anki-revlogs-10k` (or pass `--data <path>`); the loaders read `revlogs/user_id=*`.
2. **Dependencies** — [uv](https://docs.astral.sh/uv/) manages the Python env and builds the Rust `fsrs_rs_python` extension:
   ```bash
   uv sync
   ```

## Running

**Speed harness** — the canonical 50-user timing run the campaign uses:
```bash
uv run compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50
```
Per-user times (ms) are written to `result/compute_parameters-FSRS-rs-short-secs-recency.jsonl`. For the full measurement protocol (min-of-3, median-of-ratios accept metric, noise floor), see *Measurement protocol* in [`CLAUDE.md`](CLAUDE.md).

#### Low-noise measurement (Windows)

Timing noise muddies the ≥5% accept bar, so the campaign runs the speed harness at high priority pinned to specific cores. On Windows:

```bat
start /high /affinity 0xFFFFFFF0 python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50
```

- `/high` raises the process priority; `/affinity 0xFFFFFFF0` keeps the workers **off logical CPUs 0–3** (the four cleared low bits of the mask), where the OS and background apps tend to land. On top of that, `compute_parameters.py` pins each worker to its own pair of cores, so cross-worker contention stays constant instead of random.

And pin the CPU clock so turbo/thermal drift can't change the timings between runs (cap the max and min processor state at 99%, which disables turbo and holds the clock flat):

```bat
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 99
powercfg /setactive SCHEME_CURRENT
```

If you want to reduce the noise further, you can:

1. Turn off as many programs as you can. I didn't do that because I still want to use my PC for other things.
2. Lock CPU fan speed and CPU voltage in BIOS. I just didn't want to bother.

**Reference / correctness harness:**
```bash
uv run benchmark.py                  # basic run
uv run benchmark.py --default        # evaluate default params (no per-user training)
uv run benchmark.py --max-user-id 100
uv run benchmark.py --processes 4
```

### Common `benchmark.py` options

Run `uv run benchmark.py --help` for the full list.

| Flag | Description | Default |
| --- | --- | --- |
| `--processes` | Number of worker processes. | `8` |
| `--data` | Path to `revlogs/*.parquet`. | `../anki-revlogs-10k` |
| `--max-user-id` | Maximum user ID to process (inclusive). No limit if unset. | unset |
| `--default` | Evaluate default parameters without per-user training. | off |
| `--n_splits` | Number of TimeSeriesSplit folds. | `5` |
| `--short` | Include short-term reviews. | off |
| `--secs` | Use `elapsed_seconds` as the interval instead of days. | off |
| `--recency` | Recency-weight the training items. | off |
| `--raw` | Save raw per-review predictions to `raw/<name>.jsonl`. | off |
| `--file` | Save per-user evaluation results to `evaluation/<name>/`. | off |

## Tracking progress

`plot_history.py` reads `result/history.jsonl` and renders two stacked views vs iteration to `result/history_plot.png` (linear y axes) and `result/history_plot_log.png` (the same panels with logarithmic y axes):

```bash
uv run plot_history.py
```

The [plot at the top of this README](#fsrs-rs-speed-autoresearch) shows two stacked panels:

1. **Cumulative speedup** — the running product of the accepted iterations' median `speed_ratio`. This is the accept metric (the median of per-user speed ratios) compounded, so it's what "progress" means here, and it's drift-immune. Only the **top-5 biggest wins** are labelled with their summary, to keep the panel readable.
2. **Median per-user time (ms)** — intuitive, but **machine-specific and measured per session, so it is *not* the metric used to accept or reject candidates** (each candidate is judged by its median per-user speed ratio, re-measured against the champion in the same session). Treat the time curve as informational context only.

### Why the cumulative number is slightly optimistic (winner's curse)

The top-panel cumulative speedup is a **product of measured ratios**, and that product is **biased a little high** — it overstates the true speedup. Two effects compound:

1. **Selection bias (the "winner's curse").** A candidate is kept only if its *measured* median speedup clears the bar (≥ 1.05). But each measurement carries ~1% noise, so the bar acts as a filter that preferentially admits iterations whose noise happened to land *favorably*. A change whose true speedup is 1.045 but measured 1.055 gets accepted and recorded at 1.055; the symmetric unlucky case (true 1.055, measured 1.045) gets rejected and never counted. So the accepted ratios skew high — you're looking at the winners, and winners are lucky on average.
2. **Noise compounds multiplicatively.** Each `speed_ratio` is a ratio of two noisy timings (~1% each). The cumulative line multiplies ~15 of these together, so the small upward biases multiply too, and the gap grows as the campaign gets longer.

**The honest fix is to re-anchor.** Every ~10–20 iterations we measure the *current champion directly against the original iter-0 baseline*, back-to-back in one session. That single ratio is **unbiased** — there's no accept/reject filter applied to it, so no selection creeps in — and it's drift-immune (same session). The distance between the product line and that anchor point is exactly the accumulated inflation. (For example: at iter 18 the product line read ×65.5, but the direct iter-0 anchor was ×62.0 — about 5–6% optimistic. The anchor is the number to trust; the final report re-validates the champion on 1000 users.)

### Final validation — 1000 users

The campaign tunes on 50 users, so the last step re-runs the **frozen champion against the original iter-0 baseline on 1000 users** (a 20× larger, representative sample — `--max-user-id 1000`, 60.4M review-items) to confirm the speedup is real and generalizes, not a 50-user artifact:

| build | median ms/user | median reviews/s | median speedup vs iter-0 |
| --- | --- | --- | --- |
| iter-0 baseline (x86) | 3659 | 7.4k | 1× |
| **champion** (x86, default build) | **36.0** | **726k** | **×98.2** |

The default-build champion lands at **×98.2** on 1000 users — consistent with the 50-user measurements, so the speedup holds across the full review-count distribution. Throughput goes from ~7k to ~726k reviews/s. (Accuracy scales gracefully too: the champion's mean log loss is +0.0014 vs iter-0 on this set, about one band-width — the accumulated precision trades don't blow up at scale.) Rebuilding with AVX2 (`RUSTFLAGS="-C target-cpu=native"`) roughly doubles this again on x86 — the pre-revert AVX2 champion measured ×198.3 (18 ms/user); the reverted champion wasn't separately AVX2-measured but scales similarly (~×188). Per-user records: `result/{iter0-1000u-baseline,champ-1000u,avx2-1000u}.jsonl`.

> **Note — one precision trade was reverted after the campaign.** On a closer cost/benefit review, one accepted change (a cruder polynomial approximation of `exp`/`ln` on the training path) was **undone**: it had cost ~0.0004 log loss for what re-measured as only ~5–6% speed — a bad trade once a *later* change had already removed most of the work it targeted. So the shipped champion is **×98.2** at mean log loss **0.3110** (50-user), rather than the campaign's plotted ×103.8 / 0.3112. The progress plot above is left as the **historical record of the campaign** (the revert is not plotted as a step).
>
> **Buying the accuracy back with more epochs.** The champion runs **8 optimizer epochs** by default (the fast setting). The speedup cost ~0.0012 log loss vs the original (0.3110 vs 0.3098). That's fully recoverable by spending more epochs — and even fully recovered, the optimizer is still ~40× faster than the original:
>
> | epochs | mean log loss (50u) | speedup vs iter-0 |
> | --- | --- | --- |
> | **8** (shipped) | 0.3110 | **~×98** |
> | 20 | 0.3098 *(= original)* | ~×43 |
>
> At 20 epochs the loss returns to the original 8-epoch baseline (0.3098), at ~2.26× the 8-epoch time — so you can have the *original accuracy and* a ~43× speedup, or the shipped config for ~98× at +0.0012 log loss. (Diagnostic via `profiling/epoch_sweep.py`; the shipped default stays 8 epochs.)

> **Update — finished-model port + dropped per-epoch validation (now over 100×).** Everything above is the *pre-release* FSRS-7 model (8 epochs, *with* a per-epoch validation pass for best-epoch selection). The shipped optimizer has since been ported to the **finished FSRS-7 model**, and its **per-epoch validation pass was removed** — best-epoch selection replaced by a fixed 9-epoch schedule. Measured on 1000 users (8-epoch + validation → 9-epoch, no validation): a **net ×1.37 speedup** (median per-user, 34.0 → 24.0 ms), and *slightly more* accurate too (mean log loss 0.31766 → 0.31744) — a Pareto win, not a trade. (The validation pass alone costs **×1.53** at fixed epochs, with bit-for-bit-identical weights.) Composed with the campaign's ×98.2, the optimizer is now **comfortably over 100× faster** than the original. Note this factor spans two model versions — the ×98.2 is the pre-release model (with validation), the ×1.37 is the finished model — so it is an honest composition, not a single re-anchored chain.

**Caveat — these numbers are all x86 (Ryzen 5950X), and the speedup is _not_ architecture-uniform.** Most of it is portable; a chunk is x86-specific:

- **AVX2 — a further x86 bonus.** Rebuilding with `RUSTFLAGS="-C target-cpu=native"` turns each 8-wide SIMD op from 2×128-bit (SSE2 baseline) into one native 256-bit (AVX2) instruction — ~1.9× more, so a typical x86 desktop/laptop (≈2015+) sees ~×188. It can't count officially because AVX2 is x86-only (constraint 7 requires portability incl. smartphones); shipping it would need runtime CPU dispatch (e.g. the [`multiversion`](https://crates.io/crates/multiversion) crate), since a `target-cpu=native` binary is built for one machine and can't be distributed.
- **ARM / Apple Silicon — closer to ~×30, not ×98.** The explicit SIMD *vectorization* (≈×3.3 of the ×98 — `wide::f32×8` batching 8 cards/lane) **did not replicate on an Apple M-chip** when the fsrs-rs maintainers tried it. `wide` does compile to real NEON there, but Apple Silicon's very wide out-of-order cores + auto-vectorization already extract the cross-card parallelism from *scalar* code, leaving little for explicit SIMD to add. What *does* transfer is the **algorithmic core** — the analytic gradient (replacing autodiff), the O(N) expanding window, host-side Adam, build-once batching — none of which depends on SIMD. So an ARM user realistically gets ≈ ×98 ÷ the ×3.3 SIMD multiplier ≈ **×30** (more if the SIMD partly helps on narrower ARM cores than Apple's).

That architecture-independent core is exactly what got **upstreamed**: [fsrs-rs PR #411](https://github.com/open-spaced-repetition/fsrs-rs/pull/411) (merged) adopted the analytic gradient, host Adam, and `card_ids` window — crediting this repo — and left the explicit SIMD out precisely because it didn't replicate off x86. A follow-up, [PR #419](https://github.com/open-spaced-repetition/fsrs-rs/pull/419) (merged), then added more of this repo's scalar, architecture-independent speedups to upstream's windowed `compute_parameters` — hoisting loop-invariant work, skipping dead padding and inactive branches, and reusing batches and scratch buffers.

### How much of the speedup was "free" vs. a precision trade?

The correctness bars allow two kinds of accepted change (see [`CLAUDE.md`](CLAUDE.md) constraint 3): **bit-for-bit** ones that leave every trained weight (and the log loss) *byte-identical*, and **precision/reassociation trades** that reorder float accumulation or approximate a transcendental, nudging the log loss but staying inside the accuracy band. Splitting the 20 accepted Phase-1 iterations by which checks they recorded (`0/50 params, ΔLogLoss = 0` vs. a drift) and multiplying out each group's `speed_ratio`s:

| kind of change | cumulative factor | share of the speedup (log) |
| --- | --- | --- |
| **bit-for-bit** (exactly accuracy-preserving) | **×3.25** | ~25% |
| **precision / reassociation trades** | **×34.6** | ~75% |
| total (product of all accepted) | ×112.7 | 100% |

The shares are of the **log** speedup — the only split that adds up to 100%, because speedups compound *multiplicatively* (×3.25 × ×34.6 = ×112.7, and log 3.25 + log 34.6 = log 112.7). A linear "share of ×112.7" wouldn't sum to 100% (the ×3.25 bit-for-bit factor is only ~3% of ×112.7 on its own, but you can't linearly attribute a product).

So the bulk of the speedup came from **trading a little floating-point precision** — the precision factor (×34.6) is more than 10× the bit-for-bit one (×3.25). The three giants are all precision trades: replacing the autodiff tape with a hand-written analytic gradient (**×2.17**), SIMD-vectorizing that gradient with `f32×8` (**×2.50**), and the O(N) expanding window (**×3.12**) — each reorders the order floats are summed in, so none can be bit-for-bit. The "free" ×3.25 is the exact restructuring: building the host batches directly and once, hoisting loop-invariant work, sharing repeated `ln`s, a hand-rolled Adam, skipping a dead final-timestep update.

Two caveats. (1) This is an exact *decomposition* of the logged product (3.25 × 34.6 = 112.7), **not** a forecast — the ratios are path-dependent (each measured against the then-current champion), so a bit-for-bit-*only* campaign would likely land somewhat **below** ×3.25, because some "free" wins were amplified by the precision wins that came before them. (2) Two of the bit-for-bit entries (SIMD/analytic *validation*) are really precision changes that came out byte-identical only because validation merely picks the best epoch and the approximation never flipped that pick; counting only *strictly* math-unchanged work drops the free factor to ~×2.25.

## Repo tour

`compute_parameters.py` (speed harness) · `benchmark.py` (reference) · `fsrs-rs/src/{model,training,inference}.rs` (the Rust FSRS crate — the optimization target) · `fsrs_rs_python/` (PyO3 binding) · `features/` (review preprocessing) · `complexity.py` (the complexity score) · `profiling/` (profiling tools). The full annotated map is the *Code layout* section of [`CLAUDE.md`](CLAUDE.md).
