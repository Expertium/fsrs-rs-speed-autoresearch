# Technical details

Background for the [README](README.md): how the numbers are measured, why the progress plot is slightly optimistic, and where the speedup came from. The authoritative spec for the campaign is [`CLAUDE.md`](CLAUDE.md).

## Measurement

### Paired A/B timing (official from iter 34)

`profiling/measure_paired.py <champion.pyd> <candidate.pyd> [out.json]` runs the champion and the candidate **on the same user at the same time**, on neighbouring CPU blocks, so background load that changes over time (other programs, thermal drift) hits both binaries equally and cancels in the ratio. Each user runs 3 times on both binaries; the per-user time is the min of the 3 (timing noise is one-sided — interference only slows a run). Between reps the binaries swap CPU blocks and start order (a fixed start order leaned ~0.3% toward the first). The next user starts only when both binaries finish, so the faster binary never leaves the slower one a quieter machine.

- **Validation:** with a ~7-core job loading the machine, A/A runs (champion vs itself, ×10) landed within 0.9952–1.0075 (worst 0.75%); the old back-to-back protocol showed ~±5% under the same load.
- **Load:** 1 pair slot by default = 2 worker processes × 2 logical CPUs = 4 threads (`FSRS_PAIRED_SLOTS=5` restores the original 10-worker load). A split per-user cache (`profiling/devbench/split_raw.py`) keeps the 1000-user run at ~1.5 GB instead of ~6 GB per worker.
- **Timed region:** the Rust `compute_parameters()` call only, fed through the bit-for-bit `compute_parameters_raw` twin from cached arrays. Each user's LogLoss under the frozen `evaluate()` is computed afterwards, untimed.

**One user alone:** every accepted speedup must also be a gain for one user optimized alone (`profiling/devbench/bench_min.py`, one process on one core, min of K reps). Wins that only appear when several users run in parallel — less contention, shared caches, cross-user amortization — do not count.

### Low-noise setup (Windows)

Workers run at HIGH priority, off logical CPUs 0–3 (where the OS and background apps tend to land), each pinned to its own pair of cores. The CPU clock is pinned by capping the max and min processor state at 99%, which disables turbo and holds the clock flat:

```bat
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 99
powercfg /setactive SCHEME_CURRENT
```

Measured effect (3×10 identical-build runs): pinning lowered the median compute level ~2.6% and "largest users first" cut wall-clock ~10%, but neither changed run-to-run noise (~1% CV). To reduce noise further you could close other programs or lock fan speed and voltage in the BIOS; I didn't, because I still use the PC for other things.

### Accuracy band

The 50-user mean LogLoss must stay in **[0.3078, 0.3113]** — the original 0.3098 with −0.0020/+0.0015. The band is anchored to the original baseline (not to the current champion), so it bounds the *cumulative* cost of all precision trades together. The lower end was widened by 0.0005 on 2026-09-29: the champion sat 1.8e-5 above the old edge (0.3083), so even a small accuracy *gain* would have failed the check. Any inexact change that moves the mean LogLoss by ≥1e-4 also gets an **equal-wall-clock** check: the faster side gets the extra epochs its speedup buys, and the trade counts as good only if it is at least as accurate at equal time.

## Why the cumulative number is slightly optimistic (winner's curse)

The top panel of the progress plot is a **product of measured ratios**, and that product is biased a little high. Two effects compound:

1. **Selection bias.** A candidate is kept only if its *measured* median speedup clears 1.05. Each measurement carries some noise, so the bar preferentially admits iterations whose noise landed favourably: a change whose true speedup is 1.045 but measured 1.055 is recorded at 1.055, while the unlucky mirror case is rejected and never counted.
2. **Multiplicative compounding.** The line multiplies dozens of noisy ratios, so the small upward biases multiply too.

The fix is to **re-anchor**: measure the current champion directly against an old baseline in one paired session. That single ratio has no accept/reject filter, so it is unbiased. At iter 18 the product read ×65.5 against a direct ×62.0 (~5% high). In round 2 the product from iter 27 to iter 58 is ×3.70 against a direct paired ×3.38 on 50 users. (Early in round 2 the 10-worker load understated some wins, so the gap is not purely winner's curse.) The final word is the 1000-user validation.

## Round 1 in detail

### 1000-user validation (end of round 1)

| build | median ms/user | median reviews/s | median speedup vs iter-0 |
| --- | --- | --- | --- |
| iter-0 baseline (x86) | 3659 | 7.4k | 1× |
| round-1 champion (x86, default build) | 36.0 | 726k | ×98.2 |

The champion's mean LogLoss was +0.0014 vs iter-0 on this set, about one band-width. Per-user records: `result/{iter0-1000u-baseline,champ-1000u,avx2-1000u}.jsonl`.

### The reverted precision trade

One accepted change (a cruder polynomial for `exp`/`ln` on the training path) was **undone** after the campaign: it cost ~0.0004 LogLoss for what re-measured as only ~5–6% speed, because a later change had already removed most of the work it targeted. So the round-1 champion is ×98.2 at mean LogLoss 0.3110 (50 users) rather than the plotted ×103.8 / 0.3112. The plot keeps the change as the historical record.

### Buying accuracy back with more epochs

The round-1 champion ran 8 epochs. Its ~0.0012 LogLoss cost vs the original is fully recoverable with more epochs, and the optimizer is still much faster:

| epochs | mean LogLoss (50u) | speedup vs iter-0 |
| --- | --- | --- |
| **8** (shipped then) | 0.3110 | ~×98 |
| 20 | 0.3098 (= original) | ~×43 |

(Diagnostic via `profiling/epoch_sweep.py`; the shipped default was not changed.)

### Finished-model port and dropped per-epoch validation

The optimizer was then ported to the finished FSRS-7 model, and its per-epoch validation pass (for best-epoch selection) was replaced by a fixed 9-epoch schedule. On 1000 users that was a **net ×1.37** (median 34.0 → 24.0 ms/user) and slightly *more* accurate (mean LogLoss 0.31766 → 0.31744). The validation pass alone cost ×1.53 at fixed epochs, with bit-for-bit identical weights. This factor spans two model versions, so the overall number composes it with round 1 rather than measuring one chain.

### How much of round 1 was "free" vs a precision trade?

Splitting the 20 accepted round-1 iterations by whether they were bit-for-bit and multiplying each group's ratios:

| kind of change | cumulative factor | share of the log speedup |
| --- | --- | --- |
| bit-for-bit (exactly accuracy-preserving) | ×3.25 | ~25% |
| precision / reassociation trades | ×34.6 | ~75% |
| total (product of all accepted) | ×112.7 | 100% |

Shares are of the **log** speedup, the only split that adds up (speedups compound multiplicatively). The three giants were all reassociations: the analytic gradient replacing autodiff (×2.17), SIMD across 8 cards (×2.50) and the O(N) expanding window (×3.12) — each changes the order floats are summed in, so none can be bit-for-bit. Caveats: the ratios are path-dependent, so this is a decomposition of the logged product, not a forecast; and two "bit-for-bit" entries (SIMD/analytic *validation*) were precision changes that happened not to flip the best-epoch pick — counting only strictly math-unchanged work gives ~×2.25.

Round 2, by contrast, was mostly exact: 11 of its 16 accepted iterations reproduced the champion bit-for-bit. Iters 40–43 (for example the one-exponential sigmoid form and base-2 logs in the recurrence) and iter 58 (card order within a length, and the order of the gradient lane sums) are reassociations; iter 56 changed one user's weights because its recency power is correctly rounded where the C library's `powf` is off by one ulp.

## Portability

All numbers are from one x86 machine (Ryzen 9 5950X). Most of the speedup is portable; part is not:

- **AVX2 is a further x86 bonus.** Rebuilding with `RUSTFLAGS="-C target-cpu=native"` turns each 8-wide SIMD op from two 128-bit SSE2 ops into one 256-bit AVX2 op — ~1.9× more at the end of round 1 (~×188). It cannot count: the binary is built for one machine, and shipping it would need runtime CPU dispatch. The round-2 kernels are also dispatch-bound partly because SSE2's two-operand form forces register copies that AVX's three-operand form would remove.
- **ARM / Apple Silicon.** The explicit SIMD did not speed up an Apple M-chip when the fsrs-rs maintainers tried it: `wide` compiles to real NEON there, but Apple's very wide out-of-order cores already extract the cross-card parallelism from scalar code. The algorithmic core — analytic gradient, O(N) window, host-side Adam, build-once batching — transfers everywhere, and is what went upstream in [PR #411](https://github.com/open-spaced-repetition/fsrs-rs/pull/411) and [PR #419](https://github.com/open-spaced-repetition/fsrs-rs/pull/419).

## `benchmark.py` options

`uv run benchmark.py --help` lists all flags. The common ones:

| Flag | Description | Default |
| --- | --- | --- |
| `--processes` | Number of worker processes. | `8` |
| `--data` | Path to `revlogs/*.parquet`. | `../anki-revlogs-10k` |
| `--max-user-id` | Maximum user ID to process (inclusive). | unset |
| `--default` | Evaluate default parameters without per-user training. | off |
| `--n_splits` | Number of TimeSeriesSplit folds. | `5` |
| `--short` | Include short-term reviews. | off |
| `--secs` | Use `elapsed_seconds` as the interval instead of days. | off |
| `--recency` | Recency-weight the training items. | off |
| `--raw` | Save raw per-review predictions to `raw/<name>.jsonl`. | off |
| `--file` | Save per-user evaluation results to `evaluation/<name>/`. | off |
