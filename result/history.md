# FSRS-rs speed autoresearch — iteration history

Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND **speed_ratio ≥ complexity_ratio^2.5** (constraint 13). speed_ratio is the *median of per-user ratios*, measured back-to-back vs the then-current champion. Times are machine/session-specific (informational).

| iter | time_before (ms) | time_after (ms) | speed_ratio | cplx_before | cplx_after | cplx_ratio | cplx^2.5 | checks | status | summary |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3067 | 3067 | 1.000 | 4771 | 4771 | 1.000 | 1.000 | ✓ | accepted | baseline: dual-trace FSRS-7 Rust port (iter-66 champion) |
| 1 | 2916 | 2884 | 1.035 | 4771 | 4703 | 0.986 | 0.965 | ✗ | rejected | hoist loop-invariant weight slices; remove single-version dispatch |
| 2 | 3012 | 1368 | 2.166 | 4771 | 5620 | 1.178 | 1.506 | ✓ | accepted | hand-written analytic BCE gradient replaces burn autodiff forward+backward in training |
| 3 | 1370 | 1274 | 1.077 | 5620 | 5605 | 0.997 | 0.993 | ✓ | accepted | parallelize analytic gradient across 2 threads; remove dead autodiff grad helpers |
| 4 | 1292 | 1295 | 0.990 | 5605 | 5605 | 1.000 | 1.000 | ✓ | rejected | validation pass via threaded manual forward instead of burn autodiff forward |
| 5 | 1276 | 1064 | 1.198 | 5605 | 5600 | 0.999 | 0.998 | ✓ | accepted | replace backward powf with division by cached forward values (b^(e-1)=b^e/b) |
| 6 | 1065 | 963 | 1.107 | 5600 | 5601 | 1.000 | 1.000 | ✓ | accepted | score validation each epoch with analytic forward over pre-extracted batches |
| 7 | 962 | 1025 | 0.944 | 5601 | 5635 | 1.006 | 1.015 | ✓ | rejected | thread the analytic validation forward across 2 threads (THREAD_MIN split) |
| 8 | 968 | 855 | 1.121 | 5602 | 5621 | 1.003 | 1.008 | ✓ | accepted | rewrite 12 forward powf as exp(e*ln), cache the ln, reuse in backward |
| 9 | 856 | 793 | 1.078 | 5621 | 5628 | 1.001 | 1.003 | ✓ | accepted | pre-extract training batches once; replicate dataloader shuffle to drive epoch order |
| 10 | 792 | 738 | 1.074 | 5628 | 5651 | 1.004 | 1.010 | ✓ | accepted | hoist loop-invariant weight constants (ln w27/w28, aa, exp 3w5) out of per-timestep loop |
| 11 | 746 | 716 | 1.039 | 5651 | 5633 | 0.997 | 0.992 | ✓ | rejected | drop dead train/valid tensor datasets + remove dummy autodiff backward (bit-for-bit, -complexity) |
| 12 | 738 | 580 | 1.292 | 5651 | 5515 | 0.976 | 0.941 | ✓ | accepted | build train/valid host batches directly (skip burn-tensor floor); remove dummy backward + dead dataloader code |
| 13 | 577 | 538 | 1.075 | 5515 | 5522 | 1.001 | 1.003 | ✓ | accepted | share ln(last_s/last_sf/last_d) across curve + both stability traces; 3 fewer ln per timestep (bit-for-bit) |
| 14 | 534 | 414 | 1.305 | 5522 | 5884 | 1.066 | 1.172 | ✓ | accepted | vectorize the validation forward with wide::f32x8 (8 cards/lane, exp8/ln8); bit-for-bit epoch selection |
| 15 | 410 | 168 | 2.500 | 5884 | 6458 | 1.098 | 1.262 | ✓ | accepted | vectorize analytic gradient forward+backward with wide::f32x8 (8 cards/lane); pad batches to multiple of 8 |
| 16 | 170 | 268 | 0.623 | 6458 | 6458 | 1.000 | 1.000 | ✓ | rejected | thread SIMD gradient across 2 cores (split card-groups) - REJECTED, vector units SMT-saturated |
| 17 | 172 | 145 | 1.156 | 6458 | 6460 | 1.000 | 1.001 | ✓ | accepted | Minimax (Remez) exp8 deg 6->4 and ln8 deg 4->2: 4 fewer FMAs |
| 18 | 145 | 46 | 3.118 | 6460 | 7207 | 1.116 | 1.315 | ✓ | accepted | O(N) expanding window: one per-card pass scores every timestep, replacing O(N^2) per-prefix forward |
| 19 | 47 | 40 | 1.145 | 7207 | 7204 | 1.000 | 0.999 | ✓ | accepted | Build host batches once (train==test): reuse for grad+validation, drop redundant clone+build |
| 20 | 40 | 39 | 1.059 | 7204 | 7196 | 0.999 | 0.997 | ✓ | accepted | Unified BCE -ln(1-|label-r|) (Andrew): vectorize validation ln8 + one-division gradient seed |
| 21 | 40 | 36 | 1.083 | 7196 | 7187 | 0.999 | 0.997 | ✓ | accepted | dedup card-regroup (bit-for-bit) + exp8 deg-3 minimax (1 fewer FMA) |
| 22 | 36 | 34 | 1.077 | 7187 | 7180 | 0.999 | 0.998 | ✓ | accepted | hand-roll Adam + param clip on host f32, drop per-step burn tensor round-trips |
| 23 | 33 | 30 | 1.096 | 7180 | 7219 | 1.005 | 1.014 | ✓ | accepted | Windowed-only cruder minimax via const-generic FAST: exp8 deg3->deg2 + ln8 deg2->deg1; benchmark path bit-for-bit |
| 24 | 30 | 28 | 1.053 | 7219 | 7231 | 1.002 | 1.004 | ✓ | accepted | Fuse 3 exp-pairs/timestep (curve se, stab pr x2) + skip dead last-timestep state-update in validation forward |
| 25 | 28 | 28 | 1.054 | 7231 | 7257 | 1.004 | 1.009 | ✓ | accepted | Gradient skip-last (curve-only at final timestep) + cached sort-key + merged longest-prefix scan; bit-for-bit |
| 26 | 27 | 26 | 1.041 | 7257 | 7285 | 1.004 | 1.010 | ✓ | rejected | Per-group padding skip: each group runs only to its longest card, not the batch-wide seq_len |
| 27 | 24 | 29 | 0.852 | 10502 | 10516 | 1.001 | 1.003 | ✓ | rejected | Branchless step: compute both init and update paths, mask-select the first review |
| 28 | 22 | 23 | 1.096 | 10502 | 10568 | 1.006 | 1.016 | ✓ | accepted | Windowed batches built from item headers: no normalize pass, regrouping, or per-item copies |
| 29 | 21 | 17 | 1.107 | 10568 | 10601 | 1.003 | 1.008 | ✓ | accepted | Stack: per-group padding skip, free input items on thread 2, hoist weight-only curve terms |
| 30 | 22 | 15 | 1.333 | 10601 | 10607 | 1.001 | 1.001 | ✓ | accepted | Force-inline the SIMD step, curve, stability and difficulty functions into the time loops |
| 31 | 14 | 16 | 0.951 | 10607 | 10607 | 1.000 | 1.000 | ✓ | rejected | Build the binding with fat LTO and one codegen unit |
| 32 | 14 | 16 | 1.000 | 10602 | 10712 | 1.010 | 1.026 | ✓ | rejected | Run two 8-card groups in lockstep so their recurrences overlap |
| 33 | 14 | 13 | 1.101 | 10607 | 10603 | 1.000 | 0.999 | ✓ | accepted | Stack: exp8 single-convert rounding, peel first step, drop dead/cheap cache fields |
| 34 | 12 | 13 | 0.948 | 10603 | 10603 | 1.000 | 1.000 | ✓ | rejected | Run the SIMD kernels 4 cards per lane (f32x4) instead of 8 |
| 35 | 12 | 12 | 1.046 | 10603 | 10622 | 1.002 | 1.004 | ✓ | rejected | Stack: inline loss-grad helper, one-check load8, cache iterator, drop proven-redundant exp clamps |
| 36 | 12 | 11 | 1.071 | 10603 | 10616 | 1.001 | 1.003 | ✓ | accepted | iter35 stack plus: skip re-clamping already-clamped states, drop ln8's redundant exponent mask |
| 37 | 13 | 10 | 1.086 | 10616 | 10775 | 1.015 | 1.038 | ✓ | accepted | Second thread shares each batch's card groups (dynamic claims), also frees the input items |
| 38 | 12 | 12 | 0.995 | 10775 | 10788 | 1.001 | 1.004 | ✓ | rejected | exp8: add n to the polynomial's exponent bits instead of multiplying by 2^n |
| 39 | 11 | 10 | 1.052 | 10775 | 10869 | 1.009 | 1.022 | ✓ | accepted | Helper lays out batches in training order; no per-step clock reads, allocations, dead penalty value |
| 40 | 11 | 10 | 1.065 | 10869 | 10907 | 1.004 | 1.009 | ✓ | accepted | Curve mix as sigmoid (one exp fewer) + exact kernel, plan, penalty, Adam trims |
| 41 | 10 | 9 | 1.053 | 10907 | 10916 | 1.001 | 1.002 | ✓ | accepted | Backward: share product adjoints, merge divisions, apply weight-only factors once per group |
| 42 | 9 | 8 | 1.097 | 10916 | 10972 | 1.005 | 1.013 | ✓ | accepted | Skip post-lapse branch on lapse-free steps; windowed path drops padding passthrough; share products |
| 43 | 8 | 7 | 1.053 | 10972 | 11066 | 1.009 | 1.022 | ✓ | accepted | Base-2 exp/log in the recurrence; lapse-only terms skipped; signed weights; no burn Model |
| 44 | 7 | 7 | 0.998 | 11066 | 11072 | 1.000 | 1.001 | ✓ | rejected | Fold aa, w23, factor2 into their exponentials (fewer multiplies per step) |
| 45 | 8 | 7 | 1.060 | 11066 | 11169 | 1.009 | 1.024 | ✓ | accepted | Forward writes cache fields straight into reused slots; items freed in touched blocks |
| 46 | 8 | 8 | 1.035 | 11169 | 11213 | 1.004 | 1.010 | ✓ | rejected | Cache cheap forward values for the backward; backward ordered so values die early |
| 47 | 7 | 7 | 1.051 | 11169 | 11258 | 1.008 | 1.020 | ✓ | accepted | iter46 stack plus: no padding mask on lapse-free steps, leaner plan, transposed lane sums |
| 48 | 7 | 6 | 1.006 | 11258 | 11317 | 1.005 | 1.013 | ✓ | rejected | Batch's last card group with at most 4 cards runs on a 4-lane kernel |

**Cumulative speed_ratio (product of accepted): ×354.263** — upward-biased (winner's curse); anchor periodically vs iter-0 baseline.

