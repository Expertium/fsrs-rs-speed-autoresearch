// The f32 SIMD kernels, written against a lane type `F` (its integer twin `I`) and lane count
// `N`: analytic.rs includes this file twice, as mod k8 (F = f32x8, every batch) and mod k4
// (F = f32x4, a batch's last card group when it holds at most 4 cards). Group column offsets
// stay in units of 8 columns (the batch layout), so k4 reads the first 4 columns of a group.
// ===================== portable SIMD transcendentals (8 lanes) =====================
// These vectorize the analytic forward across 8 cards/lane. A wide::F exp is ~3.8x faster
// than 8x scalar libm (microbench profiling/simd_bench) at ~2e-7 rel err — the forward is ~92%
// transcendentals, so this is the lever. Precision-trading (3b band), portable (constraint 7).

#[inline(always)]
pub(super) fn clamp8(x: F, lo: f32, hi: f32) -> F {
    x.fast_max(F::splat(lo)).fast_min(F::splat(hi))
}

/// exp over 8 lanes: 2^n · poly(r), x = n·ln2 + r. Same algorithm as the scalar floor-bench.
///
/// `FAST` (compile-time const): `false` = degree-3 minimax (rel err 7.5e-5) — the accurate version,
/// now used by EVERY path (including the windowed compute_parameters() recurrence). `true` = degree-2
/// (rel err 1.7e-3, ONE fewer FMA) — this was iter23's cruder windowed-only trade, but it was
/// REVERTED 2026-06-04 on cost/benefit (it saved ~6.5% median speed but cost ~0.00025 cp log loss),
/// so the `true` branch is currently UNUSED (retained for a possible future precision A/B). Portable (c7).
#[inline(always)]
pub(super) fn exp8<const FAST: bool>(x: F) -> F {
    // The clamp also maps NaN to -87 (maxps returns its 2nd operand on NaN).
    exp8_in_range::<FAST>(x.fast_max(F::splat(-87.0)).fast_min(F::splat(88.0)))
}

/// exp8 without its clamp, for arguments already inside [-87, 88]. Every forward call in the
/// SIMD kernels qualifies, because clip_parameters bounds the weights and the states are clamped:
/// p35/p31/cc have |x| <= 2 * ln(36500 / 1e-4) ~ 21; se/ex34/qbase/pr/expr/init |x| <= ~13;
/// e1 has an explicit min(60) and q1 > 0; r1/r2 = decay * ln(b) >= -0.95 * 88.7 (ln8 of even an
/// overflowed +inf base is 128 * ln2), so the clamp never changes a value there (bit-for-bit).
/// x*LOG2E is then finite and within +-128: ONE plain nearest-even convert gives n exactly as
/// round() did, without round()'s and round_int()'s NaN/overflow fix-ups (~10 ops/half on SSE2).
#[inline(always)]
pub(super) fn exp8_in_range<const FAST: bool>(x: F) -> F {
    exp2_8_in_range::<FAST>(x * F::splat(LOG2E))
}

/// 2^y over 8 lanes (= exp8_in_range(y * ln2)); |y| <= ~126. Callers whose exp argument is a
/// constant times a variable fold log2(e) into the constant (WConsts::l2e) and call this directly.
#[inline(always)]
pub(super) fn exp2_8_in_range<const FAST: bool>(y: F) -> F {
    let ni = y.fast_round_int();
    let n = ni.round_float();
    let c = |v: f32| F::splat(v);
    let p = if FAST {
        // degree-2 RELATIVE-minimax of exp(r) over [-ln2/2, ln2/2]; max rel err 1.7e-3.
        let r = (y - n) * F::splat(LN2);
        c(1.00044314196) + r * (c(1.01486094962) + r * c(0.496258591073))
    } else {
        // The degree-3 RELATIVE-minimax of exp(r) over [-ln2/2, ln2/2] (Remez; max rel err 7.5e-5,
        // profiling/minimax_coeffs.py), written as a polynomial in f = y - n = r / ln2 (so 2^f):
        // coefficient k times ln2^k. The same approximation, one multiply (n * ln2) less.
        let f = y - n;
        c(0.999928073539) + f * (c(0.6932609854635089) + f * (c(0.24261112219325395) + f * c(0.055171669057997336)))
    };
    let bits: I = (ni + I::splat(127)) << 23;
    let two_n: F = bytemuck::cast(bits);
    p * two_n
}

/// ln over 8 lanes: x = m·2^e, ln = e·ln2 + minimax(atanh-series) in t=(m-1)/(m+1).
/// All forward ln inputs are > 0 (stabilities ≥ S_MIN, difficulty ≥ 1, the b/qbase bases > 0).
///
/// `FAST` (compile-time const): `false` = degree-2-in-u minimax (abs err 4.9e-6) — the accurate
/// version, now used by EVERY path (including the windowed compute_parameters() recurrence). `true` =
/// degree-1-in-u (abs err 2.3e-4, ONE fewer FMA) — iter23's cruder windowed-only trade, REVERTED
/// 2026-06-04 on cost/benefit, so the `true` branch is currently UNUSED. Portable (c7).
#[inline(always)]
pub(super) fn ln8<const FAST: bool>(x: F) -> F {
    let bits: I = bytemuck::cast(x);
    // x > 0, so the sign bit is 0 and bits >> 23 is the biased exponent (no & 0xff needed).
    let e: I = (bits >> 23) - I::splat(127);
    let mant_bits: I = (bits & I::splat(0x007f_ffff)) | I::splat(127 << 23);
    let m: F = bytemuck::cast(mant_bits);
    let one = F::splat(1.0);
    let t = (m - one) / (m + one);
    let t2 = t * t;
    let c = |v: f32| F::splat(v);
    let poly = if FAST {
        // degree-1-in-u (u=t^2) minimax of atanh(t)/t over u in [0,1/9]; abs ln err 2.3e-4.
        c(2.0) * t * (c(0.999650356749) + t2 * c(0.357486937559))
    } else {
        // degree-2-in-u minimax; reconstructed abs ln err 4.9e-6 (profiling/minimax_coeffs.py).
        c(2.0) * t * (c(1.0000073389) + t2 * (c(0.332179529507) + t2 * c(0.226577770996)))
    };
    let e_f: F = e.round_float();
    e_f * F::splat(LN2) + poly
}

/// log2 over 8 lanes: ln8's approximation divided by ln2 (the poly's leading constant scaled
/// by log2(e)), without ln8's e * ln2 multiply. The recurrence keeps its logs in base 2: they only
/// feed 2^x (exp2_8_in_range) and weight gradients (finish_gw applies the ln2 once per group).
#[inline(always)]
pub(super) fn log2_8(x: F) -> F {
    let bits: I = bytemuck::cast(x);
    let e: I = (bits >> 23) - I::splat(127);
    let mant_bits: I = (bits & I::splat(0x007f_ffff)) | I::splat(127 << 23);
    let m: F = bytemuck::cast(mant_bits);
    let one = F::splat(1.0);
    let t = (m - one) / (m + one);
    let t2 = t * t;
    let c = |v: f32| F::splat(v);
    let poly = c(2.0 * LOG2E) * t * (c(1.0000073389) + t2 * (c(0.332179529507) + t2 * c(0.226577770996)));
    e.round_float() + poly
}

// ===================== SIMD validation forward (8 cards/lane) =====================
// Vectorized, FORWARD-ONLY mirror of step_fwd's recurrence + the final curve, used by the
// per-epoch validation scorer. exp8/ln8 replace libm exp/ln (precision-trade, 3b band). The
// data layout [seq_len, batch] makes 8 consecutive cards at one timestep a contiguous F load.

#[inline(always)]
pub(super) fn load8(s: &[f32], i: usize) -> F {
    // One bounds check for the 8 lanes (was 8 index checks).
    let a: [f32; N] = s[i..i + N].try_into().unwrap();
    F::from(a)
}

// F forgetting curve forward, storing the intermediates its backward needs (the F
// analogue of CurveCache). `out` is bit-identical to the old forward-only curve_out8 — it just
// also stashes the products + cached lns so curve8_bwd never recomputes a transcendental.
#[derive(Default)]
pub(super) struct Curve8 {
    out: F,
    a: F,
    q2: F,
    r1: F,
    q1: F,
    e1: F,
    p35: F,
    m1: F,
    decay1: F,
    factor1: F,
    b1: F,
    b2: F,
    r2: F,
    sig: F,
    oms: F,
    ln_sf: F,
    ln_b1: F,
    ln_b2: F,
    ln_s: F,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn curve8_fwd<const FAST: bool>(
    w: &[f32], t: F, s: F, sf: F, d: F, wc: &WConsts, ln_s: F, ln_sf: F,
) -> Curve8 {
    let mut c = Curve8::default();
    curve8_fwd_into::<FAST>(w, t, s, sf, d, wc, ln_s, ln_sf, &mut c);
    c
}

/// curve8_fwd writing into `c` (a per-step cache slot). Each field is stored as soon as it is
/// computed, so it need not stay live (or be spilled and copied) until the end of the step.
/// Returns (out, r1).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn curve8_fwd_into<const FAST: bool>(
    w: &[f32], t: F, s: F, sf: F, d: F, wc: &WConsts, ln_s: F, ln_sf: F,
    c: &mut Curve8,
) -> (F, F) {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    c.ln_s = ln_s;
    c.ln_sf = ln_sf;
    let t = t.fast_max(k(0.0));
    let a = t / sf;
    c.a = a;
    let bv = t / s;
    // 34-param remap + all-positive offsets: p35=s_short^(s_decay1[w33]-0.3), m1=decay1[w23]*p35,
    // ex34=exp((d-5)*(d_decay[w32]-0.3)), m2=decay2[w24]*ex34, p28=base2[w26]^inv2,
    // p31=s_short^-s_weight_power1[w29], weight1=base_weight1[w27]*p31. ln_w27=ln(base1=w25),
    // ln_w28=ln(base2=w26).
    let p35 = exp2_8_in_range::<FAST>((sp(33) - k(0.3)) * ln_sf); // ln_sf, ln_s are log2
    c.p35 = p35;
    let m1 = sp(23) * p35;
    c.m1 = m1;
    let dm1 = clamp8(m1, 0.01, 0.95);
    let decay1 = k(0.0) - dm1;
    c.decay1 = decay1;
    let q1 = k(wc.ln_w27) / decay1;
    c.q1 = q1;
    let e1 = exp8_in_range::<FAST>(q1.fast_min(k(60.0)));
    c.e1 = e1;
    let factor1 = e1 - k(1.0);
    c.factor1 = factor1;
    let b1 = a * factor1 + k(1.0);
    c.b1 = b1;
    let ln_b1 = log2_8(b1);
    c.ln_b1 = ln_b1;
    let r1 = exp2_8_in_range::<FAST>(decay1 * ln_b1);
    c.r1 = r1;
    // iter-165: decay2 no longer d-modulated (m2 = w24 plain); ex34 = exp((d-5)*(d_decay-0.3))
    // is now the TIME-SCALE inside b2.
    let ex34 = exp2_8_in_range::<FAST>((d - k(5.0)) * k(wc.w32m_l));
    // decay2 / p28 = base2[w26]^inv2 / factor2 are weight-only: hoisted into wc.
    let q2 = bv * F::splat(wc.factor2) * ex34;
    c.q2 = q2;
    let b2 = q2 + k(1.0);
    c.b2 = b2;
    let ln_b2 = log2_8(b2);
    c.ln_b2 = ln_b2;
    let r2 = exp2_8_in_range::<FAST>(F::splat(wc.decay2) * ln_b2);
    c.r2 = r2;
    // ret = (weight1*r1 + weight2*r2) / (weight1 + weight2), weight1 = base_weight1[w27] *
    // sf^-s_weight_power1[w29], weight2 = base_weight2[w28] * s^s_weight_power2[w30] *
    // exp((d_weight[w31]-0.5)(d-5)). Written as r1*(1-sig) + r2*sig with sig = sigmoid(z) and
    // z = ln(weight2/weight1): the same function with ONE exp instead of two (3b reassociation).
    // |z| <= ln(100) + 1.1*ln(36500) + 2.5 + 0.9*ln(36500) < 29, inside exp8's range.
    let zl = k(wc.lz0_l) + sp(30) * ln_s + (d - k(5.0)) * k(wc.w31m_l) + sp(29) * ln_sf;
    let ez = exp2_8_in_range::<FAST>(zl); // zl = z * log2(e)
    let oms = k(1.0) / (ez + k(1.0)); // 1 - sig
    c.oms = oms;
    let sig = ez * oms;
    c.sig = sig;
    let ret = r1 * oms + r2 * sig;
    let out = ret * k(1.0 - 2e-5) + k(1e-5);
    c.out = out;
    (out, r1)
}

/// VJP of curve8_fwd (the F analogue of curve_bwd; every scalar `if` is a lane blend, every
/// gw[] += is an F accumulate). Returns (g_s, g_sf, g_d); accumulates into the F gw bank.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn curve8_bwd(
    w: &[f32], c: &Curve8, s: F, sf: F, d: F, g_out: F, g_r1_extra: F,
    gw: &mut [F; 34], wc: &WConsts,
) -> (F, F, F) {
    let z = F::splat(0.0);
    curve8_bwd_acc::<false>(w, c, s, sf, d, g_out, g_r1_extra, gw, wc, (z, z, z))
}

/// curve8_bwd; with ACC, returns (acc.0 + g_s, acc.1 + g_sf, acc.2 + g_d) instead, each sum formed
/// as soon as its curve term is known. The code is ordered so values die early: the kernel's basic
/// blocks are far longer than LLVM's scheduling window, so the source order decides how many
/// values are live (and spilled) at once.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn curve8_bwd_acc<const ACC: bool>(
    w: &[f32], c: &Curve8, s: F, sf: F, d: F, g_out: F, g_r1_extra: F,
    gw: &mut [F; 34], wc: &WConsts, acc: (F, F, F),
) -> (F, F, F) {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    let z = k(0.0);
    let g_ret = g_out * k(1.0 - 2e-5);
    // ret = r1*(1-sig) + r2*sig. r1 also feeds the short-trace stability update (reads r1, not
    // mixed R) -> g_r1_extra.
    let g_r1 = g_ret * c.oms + g_r1_extra;
    let g_r2 = g_ret * c.sig;
    // sig = sigmoid(z), dsig/dz = sig*(1-sig); z = ln(w28/w27) + w30*ln_s + (w31-0.5)*(d-5) + w29*ln_sf
    let g_z = g_ret * (c.r2 - c.r1) * (c.sig * c.oms);
    // Weight-only factors are applied once per group in finish_gw (slot 27 holds sum(g_z)).
    gw[27] += g_z;
    gw[29] += g_z * c.ln_sf;
    gw[30] += g_z * c.ln_s;
    gw[31] += g_z * (d - k(5.0));
    let mut g_d = g_z * (sp(31) - k(0.5));
    let gz30 = g_z * sp(30); // the ln(s) and ln(sf) terms of g_s / g_sf (below)
    let gz29 = g_z * sp(29);
    // r2 = b2^decay2, b2 = q2 + 1 with q2 = (t/s)*factor2*ex34 (ex34 = exp((d-5)*(d_decay-0.3)),
    // factor2 = p28 - 1, p28 = base2[w26]^(1/decay2), decay2 = -clamp(w24)). q2 is a product, so
    // the adjoints of ln(t/s), ln(ex34) and ln(factor2) are all x2 = g_b2*q2.
    let v2 = g_r2 * c.r2; // adjoint of ln(r2) = decay2 * ln(b2)
    // w26 and w24 enter r2 only through weight-only factors: slot 26 holds sum(x2), slot 28 the
    // direct decay2 term; finish_gw forms gw[26] and gw[24] from them.
    gw[28] += v2 * c.ln_b2;
    let x2 = v2 * F::splat(wc.decay2) / c.b2 * c.q2;
    g_d += x2 * (sp(32) - k(0.3));
    gw[32] += x2 * (d - k(5.0));
    gw[26] += x2;
    // d/ds of every ln(s) term, over ONE division.
    let g_s = (gz30 - x2) / s;
    let g_s = if ACC { acc.0 + g_s } else { g_s };
    let g_d = if ACC { acc.2 + g_d } else { g_d };
    // r1 = b1^decay1, b1 = a*factor1 + 1, a = t/sf, factor1 = e1 - 1, e1 = exp(min(q1, 60)),
    // q1 = ln(base1[w25]) / decay1, decay1 = -clamp(w23 * p35), p35 = sf^(s_decay1[w33]-0.3).
    let decay1 = c.decay1;
    let v1 = g_r1 * c.r1; // adjoint of ln(r1) = decay1 * ln(b1)
    let mut g_decay1 = v1 * c.ln_b1 * k(LN2); // ln(b1) = log2(b1) * ln2
    let g_b1_a = v1 * decay1 / c.b1 * c.a; // adjoint of b1, times a
    let g_a_a = g_b1_a * c.factor1; // adjoint of ln(a)
    let g_q1 = c.q1.cmp_lt(k(60.0)).blend(g_b1_a * c.e1, z);
    let g_lw27 = g_q1 / decay1;
    g_decay1 -= g_lw27 * c.q1; // d(q1)/d(decay1) = -q1/decay1
    gw[25] += g_lw27; // ln_base1 = ln(w[25]); finish_gw applies 1/w25
    let m1 = c.m1;
    let g_m1 = (m1.cmp_gt(k(0.01)) & m1.cmp_lt(k(0.95))).blend(z - g_decay1, z);
    let u = g_m1 * c.p35;
    gw[23] += u;
    let y = u * sp(23); // adjoint of ln(p35)
    gw[33] += y * c.ln_sf;
    // d/dsf of every ln(sf) term, over ONE division.
    let g_sf = (gz29 + y * (sp(33) - k(0.3)) - g_a_a) / sf;
    let g_sf = if ACC { acc.1 + g_sf } else { g_sf };
    (g_s, g_sf, g_d)
}

/// hard_penalty (rating 2) or easy_bonus (rating 4), else 1. At most one of the two factors of
/// `x * hard * easy` differs from 1 and a product with 1 is exact, so `x * hard_easy8` is the same
/// value with one multiply less.
#[inline(always)]
pub(super) fn hard_easy8(w: &[f32], rating: F, start: usize) -> F {
    let k = F::splat;
    rating.cmp_eq(k(2.0)).blend(k(w[start + 6]), rating.cmp_eq(k(4.0)).blend(k(w[start + 7]), k(1.0)))
}

// F stability-after-review forward + the intermediates its backward needs (analogue of StabCache).
#[derive(Default)]
pub(super) struct Stab8 {
    out: F,
    nsf_fail: F,
    sinc: F,
    cc: F,
    expr: F,
    /// hard_easy8, expr - 1, aa * bb * cc and aa * bb * cc * (expr - 1): partial products of sinc.
    he: F,
    em1: F,
    abc: F,
    base: F,
    pr: F,
    qbase: F,
    ln_ls1: F,
    /// The post-lapse branch was computed (some lane lapsed or tied); else nsf_fail/pr/qbase/ln_ls1
    /// are not written (stale) and unused.
    full: bool,
}

/// Stability after a review, writing the intermediates its backward needs into `c` (a per-step
/// cache slot, see curve8_fwd_into); returns the new stability.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn stab8_fwd_into<const FAST: bool>(
    w: &[f32], wc: &WConsts, last_s: F, last_d: F, r: F, rating: F, start: usize, aa: f32,
    ln_ls: F, c: &mut Stab8,
) -> F {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    let one = k(1.0);
    let he = hard_easy8(w, rating, start);
    c.he = he;
    let bb = k(11.0) - last_d;
    let cc = exp2_8_in_range::<FAST>(k(-w[start + 1]) * ln_ls);
    c.cc = cc;
    let expr = exp2_8_in_range::<FAST>((one - r) * k(wc.l2e[start + 2]));
    c.expr = expr;
    let em1 = expr - one;
    c.em1 = em1;
    let abc = k(aa) * bb * cc;
    c.abc = abc;
    let base = abc * em1;
    c.base = base;
    let sinc = base * he + one; // = aa * bb * cc * (expr - 1) * he + 1
    c.sinc = sinc;
    let ls_sinc = last_s * sinc;
    // sinc >= 1, so ls_sinc >= last_s >= pls = min(last_s, nsf_fail): on a success lane the new
    // stability max(pls, ls_sinc) is ls_sinc and pls takes the gradient only on a tie. So the
    // post-lapse branch (a ln and two exps) is needed only if some lane lapses or ties; the value
    // of a padding lane (rating 0) is never used (step8 PAD / trailing weight-0 padding).
    let full = (rating.cmp_eq(one) | ls_sinc.cmp_eq(last_s)).any();
    c.full = full;
    if !full {
        c.out = ls_sinc;
        return ls_sinc;
    }
    let ln_ls1 = log2_8(last_s + one);
    c.ln_ls1 = ln_ls1;
    let qbase = exp2_8_in_range::<FAST>(sp(start + 4) * ln_ls1); // (last_s+1)^fail_s_exp[start+4]
    c.qbase = qbase;
    // fail_d_exp DROPPED: post-lapse stability is D-independent, so the legacy `pr` cache field now
    // holds just rexp (no pp = last_d^-fail_d_exp factor, so ln(last_d) is not needed).
    let pr = exp2_8_in_range::<FAST>((one - r) * k(wc.l2e[start + 5])); // rexp = exp((1-r)*fail_r_mult[start+5])
    c.pr = pr;
    let nsf_fail = sp(start + 3) * pr * (qbase - one);
    c.nsf_fail = nsf_fail;
    let pls = last_s.fast_min(nsf_fail);
    let nss = pls.fast_max(ls_sinc);
    let out = rating.cmp_gt(one).blend(nss, pls);
    c.out = out;
    out
}

/// VJP of stab8_fwd_into (F analogue of stab_bwd). Returns (g_last_s, g_last_d, g_r).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn stab8_bwd(
    w: &[f32], c: &Stab8, last_s: F, r: F, rating: F, start: usize,
    aa: f32, ln_ls: F, g_out: F, gw: &mut [F; 34],
) -> (F, F, F) {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    let one = k(1.0);
    let z = k(0.0);
    // aa / ln_ls are passed in (not cached): the same values stab8_fwd_into used.
    let aa = k(aa);
    let he = c.he;
    // Without the post-lapse branch (see stab8_fwd_into) every lane routes to ls_sinc: no lane
    // lapsed, so every real lane has rating > 1, and a padding lane's adjoint is already zero.
    let (mut g_last_s, g_ls_sinc, g_nsf_fail) = if c.full {
        let gt1 = rating.cmp_gt(one);
        let g_nss = gt1.blend(g_out, z);
        let pls = last_s.fast_min(c.nsf_fail);
        let ls_sinc = last_s * c.sinc;
        let g_pls_direct = gt1.blend(z, g_out);
        // nss = max(pls, ls_sinc)  (ties to pls, matching the scalar >= / > split)
        let g_pls_from_nss = pls.cmp_ge(ls_sinc).blend(g_nss, z);
        let g_ls_sinc = ls_sinc.cmp_gt(pls).blend(g_nss, z);
        let g_pls = g_pls_direct + g_pls_from_nss;
        // pls = min(last_s, nsf_fail)
        let g_ls = last_s.cmp_le(c.nsf_fail).blend(g_pls, z);
        (g_ls, g_ls_sinc, c.nsf_fail.cmp_lt(last_s).blend(g_pls, z))
    } else {
        (z, g_out, z)
    };
    g_last_s += g_ls_sinc * c.sinc;
    let g_sinc = g_ls_sinc * last_s;
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1
    let em1 = c.em1;
    let g_prod = g_sinc;
    let base = c.base;
    // d(prod)/d(hard_penalty) on rating-2 lanes and d(prod)/d(easy_bonus) on rating-4 lanes are
    // both `base` (the other factor is exactly 1 there).
    let g_base = g_prod * base;
    // prod = base * he is a product, so the adjoint of ln(aa) and of ln(cc) are both g_prod * prod.
    let p_ln = g_base * he;
    gw[start] += p_ln; // aa = exp(w[start]-1.5)
    let g_he = g_prod * he;
    let g_bb = g_he * (aa * c.cc * em1);
    let g_em1 = g_he * c.abc;
    gw[start + 6] += rating.cmp_eq(k(2.0)).blend(g_base, z); // hard_penalty
    gw[start + 7] += rating.cmp_eq(k(4.0)).blend(g_base, z); // easy_bonus
    let g_last_d = g_bb * (z - one); // bb = 11 - last_d (the ONLY D-dependence of stab now)
    // cc = last_s^-w[start+1] = exp(-w[start+1] * ln_ls)
    g_last_s += p_ln * k(-w[start + 1]) / last_s;
    gw[start + 1] -= p_ln * ln_ls;
    // expr = exp((1-r)*w[start+2])
    let mut g_r = g_em1 * c.expr * k(-w[start + 2]);
    gw[start + 2] += g_em1 * c.expr * (one - r);
    if !c.full {
        return (g_last_s, g_last_d, g_r);
    }
    // nsf_fail = fail_mult[start+3] * pr * (qbase-1) ; pr = rexp = exp((1-r)*fail_r_mult[start+5])
    // (fail_d_exp DROPPED: no pp = last_d^-x factor, so nsf_fail is D-independent).
    // The adjoint of ln(fail_mult) and of ln(pr) are both n_ln = g_nsf_fail * nsf_fail.
    let n_ln = g_nsf_fail * c.nsf_fail;
    gw[start + 3] += n_ln; // finish_gw applies 1/fail_mult
    // pr = exp((1-r)*fail_r_mult[start+5])
    g_r += n_ln * k(-w[start + 5]);
    gw[start + 5] += n_ln * (one - r);
    // qbase = (last_s+1)^fail_s_exp[start+4]; q_ln = adjoint of ln(qbase)
    let q_ln = g_nsf_fail * sp(start + 3) * c.pr * c.qbase;
    g_last_s += q_ln * sp(start + 4) / (last_s + one);
    gw[start + 4] += q_ln * c.ln_ls1;
    (g_last_s, g_last_d, g_r)
}

/// F next-difficulty forward; returns (clamped out, pre-clamp out, delta_d) for the backward.
#[inline(always)]
pub(super) fn next_d8_fwd(
    w: &[f32], last_d: F, rating: F, r: F, init: f32, lapse: bool,
) -> (F, F, F) {
    let k = F::splat;
    let delta_d_base = k(-w[6]) * (rating - k(3.0));
    // Surprise-weighted lapse: on a lapse scale delta_d by (r+0.1) = 1 + (R-0.9).
    let delta_d = if lapse {
        rating.cmp_eq(k(1.0)).blend(delta_d_base * (r + k(0.1)), delta_d_base)
    } else {
        delta_d_base
    };
    let new_d = last_d + (k(10.0) - last_d) * delta_d / k(9.0);
    let out_pre = k(0.01) * k(init) + k(0.99) * new_d;
    (clamp8(out_pre, D_MIN, D_MAX), out_pre, delta_d) // delta_d returned = EFFECTIVE delta_d
}

/// VJP of next_d8_fwd. Returns (g_last_d, g_r); accumulates gw[4], gw[5], gw[6]. `delta_d` is the
/// EFFECTIVE delta_d; `r` is the curve retention (feeds the lapse surprise weighting).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn next_d8_bwd(
    w: &[f32], out_pre: F, delta_d: F, last_d: F, rating: F, r: F, g_out: F,
    gw: &mut [F; 34], exp3w5: f64, lapse: bool,
) -> (F, F) {
    let k = F::splat;
    let z = k(0.0);
    let g_out_pre = (out_pre.cmp_gt(k(D_MIN)) & out_pre.cmp_lt(k(D_MAX))).blend(g_out, z);
    let g_init = g_out_pre * k(0.01);
    let g_new_d = g_out_pre * k(0.99);
    gw[4] += g_init;
    gw[5] += g_init * k(-(exp3w5 as f32) * 3.0); // init = w4 - exp(3 w5) + 1
    let g_last_d = g_new_d * (k(1.0) - delta_d / k(9.0));
    let g_delta_d = g_new_d * (k(10.0) - last_d) / k(9.0);
    let rm3 = rating - k(3.0);
    if !lapse {
        gw[6] += g_delta_d * (z - rm3); // d(delta_d)/d(w6) = -(rating-3)
        return (g_last_d, z);
    }
    let is_lapse = rating.cmp_eq(k(1.0));
    // d(delta_d_eff)/d(w6) = -(rating-3), scaled by (r+0.1) on a lapse.
    gw[6] += g_delta_d * is_lapse.blend((z - rm3) * (r + k(0.1)), z - rm3);
    // d(delta_d_eff)/dr = delta_d_base = -w6*(rating-3), only on a lapse.
    let g_r = is_lapse.blend(g_delta_d * (k(-w[6]) * rm3), z);
    (g_last_d, g_r)
}

/// Vectorized forward-only BCE loss (validation). Processes the batch 8 cards at a time; the
/// remainder (<8) uses the scalar batch_loss path. Precision-trade vs batch_loss (exp8/ln8 differ
/// from libm by ~1e-6) — judged by the 3b average-log-loss band. The BCE itself stays f64.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss_simd(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    delta_ts: &[f32], labels: &[f32], weights: &[f32],
) -> f64 {
    let wc = wconsts(w);
    let k = F::splat;
    let one = k(1.0);
    let mut loss = 0.0f64;
    let n_groups = batch / 8;
    for g in 0..n_groups {
        let c0 = g * 8;
        // Per-group active length: trailing all-padding timesteps (rating 0) pass the state through
        // unchanged, so the final state — and thus the scored curve below — is bit-for-bit identical
        // when we stop at `sl`. (iter26's per-group skip; see loss_and_grad_range_simd.)
        let mut sl = seq_len;
        while sl > 1 && load8(r_hist, (sl - 1) * batch + c0).reduce_add() == 0.0 {
            sl -= 1;
        }
        let (mut s, mut d, mut sf) = (k(0.0), k(0.0), k(0.0));
        for t in 0..sl {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            if t == 0 {
                // First review: initialise state from rating (mirrors step_fwd's nth0 override).
                let rc = clamp8(rating, 1.0, 4.0);
                let init_s = rc.cmp_eq(one).blend(
                    k(w[0]),
                    rc.cmp_eq(k(2.0)).blend(k(w[1]), rc.cmp_eq(k(3.0)).blend(k(w[2]), k(w[3]))),
                );
                let init_d = clamp8(k(w[4]) - exp8::<false>(k(w[5]) * (rc - one)) + one, D_MIN, D_MAX);
                s = clamp8(init_s, S_MIN, S_MAX);
                d = init_d;
                sf = clamp8(k(0.8) * init_s, S_MIN, S_MAX);
            } else {
                let dt = load8(t_hist, base).fast_max(k(0.0));
                let s_c = clamp8(s, S_MIN, S_MAX);
                let d_c = clamp8(d, D_MIN, D_MAX);
                let sf_c = clamp8(sf, S_MIN, S_MAX);
                let ln_s = log2_8(s_c);
                let ln_sf = log2_8(sf_c);
                let curve = curve8_fwd::<false>(w, dt, s_c, sf_c, d_c, &wc, ln_s, ln_sf);
                let rr = curve.out;
                let scratch = &mut Stab8::default();
                let ns = stab8_fwd_into::<false>(w, &wc, s_c, d_c, rr, rating, 7, wc.aa7, ln_s, scratch);
                let nsf_raw = stab8_fwd_into::<false>(w, &wc, sf_c, d_c, curve.r1, rating, 15, wc.aa16, ln_sf, scratch);
                // POST-LAPSE short reset: on a lapse cap s_short at 0.8 * post-lapse s_long.
                let nsf = rating.cmp_eq(one).blend(nsf_raw.fast_min(k(0.8) * ns), nsf_raw);
                let nd = next_d8_fwd(w, d_c, rating, rr, wc.init, true).0;
                // rating==0 (padding) passes the state through unchanged.
                let m0 = rating.cmp_eq(k(0.0));
                s = clamp8(m0.blend(s_c, ns), S_MIN, S_MAX);
                d = m0.blend(d_c, nd);
                sf = clamp8(m0.blend(sf_c, nsf), S_MIN, S_MAX);
            }
        }
        let dts = load8(delta_ts, c0);
        let r = clamp8(
            curve8_fwd::<false>(w, dts, s, sf, d, &wc, log2_8(s), log2_8(sf)).out,
            MIN_R,
            MAX_R,
        );
        // Branchless BCE for 0/1 labels (Andrew): -ln(1 - |label - r|) = -ln(r) for label 1,
        // -ln(1-r) for label 0 — math-identical for hard labels. Computes the whole 8-lane BCE as
        // ONE vectorized ln8 (vs r.to_array() + 8 scalar f64 lns), summed in f64 via reduce_add.
        // Inexact (f32 minimax ln8 vs f64 libm ln), but this loss only feeds best-epoch argmin and the
        // error ≪ the epoch-to-epoch gaps, so best_w (hence params/benchmark loss) is unchanged.
        let lbl = load8(labels, c0);
        let wt = load8(weights, c0);
        let arg = one - (lbl - r).fast_max(r - lbl); // 1 - |label - r|  (r already clamped into (0,1))
        loss += ((k(0.0) - wt) * ln8::<false>(arg)).reduce_add() as f64;
    }
    // Remainder (< 8 cards) via the scalar path.
    for c in (n_groups * 8)..batch {
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            state = step_fwd(w, t_hist[t * batch + c], r_hist[t * batch + c], state, t == 0, &wc).0;
        }
        let (s, d, sf) = state;
        let r = clamp(
            curve_fwd(w, delta_ts[c], s, sf, d, wc.ln_w27, wc.ln_w28, s.ln(), sf.ln()).out,
            MIN_R,
            MAX_R,
        );
        loss += -(weights[c] as f64)
            * (labels[c] as f64 * (r as f64).ln() + (1.0 - labels[c] as f64) * (1.0 - r as f64).ln());
    }
    loss
}

// ===================== SIMD recurrence step (8 cards/lane) with cache =====================
// Per-timestep cache for the vectorized backward. The first review (t==0) only needs the init
// override's data (the curve/stab/next_d it computes are dead, overridden), so it gets a small
// `First` variant; every later step stores the full forward intermediates (`Full`).
pub(crate) enum Step8 {
    First {
        rc: F,
        init_s: F,
        ex_w5: F,
        id_in: F,
    },
    Full {
        s0: F,
        d0: F,
        sf0: F,
        rating: F,
        curve: Curve8,
        slow: Stab8,
        fast: Stab8,
        nd_out_pre: F,
        nd_delta_d: F,
        /// Some lane lapses (rating 1) at this step; else the lapse-only terms are skipped.
        lapse: bool,
    },
}

impl Step8 {
    /// A `Full` slot for step8_fwd_into to fill.
    pub(crate) fn blank() -> Self {
        let z = F::splat(0.0);
        Step8::Full {
            s0: z, d0: z, sf0: z, rating: z, curve: Curve8::default(), slow: Stab8::default(),
            fast: Stab8::default(), nd_out_pre: z, nd_delta_d: z, lapse: false,
        }
    }

    /// The retrievability the curve predicted at this step (= the loss target R_t in the windowed
    /// O(N) forward). The first review (t==0) makes no prediction, so it returns 0 (the windowed
    /// kernels never score t==0; the minimum surviving prefix length is 2).
    #[inline(always)]
    fn curve_out(&self) -> F {
        match self {
            Step8::Full { curve, .. } => curve.out,
            Step8::First { .. } => F::splat(0.0),
        }
    }
}

/// One recurrence step over 8 cards. `first` (t==0) initialises state from the rating exactly like
/// batch_loss_simd; otherwise it mirrors step_fwd (curve + both stability traces + next-difficulty,
/// then the rating==0 padding passthrough). Returns the new state and the backward cache.
#[inline(always)]
pub(super) fn step8_fwd<const FAST: bool, const PAD: bool>(
    w: &[f32], dt_raw: F, rating: F, state: (F, F, F), first: bool, wc: &WConsts,
) -> ((F, F, F), Step8) {
    let mut c = Step8::blank();
    let ns = step8_fwd_into::<FAST, PAD>(w, dt_raw, rating, state, first, wc, &mut c);
    (ns, c)
}

/// step8_fwd writing its cache into `slot` (reused across groups by card_group_grad): the forward
/// stores each intermediate once, straight into the cache, instead of building the Step8 in
/// registers and spill slots and copying it at the push. Returns the new state.
#[inline(always)]
pub(super) fn step8_fwd_into<const FAST: bool, const PAD: bool>(
    w: &[f32], dt_raw: F, rating: F, state: (F, F, F), first: bool, wc: &WConsts,
    slot: &mut Step8,
) -> (F, F, F) {
    let k = F::splat;
    let one = k(1.0);
    let (s0, d0, sf0) = state;
    if first {
        let rc = clamp8(rating, 1.0, 4.0);
        let init_s = rc.cmp_eq(one).blend(
            k(w[0]),
            rc.cmp_eq(k(2.0)).blend(k(w[1]), rc.cmp_eq(k(3.0)).blend(k(w[2]), k(w[3]))),
        );
        let ex_w5 = exp8_in_range::<FAST>(k(w[5]) * (rc - one));
        let id_in = k(w[4]) - ex_w5 + one;
        let init_d = clamp8(id_in, D_MIN, D_MAX);
        let out = (
            clamp8(init_s, S_MIN, S_MAX),
            init_d,
            clamp8(k(0.8) * init_s, S_MIN, S_MAX),
        );
        *slot = Step8::First { rc, init_s, ex_w5, id_in };
        out
    } else {
        // The incoming state is a previous step8_fwd output, which is already clamped (the first
        // step clamps its init values too), so clamping it again would change nothing.
        let (last_s, last_d, last_sf) = (s0, d0, sf0);
        if let Step8::First { .. } = slot {
            *slot = Step8::blank();
        }
        let Step8::Full {
            s0: c_s0, d0: c_d0, sf0: c_sf0, rating: c_rating, curve, slow, fast, nd_out_pre, nd_delta_d,
            lapse: c_lapse,
        } = slot
        else {
            unreachable!()
        };
        (*c_s0, *c_d0, *c_sf0, *c_rating) = (s0, d0, sf0, rating);
        let ln_last_s = log2_8(last_s);
        let ln_last_sf = log2_8(last_sf);
        // r1 = short component recall — drives the short-trace update (iter-71)
        let (r, r1) = curve8_fwd_into::<FAST>(w, dt_raw, last_s, last_sf, last_d, wc, ln_last_s, ln_last_sf, curve);
        let slow_out = stab8_fwd_into::<FAST>(w, wc, last_s, last_d, r, rating, 7, wc.aa7, ln_last_s, slow);
        let fast_out = stab8_fwd_into::<FAST>(w, wc, last_sf, last_d, r1, rating, 15, wc.aa16, ln_last_sf, fast);
        let lapse = rating.cmp_eq(one).any();
        *c_lapse = lapse;
        let (nd, pre, dd) = next_d8_fwd(w, last_d, rating, r, wc.init, lapse);
        (*nd_out_pre, *nd_delta_d) = (pre, dd);
        // POST-LAPSE short reset (iter-97): on a lapse cap s_short at 0.8 * post-lapse s_long.
        let nsf_pre = if lapse {
            rating.cmp_eq(one).blend(fast_out.fast_min(k(0.8) * slow_out), fast_out)
        } else {
            fast_out
        };
        // PAD: rating==0 (padding) passes the input state through unchanged. Needed where a state is
        // read AFTER trailing padding (the O(N^2) paths score the final state). The windowed path
        // (PAD = false) never does: a card's padding only trails it, carries weight 0, and lanes are
        // independent, so padding lanes may evolve (their values stay finite: rating 0 takes the
        // clamped post-lapse branch) and contribute only zero gradients.
        let (ns3, nsf3, nd3) = if PAD {
            let m0 = rating.cmp_eq(k(0.0));
            (m0.blend(last_s, slow_out), m0.blend(last_sf, nsf_pre), m0.blend(last_d, nd))
        } else {
            (slow_out, nsf_pre, nd)
        };
        (clamp8(ns3, S_MIN, S_MAX), nd3, clamp8(nsf3, S_MIN, S_MAX))
    }
}

/// VJP of one step (F analogue of step_bwd). Given output-state adjoints, returns input-state
/// adjoints and accumulates the weight gradient into `gw`. `g_r_loss` is the adjoint of any loss
/// scored directly off this step's curve.out (nonzero only in the windowed O(N) forward, where every
/// step emits a prediction); it is added to the two stab r-adjoints before curve8_bwd, since curve.out
/// feeds the loss AND both stability traces. The O(N^2) callers pass 0 (their loss is the separate
/// final curve). The First variant ignores it (t==0 makes no prediction).
#[inline(always)]
pub(super) fn step8_bwd<const PAD: bool>(
    w: &[f32], c: &Step8, g_out: (F, F, F), g_r_loss: F, gw: &mut [F; 34], wc: &WConsts,
) -> (F, F, F) {
    let k = F::splat;
    let one = k(1.0);
    let z = k(0.0);
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    match c {
        // t==0 init override: the only weight grads are gw[rc-1] (init stability, scattered by the
        // per-lane rating via 4 masked adds) and gw[4]/gw[5] (init difficulty). Input adjoints are 0
        // (state before the first review is constant 0).
        Step8::First { rc, init_s, ex_w5, id_in } => {
            let g_ns3 = (init_s.cmp_gt(k(S_MIN)) & init_s.cmp_lt(k(S_MAX))).blend(g_ns_out, z);
            let nsf3 = k(0.8) * *init_s;
            let g_nsf3 = (nsf3.cmp_gt(k(S_MIN)) & nsf3.cmp_lt(k(S_MAX))).blend(g_nsf_out, z);
            let g_init_s = g_ns3 + g_nsf3 * k(0.8);
            gw[0] += rc.cmp_eq(k(1.0)).blend(g_init_s, z);
            gw[1] += rc.cmp_eq(k(2.0)).blend(g_init_s, z);
            gw[2] += rc.cmp_eq(k(3.0)).blend(g_init_s, z);
            gw[3] += rc.cmp_eq(k(4.0)).blend(g_init_s, z);
            let idmask = id_in.cmp_gt(k(D_MIN)) & id_in.cmp_lt(k(D_MAX));
            gw[4] += idmask.blend(g_nd_out, z);
            gw[5] += idmask.blend(g_nd_out * (z - *ex_w5 * (*rc - one)), z);
            (z, z, z)
        }
        Step8::Full {
            s0, d0, sf0, rating, curve, slow, fast, nd_out_pre, nd_delta_d, lapse,
        } => {
            // Cheap forward values recomputed with step8_fwd's exact ops (smaller per-step cache).
            let (last_s, last_d, last_sf) = (*s0, *d0, *sf0); // already clamped (see step8_fwd)
            let nsf_pre = if *lapse {
                rating.cmp_eq(one).blend(fast.out.fast_min(k(0.8) * slow.out), fast.out)
            } else {
                fast.out
            };
            // PAD: rating==0 padding passes the state through (output == input), so its adjoint
            // flows straight through (see step8_fwd; the windowed path does not need this).
            let m0 = rating.cmp_eq(z);
            let (ns3, nsf3) = if PAD {
                (m0.blend(last_s, slow.out), m0.blend(last_sf, nsf_pre))
            } else {
                (slow.out, nsf_pre)
            };
            let g_ns3 = (ns3.cmp_gt(k(S_MIN)) & ns3.cmp_lt(k(S_MAX))).blend(g_ns_out, z);
            let g_nsf3 = (nsf3.cmp_gt(k(S_MIN)) & nsf3.cmp_lt(k(S_MAX))).blend(g_nsf_out, z);
            let g_nd3 = g_nd_out;
            let (g_ns2, g_nsf2, g_nd2) = if PAD {
                (m0.blend(z, g_ns3), m0.blend(z, g_nsf3), m0.blend(z, g_nd3))
            } else {
                (g_ns3, g_nsf3, g_nd3)
            };
            // POST-LAPSE min routing: nsf_pre = (rating==1)? min(fast.out, 0.8*slow.out) : fast.out.
            let (g_fast_out, g_slow_from_relearn) = if *lapse {
                let is_lapse = rating.cmp_eq(one);
                let fast_wins = fast.out.cmp_le(k(0.8) * slow.out);
                (
                    is_lapse.blend(fast_wins.blend(g_nsf2, z), g_nsf2),
                    is_lapse.blend(fast_wins.blend(z, g_nsf2 * k(0.8)), z),
                )
            } else {
                (g_nsf2, z)
            };
            // LONG stab reads mixed retention curve.out; SHORT stab reads r1=curve.r1 (start 15).
            let (g_ls_a, g_ld_a, g_r_long) =
                stab8_bwd(
                    w, slow, last_s, curve.out, *rating, 7, wc.aa7, curve.ln_s,
                    g_ns2 + g_slow_from_relearn, gw,
                );
            let g_r_curve = g_r_long + g_r_loss;
            let (g_lsf_b, g_ld_b, g_r1_short) =
                stab8_bwd(
                    w, fast, last_sf, curve.r1, *rating, 15, wc.aa16, curve.ln_sf, g_fast_out,
                    gw,
                );
            let g_ld_ab = g_ld_a + g_ld_b;
            let (g_ld_c, g_r_nextd) =
                next_d8_bwd(w, *nd_out_pre, *nd_delta_d, last_d, *rating, curve.out, g_nd2, gw, wc.exp3w5, *lapse);
            // curve.out adjoint = long-stab r + windowed loss adjoint + next_d r; curve.r1 = short-stab r.
            // The input-state adjoints: g_last_s = g_ls_a + g_ls_d, g_last_sf = g_lsf_b + g_lsf_d and
            // g_last_d = ((g_ld_a + g_ld_b) + g_ld_c) + g_ld_d, summed inside curve8_bwd_acc.
            let (mut g_last_s, mut g_last_sf, mut g_last_d) = curve8_bwd_acc::<true>(
                w, curve, last_s, last_sf, last_d, g_r_curve + g_r_nextd, g_r1_short,
                gw, wc, (g_ls_a, g_lsf_b, g_ld_ab + g_ld_c),
            );
            if PAD {
                g_last_s += m0.blend(g_ns3, z);
                g_last_sf += m0.blend(g_nsf3, z);
                g_last_d += m0.blend(g_nd3, z);
            }
            // No clamp gate here: s0/d0/sf0 are the previous step's clamped outputs, and that step's
            // backward applies the same mask to this adjoint (S_MIN < x < S_MAX holds for the
            // clamped value exactly when it holds for the unclamped one; a padding passthrough
            // carries a zero adjoint), so gating here too would change nothing.
            (g_last_s, g_last_d, g_last_sf)
        }
    }
}

/// Vectorized forward + reverse-mode backward over card groups `[g_start, g_end)` (8 cards/lane).
/// Per-group weight gradients accumulate in an F bank, then horizontal-sum into the f64 `gw`
/// once per group — so cross-card/cross-group accumulation stays f64 while the per-card VJP is f32.
/// Requires the batch to be padded to a multiple of 8 (build_host_batches does this).
///
/// The loss VALUE is unused by training (only the gradient drives Adam — the training caller
/// discards the return, and validation uses `batch_loss_simd`), so the per-lane f64 BCE is skipped
/// here and 0.0 is returned. This is bit-for-bit on the gradient: `gw` is built only from `g_r`
/// (computed from `r`), never from the loss accumulator. (Mirrors `card_loss_and_grad_simd`.)
#[allow(clippy::too_many_arguments)]
pub(super) fn loss_and_grad_range_simd(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    delta_ts: &[f32], labels: &[f32], weights: &[f32], gw: &mut [f64], g_start: usize, g_end: usize,
) -> f64 {
    let wc = wconsts(w);
    let k = F::splat;
    let one = k(1.0);
    let mut caches: Vec<Step8> = Vec::with_capacity(seq_len);
    for g in g_start..g_end {
        let c0 = g * 8;
        caches.clear();
        // Per-group active length: this chunk runs to the batch-wide seq_len (= its longest card),
        // but a group of 8 length-similar cards is usually shorter. Trailing timesteps where all 8
        // lanes are padding (rating 0) are pure state passthrough (the rating==0 blend keeps state
        // and adds 0 to the gradient), so the final state — and thus the scored curve below and the
        // whole backward — are bit-for-bit identical when we stop at `sl`. (iter26's per-group skip,
        // here on the O(N^2) path.) ratings are >=1 for real reviews, so reduce_add==0 <=> all pad.
        let mut sl = seq_len;
        while sl > 1 && load8(r_hist, (sl - 1) * batch + c0).reduce_add() == 0.0 {
            sl -= 1;
        }
        let (mut s, mut d, mut sf) = (k(0.0), k(0.0), k(0.0));
        for t in 0..sl {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            let dt = if t == 0 { k(0.0) } else { load8(t_hist, base) };
            let (ns, cache) = step8_fwd::<false, true>(w, dt, rating, (s, d, sf), t == 0, &wc);
            s = ns.0;
            d = ns.1;
            sf = ns.2;
            caches.push(cache);
        }
        let dts = load8(delta_ts, c0);
        let lbl = load8(labels, c0);
        let wt = load8(weights, c0);
        let ln_s = log2_8(s);
        let ln_sf = log2_8(sf);
        let fc = curve8_fwd::<false>(w, dts, s, sf, d, &wc, ln_s, ln_sf);
        let r_raw = fc.out;
        let r = clamp8(r_raw, MIN_R, MAX_R);
        // (loss VALUE skipped — see the fn doc; training discards it, validation uses batch_loss_simd.)
        // d loss / d r, then the [MIN_R, MAX_R] clamp, in F.
        let g_r = (k(0.0) - wt) * (lbl / r - (one - lbl) / (one - r));
        let g_rraw = (r_raw.cmp_gt(k(MIN_R)) & r_raw.cmp_lt(k(MAX_R))).blend(g_r, k(0.0));
        let mut gw_g = [F::splat(0.0); 34];
        let (mut g_s, mut g_sf, mut g_d) =
            curve8_bwd(w, &fc, s, sf, d, g_rraw, F::splat(0.0), &mut gw_g, &wc);
        for t in (0..sl).rev() {
            // O(N^2) path: the only loss is the final curve (handled above), so no per-step adjoint.
            let (gs0, gd0, gsf0) =
                step8_bwd::<true>(w, &caches[t], (g_s, g_d, g_sf), F::splat(0.0), &mut gw_g, &wc);
            g_s = gs0;
            g_d = gd0;
            g_sf = gsf0;
        }
        let gg = finish_gw(&gw_g, w, &wc);
        for i in 0..34 {
            gw[i] += gg[i] as f64;
        }
    }
    0.0
}

/// Vectorized forward+backward BCE gradient for one batch (8 cards/lane). Accumulates d(loss)/d(w)
/// into `gw` (length 36) and returns 0.0 — the loss VALUE is unused by training (see
/// loss_and_grad_range_simd; validation uses batch_loss_simd). The batch must be padded to a
/// multiple of 8. This is the precision-trading (3b band) F replacement for batch_loss_and_grad.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss_and_grad_simd(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    delta_ts: &[f32], labels: &[f32], weights: &[f32], gw: &mut [f64],
) -> f64 {
    debug_assert!(batch % 8 == 0, "batch_loss_and_grad_simd needs batch padded to a multiple of 8");
    let n_groups = batch / 8;
    // Single-threaded ON PURPOSE: iter16 tried splitting the groups across the worker's 2nd pinned
    // CPU (mirroring iter3's scalar-grad threading) and REGRESSED to 0.62x. The F forward
    // saturates the physical core's vector units with one thread, so the 2nd CPU (an SMT sibling
    // sharing those units) adds no vector throughput — only spawn overhead. (iter3 helped because
    // the SCALAR grad used scalar units SMT could overlap; vectorized work can't.) The idle 2nd core
    // would need coarser-than-per-batch parallelism to pay off, which isn't worth the complexity.
    loss_and_grad_range_simd(
        w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, gw, 0, n_groups,
    )
}

// ===================== SIMD windowed O(N) forward+backward (8 cards/lane) =====================
// The O(N^2) -> O(N) expanding window. A card with K reviews became K-1 prefix-items (lengths 2..K),
// each re-running the recurrence over its whole prefix => O(K^2) timestep-work. Here each CARD is a
// single column whose recurrence runs ONCE over its full review sequence: at step t (t>=1) the curve
// curve8_fwd(state_{t-1}, delta_t[t]) it computes for the stability update IS EXACTLY the prediction
// R_t the length-(t+1) prefix used to score review t (same input state, same delta_t), so a loss is
// read off every step for free. wts/lbl are row-major [seq, bsz]; wts[t][c]==0 marks "no prediction"
// (t==0, an outlier-filtered prefix, or a padding column/timestep). The total loss and gradient equal
// the per-prefix sums (the recurrence is deterministic), so this is math-identical to the O(N^2) path
// up to FP reassociation — judged by the 3b average-log-loss band (build_host_batches groups the SAME
// cards-as-units as the Phase-1 probe, so the trained params match it modulo FP).

/// d/dr of -wt*BCE via the unified label identity: d[-ln(1-|label-r|)]/dr = -sign(label-r)/
/// (1-|label-r|). ONE division (label is 0/1); padding/filtered steps have wt==0 -> 0; the
/// [MIN_R,MAX_R] clamp zeroes the adjoint outside the range. (3b trade vs lbl/r - (1-lbl)/(1-r).)
/// A fn, not a closure: LLVM did not inline the closure (a real call per step).
/// `weights` are the windowed batches' SIGNED weights (signed_weight): sign(label - r) is +1 for
/// label 1 and -1 for label 0 (r is inside (0, 1)), so (0 - wt) * (sign / den) is exactly
/// swt * (1 / den) with swt = sign * (0 - wt) precomputed by the layout.
#[inline(always)]
pub(super) fn g_r_loss_at(r_raw: F, weights: &[f32], labels: &[f32], base: usize) -> F {
    let k = F::splat;
    let (one, z) = (k(1.0), k(0.0));
    let swt = load8(weights, base);
    let lbl = load8(labels, base);
    let r = clamp8(r_raw, MIN_R, MAX_R);
    let dd = lbl - r;
    let g_r = swt * (one / (one - dd.fast_max(z - dd)));
    (r_raw.cmp_gt(k(MIN_R)) & r_raw.cmp_lt(k(MAX_R))).blend(g_r, z)
}


/// Windowed forward + reverse-mode backward for one card-grouped batch. Accumulates d(loss)/d(w) into
/// `gw` (length 36); the loss VALUE is unused by training (only the gradient drives Adam), so the f64
/// per-lane BCE is skipped here and 0.0 is returned — validation uses `card_loss_simd`. Batch padded
/// to a multiple of 8. Training splits the groups over two threads via `card_group_grad` instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn card_loss_and_grad_simd(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    labels: &[f32], weights: &[f32], gw: &mut [f64],
) -> f64 {
    debug_assert!(batch % 8 == 0, "card_loss_and_grad_simd needs batch padded to a multiple of 8");
    let wc = wconsts(w);
    let mut caches: Vec<Step8> = Vec::with_capacity(seq_len);
    for g in 0..batch / 8 {
        let gg = card_group_grad(w, &wc, t_hist, r_hist, seq_len, batch, labels, weights, g, &mut caches);
        for i in 0..34 {
            gw[i] += gg[i] as f64;
        }
    }
    0.0
}

/// Windowed forward + backward of 8-card group `g` (columns 8g..8g+8) of a batch. Returns the
/// group's per-parameter gradient (the lane sums, f32); callers add the groups to their f64 total
/// IN GROUP ORDER, so any split of the groups over threads stays bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn card_group_grad(
    w: &[f32], wc: &WConsts, t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    labels: &[f32], weights: &[f32], g: usize, caches: &mut Vec<Step8>,
) -> [f32; 34] {
    debug_assert!(seq_len >= 2, "windowed grad needs seq_len >= 2 (min surviving prefix length is 2)");
    let k = F::splat;
    let z = k(0.0);
    let c0 = g * 8;
    // This group's own length: trailing timesteps where all 8 lanes are padding (rating 0; real
    // ratings are >= 1) carry weight 0 and only pass the state through, so they add exactly 0
    // to the gradient — skip them (bit-for-bit). Min 2 = the shortest card.
    let mut seq_len = seq_len;
    while seq_len > 2 && load8(r_hist, (seq_len - 1) * batch + c0).reduce_add() == 0.0 {
        seq_len -= 1;
    }
    let (mut s, mut d, mut sf) = (z, z, z);
    // Forward over every step EXCEPT the last (full step + cache). The last step's stability/
    // next-difficulty update feeds no t+1, so we compute its curve only (just below) — exactly
    // like card_loss_simd's validation skip-last. The first review (t==0) is peeled off, so the
    // loop body has no t==0 branch. The cache slots are reused across groups (only the first
    // seq_len - 1 are this group's).
    let n_cached = seq_len - 1;
    while caches.len() < n_cached {
        caches.push(Step8::blank());
    }
    let caches = &mut caches[..n_cached];
    (s, d, sf) = step8_fwd_into::<false, false>(w, z, load8(r_hist, c0), (s, d, sf), true, wc, &mut caches[0]);
    for (t, slot) in caches.iter_mut().enumerate().skip(1) {
        let base = t * batch + c0;
        (s, d, sf) =
            step8_fwd_into::<false, false>(w, load8(t_hist, base), load8(r_hist, base), (s, d, sf), false, wc, slot);
    }
    // Last step (t = seq_len-1, always >= 1): curve ONLY, from the incoming state (step8_fwd
    // outputs are already clamped) — the exact curve step8_fwd would compute.
    let lbase = (seq_len - 1) * batch + c0;
    let dt_last = load8(t_hist, lbase); // curve8_fwd/bwd clamp it at 0 themselves
    let fc_last = curve8_fwd::<false>(w, dt_last, s, sf, d, wc, log2_8(s), log2_8(sf));
    // Reverse pass. The final state's adjoint is 0, so at the last step the stab/next_d backward
    // all multiply 0 -> only curve8_bwd(loss-adjoint) contributes. BIT-FOR-BIT identical to a full
    // step8_bwd with g_out=0. The clamp gate uses the UNCLAMPED incoming state (= s0/d0/sf0).
    let mut gw_g = [z; 34];
    let g_rraw_last = g_r_loss_at(fc_last.out, weights, labels, lbase);
    // Ungated, like step8_bwd's input adjoints (the previous step's backward applies the mask).
    let (mut g_s, mut g_sf, mut g_d) =
        curve8_bwd(w, &fc_last, s, sf, d, g_rraw_last, z, &mut gw_g, wc);
    for (t, cache) in caches.iter().enumerate().skip(1).rev() {
        let g_r_loss = g_r_loss_at(cache.curve_out(), weights, labels, t * batch + c0);
        (g_s, g_d, g_sf) = step8_bwd::<false>(w, cache, (g_s, g_d, g_sf), g_r_loss, &mut gw_g, wc);
    }
    // init step (t==0): no prediction (min surviving prefix length is 2).
    step8_bwd::<false>(w, &caches[0], (g_s, g_d, g_sf), z, &mut gw_g, wc);
    finish_gw(&gw_g, w, wc)
}

/// Lane sums of a group's gradient bank, with the weight-only factors that the per-step backward
/// leaves out: slot 27 = sum(g_z) (-> gw[27], gw[28]), slot 26 = sum(x2) and slot 28 = the direct
/// decay2 term (-> gw[26], gw[24]), slot 25 = sum(g_lw27), slots 10 / 18 = sum(n_ln) per trace.
pub(super) fn finish_gw(gw_g: &[F; 34], w: &[f32], wc: &WConsts) -> [f32; 34] {
    // The lane sums in reduce_add's exact order (per half ((x0 + x1) + x2) + x3, as f32 sum() from
    // -0.0 gives, then low + high), four slots at a time on transposed halves instead of one slot
    // at a time through memory.
    // N = 4 (the 4-lane tail groups, see card_group_grad): one half per slot; the 8-lane sum
    // would add the empty lanes' exact zeros as the high half.
    let halves: &[f32x4] = bytemuck::cast_slice(&gw_g[..]);
    let nh = N / 4;
    let mut g = [0.0f32; 34];
    for c in (0..34).step_by(4) {
        let h = |s: usize, k: usize| if s < 34 { halves[nh * s + k] } else { f32x4::ZERO };
        let lo = f32x4::transpose([h(c, 0), h(c + 1, 0), h(c + 2, 0), h(c + 3, 0)]);
        let mut sum = ((lo[0] + lo[1]) + lo[2]) + lo[3];
        if nh == 2 {
            let hi = f32x4::transpose([h(c, 1), h(c + 1, 1), h(c + 2, 1), h(c + 3, 1)]);
            sum += ((hi[0] + hi[1]) + hi[2]) + hi[3];
        }
        for (k, v) in sum.to_array().into_iter().enumerate().take(34 - c) {
            g[c + k] = v;
        }
    }
    let (s_z, s_x2, s_dec2) = (g[27], g[26], g[28]);
    g[27] = s_z * wc.nrw27;
    g[28] = s_z * wc.rw28;
    g[26] = s_x2 * wc.k26;
    g[24] = if wc.live24 { -(s_dec2 * LN2 + s_x2 * wc.k24) } else { 0.0 };
    // Slots accumulated as g * log2(x) (the recurrence's logs are base 2): times ln2.
    for i in [8, 11, 16, 19, 29, 30, 33] {
        g[i] *= LN2;
    }
    g[25] *= wc.rw25;
    g[10] *= 1.0 / w[10];
    g[18] *= 1.0 / w[18];
    g
}

/// Windowed forward-only BCE loss (validation) — `card_loss_and_grad_simd`'s forward without the
/// backward. Emits a per-lane f64 BCE at every t>=1 with wts>0 (matching `batch_loss_simd`'s f64
/// accumulation). Batch padded to a multiple of 8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn card_loss_simd(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    labels: &[f32], weights: &[f32],
) -> f64 {
    let wc = wconsts(w);
    let k = F::splat;
    let z = k(0.0);
    let mut loss = 0.0f64;
    let n_groups = batch / 8;
    for g in 0..n_groups {
        let c0 = g * 8;
        let (mut s, mut d, mut sf) = (z, z, z);
        for t in 0..seq_len {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            let dt = if t == 0 { z } else { load8(t_hist, base) };
            // Prediction r_t = curve(state_{t-1}, dt). At the LAST timestep the state update would feed
            // no t+1, so compute the curve ONLY there (skip the 2 stab traces + next_d + ln(last_d)).
            // BIT-FOR-BIT: the BCE uses only curve.out, which is identical to step8_fwd's curve; the
            // dropped state is never read. (curve_out() for the t==0 init returns 0, unused below.)
            let r = if t == seq_len - 1 {
                let (ls, lsf, ld) =
                    (s, sf, d); // step8_fwd outputs are already clamped
                clamp8(
                    curve8_fwd::<false>(
                        w, dt.fast_max(z), ls, lsf, ld, &wc,
                        log2_8(ls), log2_8(lsf),
                    )
                    .out,
                    MIN_R, MAX_R,
                )
            } else {
                let (ns, cache) = step8_fwd::<false, true>(w, dt, rating, (s, d, sf), t == 0, &wc);
                s = ns.0;
                d = ns.1;
                sf = ns.2;
                clamp8(cache.curve_out(), MIN_R, MAX_R)
            };
            if t >= 1 {
                // Per-prediction BCE via the unified identity  -ln(1 - |label - r|)  (= -ln(r) for
                // label 1, -ln(1-r) for label 0). Branchless, so the whole 8-lane BCE is ONE vectorized
                // ln8. Padding/filtered lanes have weight 0. The BCE ln uses the accurate ::<false>,
                // as does the forward recurrence now (iter23's cruder windowed minimax was reverted).
                let lbl = load8(labels, base);
                let wt = load8(weights, base).abs(); // signed weights (signed_weight)
                let arg = k(1.0) - (lbl - r).fast_max(r - lbl); // 1 - |label - r|
                loss += ((z - wt) * ln8::<false>(arg)).reduce_add() as f64;
            }
        }
    }
    loss
}

