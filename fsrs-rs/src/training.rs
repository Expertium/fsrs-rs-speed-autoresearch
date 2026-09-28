use crate::error::Result;
use crate::model::{clip_parameters, clip_parameters_in_place, parameters_to_model, Model};
use crate::{DEFAULT_PARAMETERS, FSRSError};
use burn::LearningRate;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Progress;
use burn::lr_scheduler::LrScheduler;
use burn::nn::loss::Reduction;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::tensor::{Float, Int, Shape, Tensor, TensorData};
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainingInterrupter;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use core::marker::PhantomData;
use itertools::Itertools;
use log::info;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod training_v7 {
use crate::model::{S_MAX, S_MIN};

pub(crate) const PARAM_LEN: usize = 34;
pub(crate) const PENALTY_W_1: f64 = 0.5;
pub(crate) const PENALTY_W_2: f64 = 0.0015;
pub(crate) const PENALTY_W_L2: f64 = 0.3333;
pub(crate) const PENALTY_N_REVIEWS: usize = 10;
pub(crate) const PENALTY_TARGET_DR: f32 = 0.90;
pub(crate) const PENALTY_TARGET_DRS: [f32; 1] = [0.99];
pub(crate) const PENALTY_N_NEWTON: usize = 4;
pub(crate) const MIN_T: f32 = 1.0 / 86400.0;
pub(crate) const MAX_T: f32 = 36500.0;
pub(crate) const ONE_DAY: f32 = 1.0;
pub(crate) const SHORT_C: f32 = 600.0 / 86400.0;
pub(crate) const INV_C: f32 = 1.0 / SHORT_C;
pub(crate) const GRAD_LEN: usize = 34;
// L2 prior sigmas for the finished 34-param layout (FSRS7_L2_SIGMA_35_VALUES, fail_d_exp dropped).
// 0..3 free (9999), 4..22 difficulty/long+short stability, 23..33 forgetting curve (31 d_weight,
// 32 d_decay, 33 s_decay1 = the D-modulation / shared fast-decay curve params, stored shifted per
// the all-positive convention; 33 s_decay1 also feeds the fast-trace stability update).
pub(crate) const PARAMS_STDDEV: [f32; 34] = [
    9999.0, 9999.0, 9999.0, 9999.0, 0.523, 0.2528, 0.4329, 0.2966, 0.2139, 0.2889, 0.1862, 0.175,
    0.3812, 0.3013, 0.9104, 0.3234, 0.2448, 0.3273, 0.1842, 0.1735, 0.4608, 0.311, 0.864, 0.0418,
    0.2596, 0.0798, 0.0682, 0.1282, 0.1397, 0.1407, 0.1489, 0.2, 0.15, 0.15,
];

/// Gradient of the L2 anchor penalty (training uses only the gradient, so the penalty VALUE is not
/// computed: its 34 f64 divisions per step were dead work). The old all-zero fallback for a
/// non-finite penalty needs a non-finite w or init_w, which clip_parameters rules out.
pub(crate) fn l2_penalty_grad(
    w: &[f32],
    init_w: &[f32],
    batch_size: usize,
    total_size: usize,
    l2_weight: f64,
    params_stddev: &[f32],
) -> Vec<f32> {
    let mut grad = vec![0.0f32; w.len()];
    if total_size == 0 {
        return grad;
    }
    let size = w.len().min(init_w.len()).min(params_stddev.len());
    let scale = l2_weight * batch_size as f64 / total_size as f64;
    for i in 0..size {
        let sigma = params_stddev[i] as f64;
        let denom = sigma * sigma;
        let diff = w[i] as f64 - init_w[i] as f64;
        grad[i] = (2.0 * diff / denom * scale) as f32;
    }
    for g in &mut grad {
        if !g.is_finite() {
            *g = 0.0;
        }
    }
    grad
}

// Keep Dual35 local to FSRS-7 training: the penalty objective and its gradients
// are expressed against FSRS-7's parameter layout and index mapping.
//
// NOTE (iter-66 dual-trace port): the schedule penalty below is DISABLED by default
// (enable_sched_penalties = false in every timed path) and was NOT migrated to the
// new dual-trace layout (now 34 params) — its w-indices (e.g. the curve at 27..34, the
// transition blend at 25/26) still refer to the pre-port single-trace layout. Do
// not enable it without first reworking these indices and the stability recurrence.
#[derive(Clone, Copy, Debug)]
struct Dual35 {
    value: f64,
    grad: [f64; GRAD_LEN],
}

impl Dual35 {
    fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; GRAD_LEN],
        }
    }

    fn variable(value: f64, idx: usize) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        if idx < GRAD_LEN {
            grad[idx] = 1.0;
        }
        Self { value, grad }
    }

    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] + rhs.grad[i];
        }
        Self {
            value: self.value + rhs.value,
            grad,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] - rhs.grad[i];
        }
        Self {
            value: self.value - rhs.value,
            grad,
        }
    }

    fn neg(self) -> Self {
        self.mul_const(-1.0)
    }

    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * rhs.value + rhs.grad[i] * self.value;
        }
        Self {
            value: self.value * rhs.value,
            grad,
        }
    }

    fn div(self, rhs: Self) -> Self {
        let denom = (rhs.value * rhs.value).max(1e-18);
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = (self.grad[i] * rhs.value - self.value * rhs.grad[i]) / denom;
        }
        Self {
            value: self.value / rhs.value,
            grad,
        }
    }

    fn add_const(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            grad: self.grad,
        }
    }

    fn sub_const(self, rhs: f64) -> Self {
        Self {
            value: self.value - rhs,
            grad: self.grad,
        }
    }

    fn const_sub(self, lhs: f64) -> Self {
        self.neg().add_const(lhs)
    }

    fn mul_const(self, rhs: f64) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * rhs;
        }
        Self {
            value: self.value * rhs,
            grad,
        }
    }

    fn div_const(self, rhs: f64) -> Self {
        self.mul_const(1.0 / rhs)
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * value;
        }
        Self { value, grad }
    }

    fn log(self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] / self.value;
        }
        Self {
            value: self.value.ln(),
            grad,
        }
    }

    fn powf(self, exp: f64) -> Self {
        let value = self.value.powf(exp);
        let coeff = exp * self.value.powf(exp - 1.0);
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * coeff;
        }
        Self { value, grad }
    }

    fn powi(self, exp: i32) -> Self {
        let value = self.value.powi(exp);
        let coeff = (exp as f64) * self.value.powi(exp - 1);
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * coeff;
        }
        Self { value, grad }
    }

    fn pow(self, exp: Self) -> Self {
        let base = self.clamp_min(1e-12);
        exp.mul(base.log()).exp()
    }

    fn clamp_min(self, min: f64) -> Self {
        if self.value < min {
            Self::constant(min)
        } else {
            self
        }
    }

    fn clamp_max(self, max: f64) -> Self {
        if self.value > max {
            Self::constant(max)
        } else {
            self
        }
    }

    fn clamp(self, min: f64, max: f64) -> Self {
        self.clamp_min(min).clamp_max(max)
    }

    fn min(self, rhs: Self) -> Self {
        if self.value <= rhs.value { self } else { rhs }
    }

    fn max(self, rhs: Self) -> Self {
        if self.value >= rhs.value { self } else { rhs }
    }
}

fn dual_weights(w: &[f32]) -> [Dual35; GRAD_LEN] {
    std::array::from_fn(|i| Dual35::variable(w[i] as f64, i))
}

fn fsrs7_fc_r_and_drdt_scalar(t: f64, s: f64, w: &[f32]) -> (f64, f64) {
    let s_safe = s.max(1e-12);
    let decay1 = -(w[27] as f64);
    let decay2 = -(w[28] as f64);
    let base1 = (w[29] as f64).max(1e-4);
    let base2 = (w[30] as f64).max(1e-4);
    let bw1 = (w[31] as f64).max(1e-4);
    let bw2 = (w[32] as f64).max(1e-4);
    let swp1 = w[33] as f64;
    let swp2 = w[34] as f64;

    let c1 = base1.powf(1.0 / decay1) - 1.0;
    let c2 = base2.powf(1.0 / decay2) - 1.0;
    let tos = t / s_safe;
    let inner1 = (1.0 + c1 * tos).max(1e-9);
    let inner2 = (1.0 + c2 * tos).max(1e-9);
    let r1 = inner1.powf(decay1);
    let r2 = inner2.powf(decay2);

    let wt1 = bw1 * s_safe.powf(-swp1);
    let wt2 = bw2 * s_safe.powf(swp2);
    let wt_sum = (wt1 + wt2).max(1e-9);
    let r = ((wt1 * r1 + wt2 * r2) / wt_sum).clamp(0.0, 1.0);

    let dr1_dt = decay1 * inner1.powf(decay1 - 1.0) * (c1 / s_safe);
    let dr2_dt = decay2 * inner2.powf(decay2 - 1.0) * (c2 / s_safe);
    let dr_dt = ((wt1 * dr1_dt + wt2 * dr2_dt) / wt_sum).clamp(-1e9, 0.0);
    (r, dr_dt)
}

fn fsrs7_fc_r_dual(t: Dual35, s: Dual35, w: &[Dual35]) -> Dual35 {
    let decay1 = w[27].neg();
    let decay2 = w[28].neg();
    let base1 = w[29].clamp_min(1e-4);
    let base2 = w[30].clamp_min(1e-4);
    let bw1 = w[31].clamp_min(1e-4);
    let bw2 = w[32].clamp_min(1e-4);
    let swp1 = w[33];
    let swp2 = w[34];

    let c1 = base1.pow(decay1.powi(-1)).sub_const(1.0);
    let c2 = base2.pow(decay2.powi(-1)).sub_const(1.0);
    let tos = t.div(s);
    let inner1 = c1.mul(tos).add_const(1.0).clamp_min(1e-9);
    let inner2 = c2.mul(tos).add_const(1.0).clamp_min(1e-9);

    let r1 = inner1.pow(decay1);
    let r2 = inner2.pow(decay2);

    let wt1 = bw1.mul(s.pow(swp1.neg()));
    let wt2 = bw2.mul(s.pow(swp2));
    let wt_sum = wt1.add(wt2).clamp_min(1e-9);
    wt1.mul(r1).add(wt2.mul(r2)).div(wt_sum).clamp(0.0, 1.0)
}

fn fsrs7_init_d_dual(rating: f64, w: &[Dual35]) -> Dual35 {
    w[4].sub(w[5].mul_const(rating - 1.0).exp())
        .add_const(1.0)
        .clamp(1.0, 10.0)
}

fn fsrs7_next_d_good_dual(d: Dual35, init_d4: Dual35) -> Dual35 {
    init_d4
        .mul_const(0.01)
        .add(d.mul_const(0.99))
        .clamp(1.0, 10.0)
}

fn fsrs7_s_fail_long_dual(s: Dual35, d: Dual35, r: Dual35, w: &[Dual35]) -> Dual35 {
    let raw = w[10]
        .mul(d.pow(w[11].neg()))
        .mul(s.add_const(1.0).pow(w[12]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[13]).exp());
    s.min(raw)
}

fn fsrs7_s_fail_short_dual(s: Dual35, d: Dual35, r: Dual35, w: &[Dual35]) -> Dual35 {
    let raw = w[19]
        .mul(d.pow(w[20].neg()))
        .mul(s.add_const(1.0).pow(w[21]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[22]).exp());
    s.min(raw)
}

fn fsrs7_next_s_good_dual(s: Dual35, d: Dual35, delta_t: Dual35, w: &[Dual35]) -> Dual35 {
    let r = fsrs7_fc_r_dual(delta_t, s, w).clamp(0.0001, 0.9999);

    let sf_l = fsrs7_s_fail_long_dual(s, d, r, w);
    let si_l = w[7]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[8].neg()))
        .mul(
            r.const_sub(1.0)
                .mul(w[9])
                .clamp_max(30.0)
                .exp()
                .sub_const(1.0),
        )
        .add_const(1.0);
    let s_lng = sf_l.max(s.mul(si_l));

    let sf_sh = fsrs7_s_fail_short_dual(s, d, r, w);
    let si_sh = w[16]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[17].neg()))
        .mul(
            r.const_sub(1.0)
                .mul(w[18])
                .clamp_max(30.0)
                .exp()
                .sub_const(1.0),
        )
        .add_const(1.0);
    let s_sht = sf_sh.max(s.mul(si_sh));

    let coef = Dual35::constant(1.0)
        .sub(w[26].mul(w[25].neg().mul(delta_t).exp()))
        .clamp(0.0, 1.0);
    coef.mul(s_lng)
        .add(Dual35::constant(1.0).sub(coef).mul(s_sht))
        .clamp(S_MIN as f64, S_MAX as f64)
}

fn fsrs7_interval_differentiable_dual(
    s: Dual35,
    target: f64,
    n_newton: usize,
    w: &[f32],
    w_dual: &[Dual35],
) -> Dual35 {
    let s_f = s.value.max(1e-10);
    let d1 = -(w[27] as f64);
    let d2 = -(w[28] as f64);
    let b1 = (w[29] as f64).max(1e-4);
    let b2 = (w[30] as f64).max(1e-4);
    let bw1 = (w[31] as f64).max(1e-4);
    let bw2 = (w[32] as f64).max(1e-4);
    let sw1 = w[33] as f64;
    let sw2 = w[34] as f64;

    let c1 = b1.powf(1.0 / d1) - 1.0;
    let c2 = b2.powf(1.0 / d2) - 1.0;
    let wt1 = bw1 * s_f.powf(-sw1);
    let wt2 = bw2 * s_f.powf(sw2);
    let wts = (wt1 + wt2).max(1e-9);

    let mut u = s_f.ln();
    for _ in 0..n_newton {
        u = u.clamp((MIN_T as f64).ln(), (MAX_T as f64).ln());
        let t = u.exp().clamp(MIN_T as f64, MAX_T as f64);
        let tos = t / s_f;
        let i1 = (1.0 + c1 * tos).max(1e-9);
        let i2 = (1.0 + c2 * tos).max(1e-9);
        let r = (wt1 * i1.powf(d1) + wt2 * i2.powf(d2)) / wts;
        let dr1 = d1 * i1.powf(d1 - 1.0) * c1 / s_f;
        let dr2 = d2 * i2.powf(d2 - 1.0) * c2 / s_f;
        let drdt = (wt1 * dr1 + wt2 * dr2) / wts;
        let dfdu = (drdt * t).min(-1e-12);
        u -= (r - target) / dfdu;
    }

    let t_star = u.exp().clamp(MIN_T as f64, MAX_T as f64);
    let residual = fsrs7_fc_r_dual(Dual35::constant(t_star), s, w_dual).sub_const(target);
    let (_, drdt_s) = fsrs7_fc_r_and_drdt_scalar(t_star, s.value, w);
    let dfdu_s = (drdt_s * t_star).clamp(-1e9, -1e-9);
    Dual35::constant(t_star.ln())
        .sub(residual.div_const(dfdu_s))
        .clamp((MIN_T as f64).ln(), (MAX_T as f64).ln())
        .exp()
}

fn fsrs7_interval_growth_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35],
    n_reviews: usize,
    target_dr: f64,
    n_newton: usize,
) -> Dual35 {
    let mut s = w_dual[2].clamp(S_MIN as f64, S_MAX as f64);
    let init_d4 = fsrs7_init_d_dual(4.0, w_dual);
    let mut d = fsrs7_init_d_dual(3.0, w_dual);
    let mut prev_interval: Option<Dual35> = None;
    let mut best_ratio: Option<Dual35> = None;
    let mut best_val = f64::NEG_INFINITY;
    for _ in 0..n_reviews {
        let t = fsrs7_interval_differentiable_dual(s, target_dr, n_newton, w, w_dual);
        if let Some(prev) = prev_interval {
            if prev.value >= ONE_DAY as f64 {
                let ratio = t.div(prev);
                if ratio.value > best_val {
                    best_val = ratio.value;
                    best_ratio = Some(ratio);
                }
            }
        }
        prev_interval = Some(t);
        s = fsrs7_next_s_good_dual(s, d, t, w_dual);
        d = fsrs7_next_d_good_dual(d, init_d4);
    }
    if let Some(ratio) = best_ratio {
        ratio.powf(2.0)
    } else {
        Dual35::constant(0.0)
    }
}

fn fsrs7_short_interval_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35],
    n_reviews: usize,
    n_newton: usize,
    target_drs: &[f32],
) -> Dual35 {
    let mut penalty_sum = Dual35::constant(0.0);
    let mut penalty_count = 0usize;
    for &target_dr in target_drs {
        let mut s = w_dual[2].clamp(S_MIN as f64, S_MAX as f64);
        let init_d4 = fsrs7_init_d_dual(4.0, w_dual);
        let mut d = fsrs7_init_d_dual(3.0, w_dual);
        let mut short_sum = Dual35::constant(0.0);
        let mut short_count = 0usize;
        for _ in 0..n_reviews {
            let t = fsrs7_interval_differentiable_dual(s, target_dr as f64, n_newton, w, w_dual);
            if t.value < ONE_DAY as f64 {
                short_sum = short_sum.add(t);
                short_count += 1;
            }
            s = fsrs7_next_s_good_dual(s, d, t, w_dual);
            d = fsrs7_next_d_good_dual(d, init_d4);
        }
        if short_count == 0 {
            continue;
        }
        let avg_t = short_sum
            .div_const(short_count as f64)
            .clamp_min(MIN_T as f64);
        let inv_x = avg_t.powf(-1.0);
        let penalty = inv_x.clamp_min(INV_C as f64).sub_const(INV_C as f64);
        penalty_sum = penalty_sum.add(penalty);
        penalty_count += 1;
    }
    if penalty_count == 0 {
        Dual35::constant(0.0)
    } else {
        penalty_sum.div_const(penalty_count as f64)
    }
}

pub(crate) fn schedule_penalty_value_and_grad(
    w: &[f32],
    batch_size: usize,
) -> (f64, [f64; GRAD_LEN]) {
    if w.len() < PARAM_LEN {
        return (0.0, [0.0; GRAD_LEN]);
    }
    let w_dual = dual_weights(w);
    let mut p1 = fsrs7_interval_growth_penalty_dual(
        w,
        &w_dual,
        PENALTY_N_REVIEWS,
        PENALTY_TARGET_DR as f64,
        PENALTY_N_NEWTON,
    );
    if !p1.value.is_finite() {
        p1 = Dual35::constant(0.0);
    }
    let mut p2 = fsrs7_short_interval_penalty_dual(
        w,
        &w_dual,
        PENALTY_N_REVIEWS,
        PENALTY_N_NEWTON,
        &PENALTY_TARGET_DRS,
    );
    if !p2.value.is_finite() {
        p2 = Dual35::constant(0.0);
    }
    let penalty = p1
        .mul_const(PENALTY_W_1)
        .add(p2.mul_const(PENALTY_W_2))
        .mul_const(batch_size as f64);
    if !penalty.value.is_finite() {
        return (0.0, [0.0; GRAD_LEN]);
    }
    let mut grad = penalty.grad;
    for g in &mut grad {
        if !g.is_finite() {
            *g = 0.0;
        }
    }
    (penalty.value, grad)
}

pub(crate) fn maybe_schedule_penalty_value_and_grad(
    w: &[f32],
    batch_size: usize,
    enable_sched_penalties: bool,
) -> (f64, [f64; GRAD_LEN]) {
    if enable_sched_penalties {
        schedule_penalty_value_and_grad(w, batch_size)
    } else {
        (0.0, [0.0; GRAD_LEN])
    }
}

}

type B = NdArray<f32>;

const L2_PENALTY_WEIGHT: f64 = training_v7::PENALTY_W_L2;
const PENALTY_GRAD_LEN: usize = training_v7::GRAD_LEN;

// Adam hyperparameters — MUST equal the AdamConfig built in compute_parameters()/benchmark().
// These are the per-cell HP tune's (9,512)-gold betas (2026-06-14, hp_grid_stage3.json; beta2
// rounded 0.9804->0.98, the 4e-4 is within noise — Andrew); keep
// this single copy in sync with both AdamConfig sites. Used by the hand-rolled host Adam in
// train(), which replaced burn's tensor optimizer with an element-wise replica of burn 0.17's
// AdaptiveMomentum.
const ADAM_BETA1: f32 = 0.70;
const ADAM_BETA2: f32 = 0.98;
const ADAM_EPS: f32 = 1e-8;

type SchedulePenaltyFn = fn(&[f32], usize, bool) -> (f64, [f64; PENALTY_GRAD_LEN]);
type L2PenaltyFn = fn(&[f32], &[f32], usize, usize, f64, &[f32]) -> Vec<f32>;

fn schedule_penalty_fn() -> SchedulePenaltyFn {
    training_v7::maybe_schedule_penalty_value_and_grad
}

fn l2_penalty_fn() -> L2PenaltyFn {
    training_v7::l2_penalty_grad
}

// ========== ModelConfig ==========

#[derive(Config, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    pub initial_forgetting_curve: Option<[f32; 8]>,
    #[config(default = false)]
    pub freeze_short_term_stability: bool,
    #[config(default = 1)]
    pub num_relearning_steps: usize,
}

// ========== CosineAnnealingLR ==========

#[derive(Clone, Debug)]
pub(crate) struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: LearningRate,
    step_count: f64,
    current_lr: LearningRate,
}

impl CosineAnnealingLR {
    pub const fn init(t_max: f64, init_lr: LearningRate) -> Self {
        Self {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: -1.0,
            current_lr: init_lr,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.step_count += 1.0;
        use std::f64::consts::PI;
        fn cosine_annealing_lr(
            init_lr: LearningRate,
            lr: LearningRate,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> LearningRate {
            if step_count == 0.0 {
                init_lr
            } else if (step_count - 1.0 - t_max) % (2.0 * t_max) == 0.0 {
                (init_lr - eta_min) * (1.0 - f64::cos(PI / t_max)) / 2.0
            } else {
                ((1.0 + f64::cos(PI * step_count / t_max))
                    / (1.0 + f64::cos(PI * (step_count - 1.0) / t_max)))
                .mul_add(lr - eta_min, eta_min)
            }
        }
        self.current_lr = cosine_annealing_lr(
            self.init_lr,
            self.current_lr,
            self.step_count,
            self.t_max,
            self.eta_min,
        );
        self.current_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step_count as usize
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.step_count = record as LearningRate;
        self
    }
}

// ========== (parameter clipping is now inlined into train()'s host Adam via clip_parameters) ==========

// ========== FSRSItem and FSRSReview ==========

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
/// When used during review, the last item should include the correct delta_t, but
/// the provided rating is ignored as all four ratings are returned by .next_states()
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct FSRSReview {
    /// 1-4
    pub rating: u32,
    /// The number of days that passed (can be fractional).
    /// # Warning
    /// `delta_t` for item first(initial) review must be 0
    pub delta_t: f32,
}

const LONG_TERM_DELTA_T_BUCKET_DAYS: f32 = 1.0;

pub(crate) fn bucket_long_term_delta_t(delta_t: f32) -> f32 {
    if !delta_t.is_finite() {
        return 1.0;
    }
    let clamped = delta_t.max(1.0);
    (clamped / LONG_TERM_DELTA_T_BUCKET_DAYS).floor() * LONG_TERM_DELTA_T_BUCKET_DAYS
}

impl FSRSItem {
    // The previous reviews done before the current one.
    pub(crate) fn history(&self) -> impl Iterator<Item = &FSRSReview> {
        self.reviews.iter().take(self.reviews.len() - 1)
    }

    pub(crate) fn current(&self) -> &FSRSReview {
        self.reviews.last().unwrap()
    }

    pub fn long_term_review_cnt(&self) -> usize {
        self.reviews
            .iter()
            .filter(|review| review.delta_t >= 1.0)
            .count()
    }

    pub(crate) fn first_long_term_review(&self) -> FSRSReview {
        *self
            .reviews
            .iter()
            .find(|review| review.delta_t >= 1.0)
            .expect("Invalid FSRS item: at least one review with delta_t >= 1.0 is required")
    }

    pub(crate) fn r_matrix_index(&self) -> (u32, u32, u32) {
        let delta_t = self.current().delta_t as f64;
        let delta_t_bin = (2.48 * 3.62f64.powf(delta_t.log(3.62).floor()) * 100.0).round() as u32;
        let length = self.long_term_review_cnt() as f64 + 1.0;
        let length_bin = (1.99 * 1.89f64.powf(length.log(1.89).floor())).round() as u32;
        let lapse = self
            .history()
            .filter(|review| review.rating == 1 && review.delta_t >= 1.0)
            .count();
        if lapse == 0 {
            return (delta_t_bin, length_bin, 0);
        }
        let lapse_bin = (1.65 * 1.73f64.powf((lapse as f64).log(1.73).floor())).round() as u32;
        (delta_t_bin, length_bin, lapse_bin)
    }
}

/// OUTLIER FILTER — DEPLOYMENT vs BENCHMARK (Andrew, 2026-06-14).
/// The (rating, delta_t)-bucket outlier filter drops reviews from rare/sparse first-interval buckets.
/// It is needed ONLY for parity with the `srs-benchmark` repo (where every algorithm filters the
/// same outliers so the comparison is fair). In DEPLOYMENT — i.e. real Anki — the optimizer is NOT
/// filtered (Anki trains on every review), and a 3k ablation measured the filter COSTING +0.000397
/// cross-val log loss. So this fork (which optimizes for the Anki user) keeps the filter OFF by
/// DEFAULT; the timed/shipped path is bit-for-bit with the filter-removed state. Set the env var
/// `FSRS_OUTLIER_FILTER` (any value) to ENABLE it when running this crate under srs-benchmark.
fn outlier_filter_enabled() -> bool {
    std::env::var_os("FSRS_OUTLIER_FILTER").is_some()
}

/// Compute the outlier-removed (rating, delta_t) buckets from the initialization subset. The old
/// `filtered_items` init set is DEAD (every caller of prepare_training_data{,_carded} discards it),
/// and the threshold logic only ever reads each bucket's SIZE — so we take the init candidates by
/// REFERENCE and tally per-(rating, delta_t-bucket) COUNTS, cloning nothing and storing no items.
/// Split out so the card-aware path reuses the exact same `removed_pairs`.
fn compute_outlier_removed_pairs<'a>(
    dataset_for_initialization: impl Iterator<Item = &'a FSRSItem>,
) -> [HashSet<u32>; 5] {
    let to_key = |delta_t: f32| bucket_long_term_delta_t(delta_t).to_bits();
    let from_key = |key: u32| f32::from_bits(key);
    let mut groups = HashMap::<u32, HashMap<u32, usize>>::new();

    for item in dataset_for_initialization {
        let first_review = item.reviews.first().unwrap();
        let first_long_term_review = item.first_long_term_review();
        *groups
            .entry(first_review.rating)
            .or_default()
            .entry(to_key(first_long_term_review.delta_t))
            .or_default() += 1;
    }

    let mut removed_pairs: [HashSet<u32>; 5] = Default::default();

    for (rating, delta_t_groups) in groups.into_iter().sorted_by_key(|&(k, _)| k) {
        let mut sub_groups = delta_t_groups.into_iter().collect::<Vec<_>>();

        // sort by bucket COUNT desc, then delta_t desc (count == the old sub_group.len()).
        sub_groups.sort_by(|(delta_t_a, cnt_a), (delta_t_b, cnt_b)| {
            cnt_b
                .cmp(cnt_a)
                .then(from_key(*delta_t_b).total_cmp(&from_key(*delta_t_a)))
        });

        let total = sub_groups.iter().map(|(_, cnt)| *cnt).sum::<usize>();
        let mut has_been_removed = 0;

        // Each bucket is either KEPT in the (vestigial) init set or its (rating, delta_t) pair is
        // flagged as an outlier in `removed_pairs`. Identical control flow to the original, on counts.
        for (delta_t, cnt) in sub_groups.iter().rev() {
            let over_threshold = has_been_removed + *cnt >= 20.max(total / 20);
            let kept_in_init_set = over_threshold
                && *cnt >= 6
                && from_key(*delta_t) <= if rating != 4 { 100.0 } else { 365.0 };
            if !kept_in_init_set {
                removed_pairs[rating as usize].insert(*delta_t);
            }
            if !over_threshold {
                has_been_removed += *cnt;
            }
        }
    }
    removed_pairs
}

/// The retain predicate: keep an item unless its (rating, first-long-term-delta_t) bucket was
/// flagged as an outlier.
fn item_survives_outlier(item: &FSRSItem, removed_pairs: &[HashSet<u32>; 5]) -> bool {
    if item.long_term_review_cnt() == 0 {
        true
    } else {
        let key = bucket_long_term_delta_t(item.first_long_term_review().delta_t).to_bits();
        !removed_pairs[item.reviews[0].rating as usize].contains(&key)
    }
}

// ========== Dataset Types ==========

#[derive(Debug, Clone)]
pub(crate) struct WeightedFSRSItem {
    pub weight: f32,
    pub item: FSRSItem,
}

#[derive(Clone)]
pub(crate) struct FSRSBatcher<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> FSRSBatcher<B> {
    pub const fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FSRSBatch<B: Backend> {
    pub t_historys: Tensor<B, 2, Float>,
    pub r_historys: Tensor<B, 2, Float>,
    pub delta_ts: Tensor<B, 1, Float>,
    pub labels: Tensor<B, 1, Int>,
    pub weights: Tensor<B, 1, Float>,
}

impl<B: Backend> Batcher<B, WeightedFSRSItem, FSRSBatch<B>> for FSRSBatcher<B> {
    fn batch(&self, weighted_items: Vec<WeightedFSRSItem>, device: &B::Device) -> FSRSBatch<B> {
        let pad_size = weighted_items
            .iter()
            .map(|x| x.item.reviews.len())
            .max()
            .expect("FSRSItem is empty")
            - 1;

        let (time_histories, rating_histories) = weighted_items
            .iter()
            .map(|weighted_item| {
                let (mut delta_t, mut rating): (Vec<_>, Vec<_>) = weighted_item
                    .item
                    .history()
                    .map(|r| (r.delta_t, r.rating))
                    .unzip();
                delta_t.resize(pad_size, 0.0);
                rating.resize(pad_size, 0);
                let delta_t = Tensor::<B, 2>::from_floats(
                    TensorData::new(
                        delta_t,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    device,
                );
                let rating = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        rating,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    device,
                );
                (delta_t, rating)
            })
            .unzip();

        let (delta_ts, labels, weights) = weighted_items
            .iter()
            .map(|weighted_item| {
                let current = weighted_item.item.current();
                let delta_t: Tensor<B, 1> = Tensor::from_floats([current.delta_t], device);
                let label = match current.rating {
                    1 => 0,
                    _ => 1,
                };
                let label: Tensor<B, 1, Int> = Tensor::from_ints([label], device);
                let weight: Tensor<B, 1> = Tensor::from_floats([weighted_item.weight], device);
                (delta_t, label, weight)
            })
            .multiunzip();

        let t_historys = Tensor::cat(time_histories, 0).transpose().to_device(device);
        let r_historys = Tensor::cat(rating_histories, 0)
            .transpose()
            .to_device(device);
        let delta_ts = Tensor::cat(delta_ts, 0).to_device(device);
        let labels = Tensor::cat(labels, 0).to_device(device);
        let weights = Tensor::cat(weights, 0).to_device(device);

        FSRSBatch {
            t_historys,
            r_historys,
            delta_ts,
            labels,
            weights,
        }
    }
}

// The training/validation batches are built directly as host arrays by build_host_batches()
// (below) — no burn-tensor dataset/dataloader layer. FSRSBatcher/FSRSBatch above are retained
// only because the frozen evaluate() path (inference.rs) still uses them.

// ========== Dataset Helper Functions ==========

pub(crate) fn sort_items_by_review_length(
    mut weighted_items: Vec<WeightedFSRSItem>,
) -> Vec<WeightedFSRSItem> {
    weighted_items.sort_by_cached_key(|weighted_item| weighted_item.item.reviews.len());
    weighted_items
}

/// The input items should be sorted by the review timestamp.
pub(crate) fn recency_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    // Denominator is the train-set size n (NOT n-1), matching CUDA gradient_weight:
    // review_ord_lin = review_ord / training_set_size, review_ord 0-based (0..n-1).
    let length = (items.len() as f32).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            // Finished FSRS-7 recency weighting: C0 + (1 - C0) * (idx/n)^EXP,
            // C0 = 0.0667 (<=4dp), EXP = 11.25 (fsrs_v7_constants RECENCY_C0 / RECENCY_EXP).
            weight: 0.0667 + 0.9333 * (idx as f32 / length).powf(11.25),
            item,
        })
        .collect()
}

/// TUNER recency override (hp_tune per-cell fine tune). Same formula as
/// `recency_weighted_fsrs_items` but reads the floor weight C0 and ramp exponent EXP from
/// FSRS_RECENCY_C0 / FSRS_RECENCY_EXP, defaulting to the shipped constants (0.0667 / 11.25) so
/// production (neither var set) is bit-for-bit identical. Used ONLY by the windowed
/// compute_parameters training path — the frozen evaluate() keeps calling the plain version, so
/// the scorer is untouched (constraint 11). Returns weight(idx, n) for item idx of n.
fn recency_weight_tuned() -> impl Fn(usize, f32) -> f32 {
    let c0_env = std::env::var("FSRS_RECENCY_C0")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok());
    let exp_env = std::env::var("FSRS_RECENCY_EXP")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok());
    // No override set -> the exact (literal-constant) production formula, so the shipped no-env
    // training is byte-identical to recency_weighted_fsrs_items (no (1.0 - c0)-vs-0.9333 ULP drift).
    let tuned = (c0_env.is_some() || exp_env.is_some())
        .then(|| (c0_env.unwrap_or(0.0667), exp_env.unwrap_or(11.25)));
    move |idx, length| match tuned {
        None => 0.0667 + 0.9333 * (idx as f32 / length).powf(11.25),
        Some((c0, exp)) => c0 + (1.0 - c0) * (idx as f32 / length).powf(exp),
    }
}

pub(crate) fn recency_weighted_fsrs_items_tuned(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    let weight = recency_weight_tuned();
    let length = (items.len() as f32).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem { weight: weight(idx, length), item })
        .collect()
}

pub(crate) fn prepare_training_data(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    // Outlier filter OFF by default (Anki deployment / this fork) -> train on every item, bit-for-bit
    // with the filter-removed state. ON only under FSRS_OUTLIER_FILTER (srs-benchmark parity); see
    // the note above `outlier_filter_enabled`. The vestigial init-set slot stays for signature compat.
    if !outlier_filter_enabled() {
        return (Vec::new(), items);
    }
    let removed_pairs =
        compute_outlier_removed_pairs(items.iter().filter(|item| item.long_term_review_cnt() == 1));
    let trainset = items
        .into_iter()
        .filter(|item| item_survives_outlier(item, &removed_pairs))
        .collect();
    (Vec::new(), trainset)
}

/// Card-aware twin of `prepare_training_data` (windowed path): same outlier toggle, applied to the
/// `items` AND their parallel `card_ids` together (order-preserving) so each kept prefix keeps its
/// card id. Default (filter OFF) passes everything through unchanged (bit-for-bit).
pub(crate) fn prepare_training_data_carded(
    items: Vec<FSRSItem>,
    card_ids: Vec<i64>,
) -> (Vec<FSRSItem>, Vec<FSRSItem>, Vec<i64>) {
    if !outlier_filter_enabled() {
        return (Vec::new(), items, card_ids);
    }
    let removed_pairs =
        compute_outlier_removed_pairs(items.iter().filter(|item| item.long_term_review_cnt() == 1));
    let mut trainset = Vec::with_capacity(items.len());
    let mut trainset_card_ids = Vec::with_capacity(items.len());
    for (item, cid) in items.into_iter().zip(card_ids) {
        if item_survives_outlier(&item, &removed_pairs) {
            trainset.push(item);
            trainset_card_ids.push(cid);
        }
    }
    (Vec::new(), trainset, trainset_card_ids)
}

// ========== BCE Loss ==========

pub struct BCELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
    pub fn forward(
        &self,
        retrievability: Tensor<B, 1>,
        labels: Tensor<B, 1>,
        weights: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss = (labels.clone() * retrievability.clone().log()
            + (-labels + 1) * (-retrievability + 1).log())
            * weights.clone();
        match mean {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => (loss.sum() / weights.sum()).neg(),
        }
    }
}

// ========== Progress Types ==========

#[derive(Debug, Default, Clone)]
pub struct ProgressState {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
}

#[derive(Debug, Default)]
pub struct CombinedProgressState {
    pub want_abort: bool,
    pub splits: Vec<ProgressState>,
    finished: bool,
}

impl CombinedProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub fn current(&self) -> usize {
        self.splits.iter().map(|s| s.current()).sum()
    }

    pub fn total(&self) -> usize {
        self.splits.iter().map(|s| s.total()).sum()
    }

    pub const fn finished(&self) -> bool {
        self.finished
    }
}

#[derive(Clone)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<CombinedProgressState>>,
    pub interrupter: TrainingInterrupter,
    /// The index of the split we should update.
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self {
            state,
            interrupter: Default::default(),
            index,
        }
    }
}

impl ProgressState {
    pub const fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub const fn total(&self) -> usize {
        self.epoch_total * self.items_total
    }
}

impl MetricsRenderer for ProgressCollector {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = item.epoch;
        split.epoch_total = item.epoch_total;
        split.items_processed = item.progress.items_processed;
        split.items_total = item.progress.items_total;
        if info.want_abort {
            self.interrupter.stop();
        }
    }

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

// ========== Training Config ==========

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = false)]
    pub enable_sched_penalties: bool,
    #[config(default = 9)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 2023)]
    pub seed: u64,
    #[config(default = 0.0118)]
    pub learning_rate: f64,
    #[config(default = 1024)]
    pub max_seq_len: usize,
}

// ========== ComputeParametersInput ==========

/// Input parameters for computing FSRS parameters
#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    /// The training set containing review history
    pub train_set: Vec<FSRSItem>,
    /// Optional progress tracking
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    /// Whether to enable short-term memory parameters
    pub enable_short_term: bool,
    /// Whether to enable FSRS-7 schedule penalties (penalty 1 & 2)
    pub enable_sched_penalties: bool,
    /// Number of relearning steps
    pub num_relearning_steps: Option<usize>,
    /// Optional per-item card ids, parallel to `train_set` (same order). When present, training
    /// groups each card's expanding-window prefix-items into the same mini-batch (the O(N) window
    /// path); when `None`, training is unchanged (each prefix-item batched independently).
    pub card_ids: Option<Vec<i64>>,
    /// DIAGNOSTIC-ONLY epoch override (default `None` = 8, the shipped count per constraint 4). Set
    /// by the profiling epoch-buyback sweep (`profiling/epoch_sweep.py`) to measure accuracy-vs-epochs;
    /// the shipped Python path never sets it, so production training stays at 8 epochs.
    pub num_epochs: Option<usize>,
    /// Optional custom initial parameters = SGD start AND L2 anchor (default `None` = the compiled
    /// `DEFAULT_PARAMETERS`, BIT-FOR-BIT). Used by the gated default-param meta-optimizer (the gate
    /// trains each user with the candidate-default as both init and anchor — see fsrs-default-param-tuner).
    pub init_w: Option<Vec<f32>>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
            card_ids: None,
            num_epochs: None,
            init_w: None,
        }
    }
}

fn normalize_training_set(train_set: Vec<FSRSItem>) -> Vec<FSRSItem> {
    train_set
        .into_iter()
        .map(|mut item| {
            for review in &mut item.reviews {
                review.delta_t = review.delta_t.max(0.0);
            }
            item
        })
        .collect()
}

/// Computes optimized parameters for the FSRS model based on training data.
///
/// This function trains the model on the provided dataset and returns optimized parameters.
///
/// # Arguments
/// * `input` - Input parameters including the training dataset and configuration
///
/// # Returns
/// A `Result<Vec<f32>>` containing the optimized parameters
pub fn compute_parameters(
    ComputeParametersInput {
        train_set,
        progress,
        enable_short_term,
        enable_sched_penalties,
        num_relearning_steps,
        card_ids,
        num_epochs,
        init_w,
        ..
    }: ComputeParametersInput,
) -> Result<Vec<f32>> {
    let finish_progress = || {
        if let Some(progress) = &progress {
            progress.lock().unwrap().finished = true;
        }
    };

    // The windowed builder clamps delta_t itself for the only reviews it reads (each card's longest
    // prefix), so the O(sum of prefix lengths) normalize pass is needed only by the outlier filter
    // and the plain (no card ids) path.
    let train_set = if card_ids.is_none() || outlier_filter_enabled() {
        normalize_training_set(train_set)
    } else {
        train_set
    };
    // Carry card ids through the (order-preserving) outlier retain when they are supplied so the
    // expanding-window prefix-items can be grouped per card downstream; otherwise behave exactly
    // as before.
    let (train_set, train_card_ids) = match card_ids {
        Some(card_ids) => {
            let (_, train_set, train_card_ids) = prepare_training_data_carded(train_set, card_ids);
            (train_set, Some(train_card_ids))
        }
        None => {
            let (_, train_set) = prepare_training_data(train_set);
            (train_set, None)
        }
    };
    // SGD start + L2 anchor: a non-empty custom init_w (gated default-param tuner) overrides the
    // compiled DEFAULT_PARAMETERS; None/empty keeps DEFAULT_PARAMETERS, so the shipped path stays
    // BIT-FOR-BIT (the two early-return tiny-user cases also returned DEFAULT_PARAMETERS before).
    let initialized_parameters = match &init_w {
        Some(w) if !w.is_empty() => w.clone(),
        _ => DEFAULT_PARAMETERS.to_vec(),
    };
    if train_set.len() < 8 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    if train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let mut config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.70)
            .with_beta_2(0.98)
            .with_epsilon(1e-8),
    )
    .with_enable_sched_penalties(enable_sched_penalties);
    // DIAGNOSTIC epoch override for the buyback sweep; default (None) keeps the shipped 8 epochs.
    // Bumping it also extends the cosine-annealing schedule over the larger iteration count below.
    if let Some(ne) = num_epochs {
        config.num_epochs = ne;
    }
    // TUNER operating-point overrides (hp_tune.py epoch x batch grid). These let the tuner sweep
    // (n_epoch, batch_size) per cell via env vars with NO rebuild — the shipped Python path sets
    // neither var, so production training keeps the defaults (8 epochs / batch 256). The explicit
    // num_epochs arg above still wins when set; FSRS_N_EPOCHS is only a fallback for it.
    if num_epochs.is_none() {
        if let Some(ne) = std::env::var("FSRS_N_EPOCHS").ok().and_then(|s| s.trim().parse().ok()) {
            config.num_epochs = ne;
        }
    }
    if let Some(bs) = std::env::var("FSRS_BATCH_SIZE").ok().and_then(|s| s.trim().parse().ok()) {
        config.batch_size = bs;
    }
    // FSRS_LR: the tuner's Adam sqrt LR-batch scaling (lr = committed_lr * sqrt(batch/256)) — a
    // cheap stand-in for re-tuning LR per grid cell, so batch != 256 cells aren't handicapped by
    // an LR tuned at batch 256. Feeds the cosine-annealing scheduler below via config.
    if let Some(lr) = std::env::var("FSRS_LR").ok().and_then(|s| s.trim().parse().ok()) {
        config.learning_rate = lr;
    }
    // The windowed path only PLANS the batches here and hands the plan plus the prefix-items to
    // train(), whose second thread lays the batches out (in the order training uses them) and then
    // frees the items (one heap block each, ~10% of a large user's time) while this one trains.
    let (batches, total_size) = match train_card_ids {
        Some(train_card_ids) => {
            let plan =
                CardedPlan::new(&train_set, &train_card_ids, config.batch_size, config.max_seq_len);
            let total_size = plan.n_preds;
            (TrainBatches::Planned(plan, train_set, train_card_ids), total_size)
        }
        None => {
            let mut weighted_train_set = recency_weighted_fsrs_items_tuned(train_set);
            weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
            let total_size = weighted_train_set.len();
            (TrainBatches::Built(build_host_batches(weighted_train_set, config.batch_size)), total_size)
        }
    };

    if let Some(progress) = &progress {
        let progress_state = ProgressState {
            epoch_total: config.num_epochs,
            items_total: total_size,
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().splits = vec![progress_state];
    }
    let model = train::<Autodiff<B>>(
        batches,
        total_size,
        &initialized_parameters,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    );
    let optimized_parameters = model
        .inspect_err(|_e| {
            finish_progress();
        })?
        .w
        .val()
        .to_data()
        .to_vec()
        .unwrap();

    finish_progress();

    if optimized_parameters
        .iter()
        .any(|parameter: &f32| parameter.is_infinite())
    {
        return Err(FSRSError::InvalidInput);
    }

    Ok(optimized_parameters)
}

/// PROXY for the gated default-param tuner: the windowed 0-epoch loss of `params` over a user's
/// cards (NO training), via the fast SIMD analytic forward `card_loss_simd` — the same kernel the
/// FSRS_VALIDATE diagnostic uses, ~50x cheaper than the burn predict path. Returns the
/// recency-weighted mean BCE (total weighted loss / total weight); NaN if too few items. This is a
/// descent DRIVER only (minimax-approximate, train==test, recency-weighted) — the real selection
/// metric is the faithful 5-fold gate (which uses the frozen evaluate()/burn predict). The same
/// setup as `compute_parameters` (normalize, carded prep, recency weighting, max_seq_len retain,
/// windowed batching) so the proxy tracks what training would see, just without the SGD.
pub fn windowed_loss_with_params(
    train_set: Vec<FSRSItem>,
    card_ids: Vec<i64>,
    params: &[f32],
) -> f64 {
    let train_set = normalize_training_set(train_set);
    let (_, train_set, train_card_ids) = prepare_training_data_carded(train_set, card_ids);
    if train_set.len() < 64 {
        return f64::NAN;
    }
    let config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: false,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: false,
            num_relearning_steps: 1,
        },
        AdamConfig::new(),
    );
    // batch_size only affects card grouping, not the summed loss; the default is fine.
    let (host, _) =
        build_host_batches_carded(&train_set, &train_card_ids, config.batch_size, config.max_seq_len);
    let w = clip_parameters(params);
    let mut total_loss = 0.0f64;
    let mut total_w = 0.0f64;
    for hb in &host {
        if hb.windowed {
            total_loss +=
                crate::analytic::card_loss_simd(&w, &hb.th, &hb.rh, hb.seq, hb.bsz, &hb.lbl, &hb.wts);
            total_w += hb.wts.iter().map(|&x| x as f64).sum::<f64>();
        }
    }
    if total_w > 0.0 {
        total_loss / total_w
    } else {
        f64::NAN
    }
}

pub fn benchmark(
    ComputeParametersInput {
        train_set,
        enable_short_term,
        enable_sched_penalties,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let train_set = normalize_training_set(train_set);
    let (_, train_set) = prepare_training_data(train_set);
    let initialized_parameters = DEFAULT_PARAMETERS.to_vec();
    let mut config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.70)
            .with_beta_2(0.98)
            .with_epsilon(1e-8),
    )
    .with_enable_sched_penalties(enable_sched_penalties);
    // save RAM and speed up training
    config.max_seq_len = 64;
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    let total_size = weighted_train_set.len();
    let model = train::<Autodiff<B>>(
        TrainBatches::Built(build_host_batches(weighted_train_set, config.batch_size)),
        total_size,
        &initialized_parameters,
        &config,
        None,
    );
    let parameters: Vec<f32> = model.unwrap().w.val().to_data().to_vec::<f32>().unwrap();
    parameters
}

/// One batch pre-extracted to host arrays (used for BOTH train and validation). The data and
/// its batching are fixed across epochs, so we snapshot the host arrays once (before the epoch
/// loop) and reuse them every epoch — eliminating the per-step `to_data().to_vec()` copies.
/// For training, only the batch ORDER changes per epoch; we replicate ShuffleDataLoader's
/// per-epoch shuffle (same StdRng seed) so the SGD trajectory is identical (bit-for-bit).
struct BatchHost {
    seq: usize,
    bsz: usize,
    real_batch_size: usize,
    th: Vec<f32>,
    rh: Vec<f32>,
    /// Plain path: `dts`/`lbl`/`wts` are [bsz] (one prediction per column, scored at the final
    /// curve). Windowed path (`windowed == true`): `lbl`/`wts` are [seq, bsz] (a prediction at every
    /// timestep t>=1 with wts>0) and `dts` is empty — the per-step delta_t is just `th[t]`.
    dts: Vec<f32>,
    lbl: Vec<f32>,
    wts: Vec<f32>,
    /// The O(N) expanding-window layout: each column is a whole card, scored at every timestep.
    windowed: bool,
}

/// Build ONE batch's host arrays from a slice of weighted prefix-items (each item = one column,
/// its loss at its last review). Same [seq, batch] row-major 0-padding as the old FSRSBatcher.
fn build_batch_host(chunk: &[WeightedFSRSItem]) -> BatchHost {
    let real_bsz = chunk.len();
    // Pad the card count up to a multiple of 8 so the f32x8 forward/gradient never has a ragged
    // tail group. The pad columns stay all-zero with weight 0: their forward is finite and their
    // loss/gradient contribution is exactly 0 (weight 0), so the result is unchanged.
    // real_batch_size keeps the TRUE count (the penalty scaling uses it).
    let bsz = real_bsz.div_ceil(8) * 8;
    // pad_size = (max reviews per card in this chunk) - 1   (matches FSRSBatcher)
    let seq = chunk.iter().map(|x| x.item.reviews.len()).max().unwrap() - 1;
    let mut th = vec![0.0f32; seq * bsz];
    let mut rh = vec![0.0f32; seq * bsz];
    let mut dts = vec![0.0f32; bsz];
    let mut lbl = vec![0.0f32; bsz];
    let mut wts = vec![0.0f32; bsz];
    for (c, wi) in chunk.iter().enumerate() {
        let reviews = &wi.item.reviews;
        // history() = every review except the last; row-major [seq, bsz] => index t*bsz + c.
        // Entries past this card's history stay 0.0 (the pad).
        for (t, r) in reviews.iter().take(reviews.len() - 1).enumerate() {
            th[t * bsz + c] = r.delta_t;
            rh[t * bsz + c] = r.rating as f32;
        }
        let current = reviews.last().unwrap();
        dts[c] = current.delta_t;
        lbl[c] = if current.rating == 1 { 0.0 } else { 1.0 };
        wts[c] = wi.weight;
    }
    BatchHost { seq, bsz, real_batch_size: real_bsz, th, rh, dts, lbl, wts, windowed: false }
}

/// Hashes an i64 card id with one multiply (std's SipHash is DoS-hardened and ~5x slower here).
#[derive(Default)]
struct CardIdHasher(u64);

impl std::hash::Hasher for CardIdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_i64(&mut self, i: i64) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Windowed path (card ids present): the batch PLAN built straight from the prefix-items. Each card
/// becomes ONE column holding its FULL review sequence, taken from its longest surviving prefix
/// (every shorter prefix is a head of it). A surviving prefix of length L (<= max_seq_len) predicts
/// review L-1, so its recency weight goes to wts[(L-1)*bsz + c] (label likewise); every other (t, c)
/// stays weight 0 (a filtered middle prefix, t==0, or padding). Cards are ordered by (full length,
/// card id) — a total order, so the batches are reproducible and length-similar (less SIMD padding)
/// — and WHOLE cards are packed into batches of at most `batch_size` predictions, so each Adam step
/// sees a whole card's predictions together. The plan reads only item headers; `layout` then fills
/// one batch's arrays (reading each card's longest prefix, clamping delta_t as
/// normalize_training_set does), so train() can lay out batches on its second thread.
struct CardedPlan {
    cards: Vec<PlanCard>,
    /// Each card's predictions (t, item index), contiguous per card and in input order.
    start: Vec<usize>,
    by_card: Vec<(u32, u32)>,
    order: Vec<usize>,
    /// Each batch = order[first..end].
    bounds: Vec<(usize, usize)>,
    n_items: usize,
    n_preds: usize,
}

struct PlanCard {
    id: i64,
    full_len: usize,
    longest: usize,
    n_preds: usize,
}

impl CardedPlan {
    fn new(items: &[FSRSItem], card_ids: &[i64], batch_size: usize, max_seq_len: usize) -> Self {
        // One pass in input order over the item headers only (a prefix's label is the rating of
        // review t of its card's longest prefix, and its weight depends only on its index, so both
        // are read at layout time — no per-item heap access here).
        let mut card_index: HashMap<i64, usize, BuildHasherDefault<CardIdHasher>> = HashMap::default();
        let mut cards: Vec<PlanCard> = Vec::new();
        let mut preds: Vec<(u32, u32, u32)> = Vec::with_capacity(items.len()); // (card, t, idx)
        for (idx, (item, &id)) in items.iter().zip(card_ids).enumerate() {
            let len = item.reviews.len();
            if len > max_seq_len {
                continue;
            }
            let ci = *card_index.entry(id).or_insert_with(|| {
                cards.push(PlanCard { id, full_len: 0, longest: 0, n_preds: 0 });
                cards.len() - 1
            });
            let card = &mut cards[ci];
            // >= keeps the LAST longest prefix (max_by_key's tie rule).
            if len >= card.full_len {
                card.full_len = len;
                card.longest = idx;
            }
            card.n_preds += 1;
            preds.push((ci as u32, len as u32 - 1, idx as u32));
        }
        // Each card's predictions, contiguous and in input order (a counting sort).
        let mut start = vec![0usize; cards.len() + 1];
        for (ci, card) in cards.iter().enumerate() {
            start[ci + 1] = start[ci] + card.n_preds;
        }
        let mut fill = start.clone();
        let mut by_card = vec![(0u32, 0u32); preds.len()];
        for &(ci, t, idx) in &preds {
            by_card[fill[ci as usize]] = (t, idx);
            fill[ci as usize] += 1;
        }
        let mut order: Vec<usize> = (0..cards.len()).collect();
        order.sort_unstable_by_key(|&ci| (cards[ci].full_len, cards[ci].id));
        let mut bounds = Vec::new();
        let (mut first, mut current_preds) = (0, 0);
        for (k, &ci) in order.iter().enumerate() {
            if k > first && current_preds + cards[ci].n_preds > batch_size {
                bounds.push((first, k));
                (first, current_preds) = (k, 0);
            }
            current_preds += cards[ci].n_preds;
        }
        if first < order.len() {
            bounds.push((first, order.len()));
        }
        Self { cards, start, by_card, order, bounds, n_items: items.len(), n_preds: preds.len() }
    }

    /// Padded column count of batch `b` (a multiple of 8: the SIMD kernels' 8-card groups).
    fn bsz(&self, b: usize) -> usize {
        (self.bounds[b].1 - self.bounds[b].0).div_ceil(8) * 8
    }

    fn layout(&self, items: &[FSRSItem], b: usize, weight: &impl Fn(usize, f32) -> f32) -> BatchHost {
        let (first, end) = self.bounds[b];
        let batch = &self.order[first..end];
        let length = (self.n_items as f32).max(1.0);
        let bsz = self.bsz(b);
        let seq = batch.iter().map(|&ci| self.cards[ci].full_len).max().unwrap_or(0);
        let mut th = vec![0.0f32; seq * bsz];
        let mut rh = vec![0.0f32; seq * bsz];
        let mut lbl = vec![0.0f32; seq * bsz];
        let mut wts = vec![0.0f32; seq * bsz];
        let mut predictions = 0;
        for (c, &ci) in batch.iter().enumerate() {
            let reviews = &items[self.cards[ci].longest].reviews;
            for (t, r) in reviews.iter().enumerate() {
                th[t * bsz + c] = r.delta_t.max(0.0);
                rh[t * bsz + c] = r.rating as f32;
            }
            for &(t, idx) in &self.by_card[self.start[ci]..self.start[ci + 1]] {
                let t = t as usize;
                wts[t * bsz + c] = weight(idx as usize, length);
                lbl[t * bsz + c] = if reviews[t].rating == 1 { 0.0 } else { 1.0 };
            }
            predictions += self.cards[ci].n_preds;
        }
        BatchHost {
            seq, bsz, real_batch_size: predictions, th, rh, dts: Vec::new(), lbl, wts, windowed: true,
        }
    }
}

/// All windowed batches on the calling thread (the tuner proxy); returns them and the prefix count.
fn build_host_batches_carded(
    items: &[FSRSItem],
    card_ids: &[i64],
    batch_size: usize,
    max_seq_len: usize,
) -> (Vec<BatchHost>, usize) {
    let plan = CardedPlan::new(items, card_ids, batch_size, max_seq_len);
    let weight = recency_weight_tuned();
    ((0..plan.bounds.len()).map(|b| plan.layout(items, b, &weight)).collect(), plan.n_preds)
}

/// train()'s batches: already laid out (plain path), or a windowed plan that train()'s second
/// thread lays out (it owns the prefix-items and frees them afterwards).
enum TrainBatches {
    Built(Vec<BatchHost>),
    Planned(CardedPlan, Vec<FSRSItem>, Vec<i64>),
}

/// Build the per-batch host arrays of the plain path (no card ids: the benchmark()/evaluate()
/// paths) DIRECTLY from the weighted items — no burn tensors. Sort by review length, chunk into
/// `batch_size` groups — BIT-FOR-BIT identical to the old BatchTensorDataset path.
fn build_host_batches(items: Vec<WeightedFSRSItem>, batch_size: usize) -> Vec<BatchHost> {
    let items = sort_items_by_review_length(items);
    items.chunks(batch_size).map(build_batch_host).collect()
}

/// State shared with train()'s second thread (constraint 2: at most 2 threads per user). For each
/// windowed batch the main thread publishes the weights and the batch, then BOTH threads claim
/// 8-card groups from `job` and store each group's gradient at its index in `out`; the main thread
/// then adds the groups in group order, so the result is bit-for-bit the one-thread sum. Batches
/// take tens of microseconds, so the threads spin (spin_loop) instead of sleeping.
struct GradShared<'a> {
    /// Filled once each: all at once (plain path) or by the helper's layout (windowed path).
    host: &'a [OnceLock<BatchHost>],
    /// (job sequence << 32) | next group to claim. The sequence in the same word keeps a thread
    /// that is still in an old job from claiming a group of the new one. JOB_STOP ends the helper.
    job: AtomicU64,
    batch: AtomicUsize,
    w: [AtomicU32; 34],
    done: AtomicUsize,
    out: Vec<[AtomicU32; 34]>,
}

const JOB_STOP: u64 = u64::MAX;

impl GradShared<'_> {
    /// Claim and compute groups of job `seq` until none is left (or the job changed).
    fn work(&self, seq: u32, w: &[f32], wc: &crate::analytic::WConsts, caches: &mut Vec<crate::analytic::Step8>) {
        let hb = self.host[self.batch.load(Ordering::Relaxed)].get().expect("published batch");
        let n = hb.bsz / 8;
        loop {
            let v = self.job.load(Ordering::Acquire);
            let k = (v & 0xffff_ffff) as usize;
            if (v >> 32) as u32 != seq || k >= n {
                return;
            }
            if self.job.compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire).is_err() {
                continue;
            }
            // Longest groups first (cards are length-sorted within a batch): better balance.
            let g = n - 1 - k;
            let gg = crate::analytic::card_group_grad(
                w, wc, &hb.th, &hb.rh, hb.seq, hb.bsz, &hb.lbl, &hb.wts, g, caches,
            );
            for (o, x) in self.out[g].iter().zip(gg) {
                o.store(x.to_bits(), Ordering::Relaxed);
            }
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    /// The second thread: lay out the windowed batches in the order training first uses them, free
    /// the prefix-items, then help with every published job.
    fn helper(&self, plan: CardedPlan, items: Vec<FSRSItem>, card_ids: Vec<i64>, first_order: &[usize]) {
        let weight = recency_weight_tuned();
        for &b in first_order {
            let _ = self.host[b].set(plan.layout(&items, b, &weight));
        }
        drop((plan, items, card_ids));
        let mut caches = Vec::new();
        let mut seen = 0u32;
        loop {
            let v = self.job.load(Ordering::Acquire);
            if v == JOB_STOP {
                return;
            }
            let seq = (v >> 32) as u32;
            if seq == seen {
                std::hint::spin_loop();
                continue;
            }
            seen = seq;
            let w: [f32; 34] = std::array::from_fn(|i| f32::from_bits(self.w[i].load(Ordering::Relaxed)));
            let wc = crate::analytic::wconsts(&w);
            self.work(seq, &w, &wc, &mut caches);
        }
    }
}

fn train<B: AutodiffBackend>(
    batches: TrainBatches,
    total_size: usize,
    initial_parameters: &[f32],
    config: &TrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    // Training data: the caller builds the host batches ONCE (before the epoch loop); reused every
    // epoch — only the batch ORDER reshuffles per epoch. (The per-epoch validation pass + its
    // best-epoch selection were REMOVED: training ships the last epoch's parameters, with the
    // 9-epoch default compensating — see the return at the bottom.)
    let (n_train_batches, max_bsz) = match &batches {
        TrainBatches::Built(host) => (host.len(), host.iter().map(|hb| hb.bsz).max().unwrap_or(0)),
        TrainBatches::Planned(plan, ..) => {
            (plan.bounds.len(), (0..plan.bounds.len()).map(|b| plan.bsz(b)).max().unwrap_or(0))
        }
    };
    let train_host: Vec<OnceLock<BatchHost>> = (0..n_train_batches).map(|_| OnceLock::new()).collect();
    // Cosine-annealing horizon = the TRUE step count (faithful to CUDA, which clamps progress
    // over the exact per-user total). The old `(total/batch + 1) * epochs` estimate undercounts
    // windowed batches (whole cards never split, so batches under-fill) — the schedule then hit
    // zero before training ended and bounced back up (torch cosine is periodic past t_max) for
    // the tail steps.
    let iterations = n_train_batches * config.num_epochs;
    // Replicates ShuffleDataLoader's RNG (StdRng::seed_from_u64(seed), advanced one shuffle per
    // epoch) so the per-epoch batch order is byte-identical to the old dataloader path.
    let mut shuffle_rng = StdRng::seed_from_u64(config.seed);
    // Epoch 1's batch order (the same shuffle the loop below makes, on a clone of the RNG): the
    // helper lays the windowed batches out in this order.
    let mut first_order: Vec<usize> = (0..n_train_batches).collect();
    first_order.shuffle(&mut shuffle_rng.clone());

    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let interrupter = TrainingInterrupter::new();
    let mut renderer: Box<dyn MetricsRenderer> = match progress {
        Some(mut progress) => {
            progress.interrupter = interrupter.clone();
            Box::new(progress)
        }
        None => Box::new(NoProgress {}),
    };

    let schedule_penalty = schedule_penalty_fn();
    let l2_penalty = l2_penalty_fn();
    // Host-resident parameters + Adam state. The gradient is already analytic, so burn's optimizer
    // was the LAST burn-tensor code in the per-step loop — each step round-tripped w through
    // model.w.val().to_data().to_vec() (and again inside the clipper) plus a fresh GradientsParams.
    // We keep w in a plain Vec<f32> and hand-roll Adam element-wise (see the loop below), so a step
    // is now pure host arithmetic on 34 floats. parameters_to_model clips, so this start == the old
    // model.w.val(); the returned Model is rebuilt once from the final w_host at the very end.
    let mut w_host: Vec<f32> = clip_parameters(initial_parameters);
    let init_w_vec = w_host.clone();
    let mut adam_m = [0.0f32; 34]; // Adam 1st moment (burn AdaptiveMomentumState.moment_1)
    let mut adam_v = [0.0f32; 34]; // Adam 2nd moment (moment_2)
    let mut adam_t = 0i32; // step count (AdaptiveMomentumState.time; becomes 1 on the first step)
    // TUNER Adam-beta overrides (hp_tune per-cell fine tune). Default to the shipped consts so
    // production (neither var set) is bit-for-bit; read once here, used in the per-step loop below.
    let beta1 = std::env::var("FSRS_BETA1")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(ADAM_BETA1);
    let beta2 = std::env::var("FSRS_BETA2")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(ADAM_BETA2);
    // TUNER L2-strength override (hp_tune per-cell fine tune). Default to the shipped const so
    // production (no env) is bit-for-bit; read once here, used as the L2 penalty weight below.
    let l2_weight = std::env::var("FSRS_L2")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(L2_PENALTY_WEIGHT);

    // PROFILING-ONLY (default off => shipped path bit-for-bit): when FSRS_VALIDATE is set, re-run
    // the per-epoch validation forward that the finished-model port removed (training.rs end note),
    // so the no-per-epoch-validation speedup can be measured on the IDENTICAL model. Read once.
    let validate = std::env::var_os("FSRS_VALIDATE").is_some();

    // PROFILING-ONLY (not committed): per-phase wall time decomposition.
    // t_bwd = analytic gradient, t_opt = hand-rolled Adam + clip. (Per-step extract is gone — the
    // train batches are pre-extracted once into train_host, so that cost is now a one-time floor.)
    let (mut t_pen, mut t_bwd, mut t_opt) = (0.0f64, 0.0f64, 0.0f64);
    // Timers only under FSRS_PROFILE: 6 clock reads per step were ~2% of a median user's time.
    let profile = std::env::var_os("FSRS_PROFILE").is_some();
    let tick = || profile.then(std::time::Instant::now);
    let secs = |t: Option<std::time::Instant>| t.map_or(0.0, |t| t.elapsed().as_secs_f64());
    let shared = GradShared {
        host: &train_host,
        job: AtomicU64::new(0),
        batch: AtomicUsize::new(0),
        w: std::array::from_fn(|_| AtomicU32::new(0)),
        done: AtomicUsize::new(0),
        out: (0..max_bsz / 8).map(|_| std::array::from_fn(|_| AtomicU32::new(0))).collect(),
    };
    let mut caches = Vec::new();
    let mut seq = 0u32;
    std::thread::scope(|scope| {
    match batches {
        TrainBatches::Built(host) => {
            for (slot, hb) in train_host.iter().zip(host) {
                let _ = slot.set(hb);
            }
        }
        TrainBatches::Planned(plan, items, card_ids) => {
            let (shared, first_order) = (&shared, &first_order);
            scope.spawn(move || shared.helper(plan, items, card_ids, first_order));
        }
    }
    for epoch in 1..=config.num_epochs {
        // Replicate the dataloader's per-epoch shuffle (one shuffle of [0, n_batches) per epoch).
        let mut order: Vec<usize> = (0..n_train_batches).collect();
        order.shuffle(&mut shuffle_rng);
        let mut iteration = 0;
        for &bi in &order {
            iteration += 1;
            // Laid out by the helper, normally well ahead of use (only the first batch is waited on).
            let hb = loop {
                match train_host[bi].get() {
                    Some(hb) => break hb,
                    None => std::hint::spin_loop(),
                }
            };
            let real_batch_size = hb.real_batch_size;
            let lr = LrScheduler::step(&mut lr_scheduler);
            let progress = Progress::new(iteration, n_train_batches);
            let _tp = tick();
            let w_vec = &w_host;
            let mut manual_grad = l2_penalty(
                w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
                &training_v7::PARAMS_STDDEV,
            );
            let (_schedule_value, schedule_grad) =
                schedule_penalty(w_vec, real_batch_size, config.enable_sched_penalties);
            let inv_total = 1.0 / total_size as f64;
            for i in 0..manual_grad.len().min(schedule_grad.len()) {
                manual_grad[i] += (schedule_grad[i] * inv_total) as f32;
            }
            t_pen += secs(_tp);
            // Host batch data was pre-extracted once into train_host (no per-step copy now).
            // Hand-written analytic BCE gradient (replaces the autodiff forward+backward),
            // plus the manual L2/schedule penalty gradient.
            let _tb = tick();
            let mut total_grad = [0.0f64; 34];
            if hb.windowed {
                // O(N) expanding-window grad: one pass per card, a loss at every timestep, with the
                // groups split over the two threads (see GradShared).
                for (a, &x) in shared.w.iter().zip(w_vec) {
                    a.store(x.to_bits(), Ordering::Relaxed);
                }
                shared.batch.store(bi, Ordering::Relaxed);
                shared.done.store(0, Ordering::Relaxed);
                seq += 1;
                shared.job.store((seq as u64) << 32, Ordering::Release);
                let wc = crate::analytic::wconsts(w_vec);
                shared.work(seq, w_vec, &wc, &mut caches);
                let n = hb.bsz / 8;
                while shared.done.load(Ordering::Acquire) < n {
                    std::hint::spin_loop();
                }
                for group in &shared.out[..n] {
                    for (t, o) in total_grad.iter_mut().zip(group) {
                        *t += f32::from_bits(o.load(Ordering::Relaxed)) as f64;
                    }
                }
            } else {
                crate::analytic::batch_loss_and_grad_simd(
                    w_vec, &hb.th, &hb.rh, hb.seq, hb.bsz, &hb.dts, &hb.lbl, &hb.wts, &mut total_grad,
                );
            }
            let mut total_grad_f32 = [0.0f32; 34];
            for i in 0..34 {
                total_grad_f32[i] = total_grad[i] as f32 + manual_grad.get(i).copied().unwrap_or(0.0);
            }
            if config.model.freeze_initial_stability {
                for v in total_grad_f32.iter_mut().take(4) {
                    *v = 0.0;
                }
            }
            if config.model.freeze_short_term_stability {
                // short stability block is 15..23 in the 34-param layout.
                for v in total_grad_f32.iter_mut().take(23).skip(15) {
                    *v = 0.0;
                }
            }
            t_bwd += secs(_tb);
            let _to = tick();
            // Hand-rolled Adam — replaces burn's tensor optimizer (the last burn code in the loop;
            // it had round-tripped w through to_data().to_vec() + a fresh GradientsParams each step).
            // Element-wise replica of burn 0.17's AdaptiveMomentum: m,v start at 0; time starts at 1;
            // bias-correct by (1 - beta^t); epsilon OUTSIDE the sqrt; w -= lr * m_hat/(sqrt(v_hat)+eps).
            // burn's first-step special case (m = g*factor) is just this formula with m=v=0, so the
            // unified loop reproduces it. Same hyperparameters (ADAM_*) as the AdamConfig → the Adam
            // math is identical; only the FP op order vs burn's ndarray kernels can differ (3b band).
            adam_t += 1;
            let bc1 = 1.0f32 - beta1.powi(adam_t);
            let bc2 = 1.0f32 - beta2.powi(adam_t);
            let f1 = 1.0f32 - beta1;
            let f2 = 1.0f32 - beta2;
            let lr_f32 = lr as f32;
            for i in 0..34 {
                let g = total_grad_f32[i];
                adam_m[i] = adam_m[i] * beta1 + g * f1;
                adam_v[i] = adam_v[i] * beta2 + g.powf(2.0) * f2;
                let m_hat = adam_m[i] / bc1;
                let v_hat = adam_v[i] / bc2;
                w_host[i] -= lr_f32 * (m_hat / (v_hat.sqrt() + ADAM_EPS));
            }
            // The same clamp burn applied via parameter_clipper (clip_parameters), now in place.
            clip_parameters_in_place(&mut w_host);
            t_opt += secs(_to);
            renderer.render_train(TrainingProgress {
                progress,
                epoch,
                epoch_total: config.num_epochs,
                iteration,
            });

            if interrupter.should_stop() {
                break;
            }
        }

        if interrupter.should_stop() {
            break;
        }

        if validate {
            // Per-epoch validation forward (gated by FSRS_VALIDATE; default off). Scores
            // card_loss_simd over the train batches (train==test in compute_parameters), exactly
            // as the removed best-epoch-selection pass did. Result discarded — black_box keeps it
            // live so the timing reflects the real validation cost; w_host is NOT touched, so the
            // trained parameters are identical whether or not validation runs.
            let mut vloss = 0.0f64;
            for hb in train_host.iter().filter_map(|slot| slot.get()) {
                if hb.windowed {
                    vloss += crate::analytic::card_loss_simd(
                        &w_host, &hb.th, &hb.rh, hb.seq, hb.bsz, &hb.lbl, &hb.wts,
                    );
                }
            }
            std::hint::black_box(vloss);
        }

        info!("epoch: {:?} done", epoch);
    }
    shared.job.store(JOB_STOP, Ordering::Release);
    });
    // Per-region training timing, silent unless FSRS_PROFILE is set (it printed on
    // every train() call before — clutter + stderr I/O in the timed path). The env
    // gate keeps the t_* accumulators live (no dead-code warnings) and lets profiling
    // re-enable the dump on demand.
    if std::env::var_os("FSRS_PROFILE").is_some() {
        eprintln!(
            "PROFILE total_train_region: pen={:.3} grad={:.3} opt={:.3}",
            t_pen, t_bwd, t_opt
        );
    }

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    // No per-epoch validation / best-epoch selection: the shipped parameters are the LAST
    // epoch's (the 9th-epoch default below compensates — dropping the selection can only
    // raise the loss by construction, the extra epoch buys it back at a fraction of the
    // validation pass's cost).
    Ok(parameters_to_model::<B>(&w_host, &B::Device::default()))
}

struct NoProgress {}

impl MetricsRenderer for NoProgress {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, _item: TrainingProgress) {}

    fn render_valid(&mut self, _item: TrainingProgress) {}
}
