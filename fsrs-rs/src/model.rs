use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::Parameters;
// FSRS-7 stability/difficulty clamp bounds (relocated from the removed simulator).
pub(crate) const S_MIN: f32 = 0.0001;
pub(crate) const S_MAX: f32 = 36500.0;
pub(crate) const D_MIN: f32 = 1.0;
pub(crate) const D_MAX: f32 = 10.0;
use crate::training::ModelConfig;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    constant,
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

pub(crate) mod model_v7 {
use super::{Fsrs7Ops, Get, MemoryStateTensors, Model, VersionOps, tensor_max, tensor_min};
use burn::tensor::{Tensor, backend::Backend};

pub(super) const PARAM_LEN: usize = 34;

impl<B: Backend> VersionOps<B> for Fsrs7Ops {
    fn apply_freeze_short_term(_initial_params: &mut [f32]) {
        // Dual-trace FSRS-7: the single-trace "transition weight" param that freeze-short-term
        // used to zero is gone (its old slot was repurposed in the dual-trace layout), and the
        // fast trace is intrinsic to the model, so "freeze short term" has nothing to zero — it
        // is a no-op. Unused in the timed paths anyway (enable_short_term is always true there).
    }

    fn power_forgetting_curve(
        model: &Model<B>,
        t: Tensor<B, 1>,
        s: Tensor<B, 1>,
        s_fast: Tensor<B, 1>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        power_forgetting_curve(model, t, s, s_fast, d)
    }

    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        last_s_fast: Tensor<B, 1>,
    ) -> MemoryStateTensors<B> {
        let delta_t = delta_t.clamp_min(0.0);
        // DUAL-TRACE (finished FSRS-7): the curve mixes a fast-trace and a slow-trace
        // recall component; after review the slow trace updates from the mixed retention
        // and the fast trace from its OWN recall r1, with a post-lapse fast reset.
        let retrievability = power_forgetting_curve(
            model,
            delta_t.clone(),
            last_s.clone(),
            last_s_fast.clone(),
            last_d.clone(),
        );
        let new_s_slow = stability_for_set(
            model,
            last_s,
            last_d.clone(),
            retrievability.clone(),
            rating.clone(),
            7,
        );
        // FAST trace: update from the fast component's own recall r1 (iter-71), reading the
        // fast trace as the stability input; short stability block start = 15.
        let r1 = fast_component_recall(model, delta_t, last_s_fast.clone());
        let new_s_fast = stability_for_set(
            model,
            last_s_fast,
            last_d.clone(),
            r1,
            rating.clone(),
            15,
        );
        // POST-LAPSE fast reset (iter-97): on a lapse cap s_fast at 0.8 * post-lapse slow S.
        let relearn = tensor_min(new_s_fast.clone(), new_s_slow.clone().mul_scalar(0.8));
        let new_s_fast = new_s_fast.mask_where(rating.clone().equal_elem(1), relearn);
        let new_d = next_difficulty(model, last_d, rating, retrievability);
        MemoryStateTensors {
            stability: new_s_slow,
            difficulty: new_d,
            stability_fast: new_s_fast,
        }
    }
}

pub(super) fn fast_component_recall<B: Backend>(
    model: &Model<B>,
    t: Tensor<B, 1>,
    s_fast: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // FAST recall component r1, driven by the fast trace; decay S-modulated (s_decay1).
    // factor1 built in LOG-SPACE with the exponent clamped at 60 so value+gradient stay
    // finite (matches fsrs7.cu fsrs7_fast_component_recall). Shared by the forgetting curve
    // (mixture) AND the fast-trace stability update (which now reads r1, not the mixed R).
    let t = t.clamp_min(0.0);
    let t_over_s_fast = t / s_fast.clone();
    let decay1_mag = (model.w.get(23) * s_fast.powf(model.w.get(33).sub_scalar(0.3))).clamp(0.01, 0.95);
    let decay1 = -decay1_mag;
    let factor1 = (model.w.get(25).log() * decay1.clone().powi_scalar(-1))
        .clamp_max(60.0)
        .exp()
        - 1.0;
    (t_over_s_fast * factor1 + 1.0).powf(decay1)
}

pub(super) fn power_forgetting_curve<B: Backend>(
    model: &Model<B>,
    t: Tensor<B, 1>,
    s: Tensor<B, 1>,
    s_fast: Tensor<B, 1>,
    d: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // DUAL-TRACE forgetting curve (finished FSRS-7, 34-param layout; subday term cut).
    // Curve indices: 23 decay1, 24 decay2, 25 base1, 26 base2, 27 base_weight1,
    // 28 base_weight2, 29 s_weight_power1, 30 s_weight_power2, 31 d_weight, 32 d_decay,
    // 33 s_decay1.
    let t = t.clamp_min(0.0);
    let t_over_s_slow = t.clone() / s.clone();

    // FAST component r1 reads the fast trace (shared with the fast-trace update).
    let r1 = fast_component_recall(model, t, s_fast.clone());

    // SLOW component r2 reads the slow trace. iter-165: the D-effect moved from the decay
    // EXPONENT to the horizontal TIME-SCALE — decay2 is no longer d-modulated; instead r2
    // sees time scaled by exp((d_decay-0.3)*(d-5)), so hard cards experience time faster
    // but keep the same asymptotic decay slope.
    let decay2_mag = model.w.get(24).clamp(0.01, 0.95);
    let decay2 = -decay2_mag;
    let factor2 = model.w.get(26).powf(decay2.clone().powi_scalar(-1)) - 1.0;
    let d_timescale = (d.clone().add_scalar(-5.0) * model.w.get(32).sub_scalar(0.3)).exp();
    let r2 = (t_over_s_slow * factor2 * d_timescale + 1.0).powf(decay2);

    // Mixture weights keyed to each trace; weight2 is D-modulated (d_weight).
    let weight1 = model.w.get(27) * s_fast.powf(-model.w.get(29));
    let weight2 =
        model.w.get(28) * s.powf(model.w.get(30)) * (d.add_scalar(-5.0) * model.w.get(31).sub_scalar(0.5)).exp();

    let retention = (weight1.clone() * r1 + weight2.clone() * r2) / (weight1 + weight2);
    // Final rescale: p = 1e-5 + (1 - 2e-5) * retention.
    retention.mul_scalar(1.0 - 2e-5).add_scalar(1e-5)
}

pub(super) fn stability_for_set<B: Backend>(
    model: &Model<B>,
    last_s: Tensor<B, 1>,
    last_d: Tensor<B, 1>,
    r: Tensor<B, 1>,
    rating: Tensor<B, 1>,
    start: usize,
) -> Tensor<B, 1> {
    let batch_size = rating.dims()[0];
    let device = rating.device();
    let hard_penalty = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(2), model.w.get(start + 6));
    let easy_bonus = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(4), model.w.get(start + 7));

    // Post-lapse stability is D-INDEPENDENT (the d^(-fail_d_exp) factor was ablated in the
    // finished model). new_s_fail = fail_mult * ((s+1)^fail_s_exp - 1) * exp((1-r)*fail_r_mult).
    let new_s_fail = model.w.get(start + 3)
        * ((last_s.clone() + 1).powf(model.w.get(start + 4)) - 1)
        * ((-r.clone() + 1) * model.w.get(start + 5)).exp();
    let pls = tensor_min(last_s.clone(), new_s_fail);

    let sinc = model.w.get(start).add_scalar(-1.5).exp()
        * last_d.neg().add_scalar(11.0)
        * last_s.clone().powf(-model.w.get(start + 1))
        * (((-r + 1) * model.w.get(start + 2)).exp() - 1)
        * hard_penalty
        * easy_bonus
        + 1;
    let new_s_success = tensor_max(pls.clone(), last_s * sinc);
    let success = rating.greater_elem(1);
    pls.mask_where(success, new_s_success)
}

pub(super) fn mean_reversion<B: Backend>(
    init: Tensor<B, 1>,
    current: Tensor<B, 1>,
) -> Tensor<B, 1> {
    init.mul_scalar(0.01) + current.mul_scalar(0.99)
}

pub(super) fn next_difficulty<B: Backend>(
    model: &Model<B>,
    difficulty: Tensor<B, 1>,
    rating: Tensor<B, 1>,
    retention: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let delta_d = -model.w.get(6) * (rating.clone() - 3);
    // SURPRISE-WEIGHTED lapse difficulty: on a lapse (rating==1) scale delta_d by
    // 1 + (retention - 0.9) = retention + 0.1. A lapse the model expected to recall
    // (high R) is more diagnostic of difficulty than an overdue lapse (low R).
    let surprise = retention.add_scalar(0.1);
    let delta_d_lapse = delta_d.clone() * surprise;
    let delta_d = delta_d.mask_where(rating.equal_elem(1), delta_d_lapse);
    let new_d = difficulty.clone() + model.linear_damping(delta_d, difficulty);
    let device = new_d.device();
    let init = model.init_difficulty(Tensor::from_floats([4.0], &device));
    mean_reversion(init, new_d).clamp(super::D_MIN, super::D_MAX)
}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelVersion {
    Fsrs7,
}

impl core::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fsrs7 => write!(f, "FSRS7"),
        }
    }
}

constant!(ModelVersion);

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
    version: ModelVersion,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

fn tensor_min<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().greater(b.clone()), b)
}

fn tensor_max<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().lower(b.clone()), b)
}

pub(super) trait VersionOps<B: Backend> {
    fn apply_freeze_short_term(initial_params: &mut [f32]);
    fn power_forgetting_curve(
        model: &Model<B>,
        t: Tensor<B, 1>,
        s: Tensor<B, 1>,
        s_fast: Tensor<B, 1>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 1>;
    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        last_s_fast: Tensor<B, 1>,
    ) -> MemoryStateTensors<B>;
}

pub(super) struct Fsrs7Ops;

type ApplyFreezeShortTermFn = fn(&mut [f32]);
type PowerForgettingCurveFn<B> =
    fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) -> Tensor<B, 1>;
type UpdateStateFn<B> = fn(
    &Model<B>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
) -> MemoryStateTensors<B>;

#[derive(Clone, Copy)]
struct VersionFns<B: Backend> {
    apply_freeze_short_term: ApplyFreezeShortTermFn,
    power_forgetting_curve: PowerForgettingCurveFn<B>,
    update_state: UpdateStateFn<B>,
}

impl<B: Backend> VersionFns<B> {
    fn from_version(version: ModelVersion) -> Self {
        match version {
            ModelVersion::Fsrs7 => Self {
                apply_freeze_short_term: <Fsrs7Ops as VersionOps<B>>::apply_freeze_short_term,
                power_forgetting_curve: <Fsrs7Ops as VersionOps<B>>::power_forgetting_curve,
                update_state: <Fsrs7Ops as VersionOps<B>>::update_state,
            },
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn new_with_device(config: ModelConfig, device: &B::Device) -> Self {
        let mut initial_params = DEFAULT_PARAMETERS.to_vec();
        let version = ModelVersion::Fsrs7;
        if let Some(initial_stability) = config.initial_stability {
            initial_params[0..4].copy_from_slice(&initial_stability);
        }
        if let Some(initial_forgetting_curve) = config.initial_forgetting_curve {
            // 8 curve params now live at indices 23..31 (34-param finished layout).
            initial_params[23..31].copy_from_slice(&initial_forgetting_curve);
        }
        if config.freeze_short_term_stability {
            let ops = VersionFns::<B>::from_version(version);
            (ops.apply_freeze_short_term)(&mut initial_params);
        }

        Self {
            w: Param::from_tensor(Tensor::from_floats(
                TensorData::new(
                    initial_params.clone(),
                    Shape {
                        dims: vec![initial_params.len()],
                    },
                ),
                device,
            )),
            version,
        }
    }

    pub(crate) fn version(&self) -> ModelVersion {
        self.version
    }

    pub fn power_forgetting_curve(
        &self,
        t: Tensor<B, 1>,
        s: Tensor<B, 1>,
        s_fast: Tensor<B, 1>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.power_forgetting_curve)(self, t, s, s_fast, d)
    }

    pub(crate) fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.val().select(0, rating.int() - 1)
    }

    fn init_difficulty(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(4) - (self.w.get(5) * (rating - 1)).exp() + 1
    }

    fn linear_damping(&self, delta_d: Tensor<B, 1>, old_d: Tensor<B, 1>) -> Tensor<B, 1> {
        old_d.neg().add_scalar(10.0) * delta_d.div_scalar(9.0)
    }

    fn step_with_ops(
        &self,
        ops: &VersionFns<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: MemoryStateTensors<B>,
        nth: usize,
    ) -> MemoryStateTensors<B> {
        let last_s = state.stability.clone().clamp(S_MIN, S_MAX);
        let last_d = state.difficulty.clone().clamp(D_MIN, D_MAX);
        let last_s_fast = state.stability_fast.clone().clamp(S_MIN, S_MAX);
        let mut new_state = (ops.update_state)(
            self,
            delta_t.clone(),
            rating.clone(),
            last_s.clone(),
            last_d.clone(),
            last_s_fast.clone(),
        );

        if nth == 0 {
            let is_initial = state.stability.clone().equal_elem(0.0);
            let init_s = self.init_stability(rating.clone().clamp(1, 4));
            let init_d = self
                .init_difficulty(rating.clone().clamp(1, 4))
                .clamp(D_MIN, D_MAX);
            // CUDA fsrs7_init: the fast trace starts at 0.8 * initial slow stability.
            let init_s_fast = init_s.clone().mul_scalar(0.8);
            new_state.stability = new_state.stability.mask_where(is_initial.clone(), init_s);
            new_state.difficulty = new_state.difficulty.mask_where(is_initial.clone(), init_d);
            new_state.stability_fast =
                new_state.stability_fast.mask_where(is_initial, init_s_fast);
        }

        // mask padding zeros for rating
        new_state.stability = new_state
            .stability
            .mask_where(rating.clone().equal_elem(0), last_s)
            .clamp(S_MIN, S_MAX);
        new_state.difficulty = new_state
            .difficulty
            .mask_where(rating.clone().equal_elem(0), last_d);
        new_state.stability_fast = new_state
            .stability_fast
            .mask_where(rating.equal_elem(0), last_s_fast)
            .clamp(S_MIN, S_MAX);

        new_state
    }

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2>,
        starting_state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, batch_size] = delta_ts.dims();
        let mut state = if let Some(state) = starting_state {
            state
        } else {
            MemoryStateTensors::zeros(batch_size)
        };
        let ops = VersionFns::<B>::from_version(self.version());
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            let rating = ratings.get(i).squeeze(0);
            state = self.step_with_ops(&ops, delta_t, rating, state, i);
        }
        state
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
    pub stability_fast: Tensor<B, 1>,
}

impl<B: Backend> MemoryStateTensors<B> {
    pub(crate) fn zeros(batch_size: usize) -> MemoryStateTensors<B> {
        let device = B::Device::default();
        MemoryStateTensors {
            stability: Tensor::zeros([batch_size], &device),
            difficulty: Tensor::zeros([batch_size], &device),
            stability_fast: Tensor::zeros([batch_size], &device),
        }
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArray> {
    model: Model<B>,
}

impl Default for FSRS<NdArray> {
    fn default() -> Self {
        Self::new(&[]).expect("Default parameters should be valid")
    }
}

impl FSRS<NdArray> {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: &Parameters) -> Result<Self> {
        Self::new_with_backend(parameters, &NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        parameters: &Parameters,
        device: &B2::Device,
    ) -> Result<FSRS<B2>> {
        let parameters = check_and_fill_parameters(parameters)?;
        let model = parameters_to_model::<B2>(&parameters, device);

        Ok(FSRS { model })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        &self.model
    }

    pub(crate) fn device(&self) -> B::Device {
        self.model().w.device()
    }
}

fn clamp_safe(value: f32, low: f32, high: f32) -> f32 {
    let low = if low.is_finite() { low } else { 0.0 };
    let high = if high.is_finite() { high } else { low };
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let value = if value.is_finite() { value } else { low };
    value.clamp(low, high)
}

fn clip_fsrs7_parameters(parameters: &mut [f32]) {
    const FSRS7_PARAM_LEN: usize = 34;
    if parameters.len() < FSRS7_PARAM_LEN {
        return;
    }

    // Independent per-parameter clamps = fsrs-autoresearch FSRS_MIN/MAX_VALUES (finished
    // 34-param layout; fail_d_exp dropped from both stability blocks).
    parameters[0] = clamp_safe(parameters[0], 0.0001, 50.0);
    parameters[1] = clamp_safe(parameters[1], 0.0001, 100.0);
    parameters[2] = clamp_safe(parameters[2], 0.0001, 100.0);
    parameters[3] = clamp_safe(parameters[3], 0.0001, 100.0);

    parameters[4] = clamp_safe(parameters[4], 1.0, 10.0);
    parameters[5] = clamp_safe(parameters[5], 0.001, 4.0);
    parameters[6] = clamp_safe(parameters[6], 0.1, 4.0);

    // 7..14 long-term stability-after-review (8 params).
    parameters[7] = clamp_safe(parameters[7], 0.0, 4.0); // sinc_base
    parameters[8] = clamp_safe(parameters[8], 0.0, 1.2); // sinc_s_exp
    parameters[9] = clamp_safe(parameters[9], 0.3, 3.0); // sinc_r_mult
    parameters[10] = clamp_safe(parameters[10], 0.01, 1.5); // fail_mult
    parameters[11] = clamp_safe(parameters[11], 0.1, 1.0); // fail_s_exp
    parameters[12] = clamp_safe(parameters[12], 0.0, 3.5); // fail_r_mult
    parameters[13] = clamp_safe(parameters[13], 0.0, 1.0); // hard_penalty
    parameters[14] = clamp_safe(parameters[14], 1.0, 7.0); // easy_bonus

    // 15..22 short-term stability-after-review (8 params).
    parameters[15] = clamp_safe(parameters[15], 0.0, 4.0); // sinc_base
    parameters[16] = clamp_safe(parameters[16], 0.0, 2.0); // sinc_s_exp
    parameters[17] = clamp_safe(parameters[17], 0.5, 6.0); // sinc_r_mult
    parameters[18] = clamp_safe(parameters[18], 0.001, 1.5); // fail_mult
    parameters[19] = clamp_safe(parameters[19], 0.001, 1.0); // fail_s_exp
    parameters[20] = clamp_safe(parameters[20], 0.0, 5.0); // fail_r_mult
    parameters[21] = clamp_safe(parameters[21], 0.0, 1.0); // hard_penalty
    parameters[22] = clamp_safe(parameters[22], 1.0, 7.0); // easy_bonus

    // 23..30 forgetting curve.
    parameters[23] = clamp_safe(parameters[23], 0.01, 0.25); // decay1
    parameters[24] = clamp_safe(parameters[24], 0.01, 0.95); // decay2
    parameters[25] = clamp_safe(parameters[25], 0.2, 0.85); // base1
    parameters[26] = clamp_safe(parameters[26], 0.5, 0.99); // base2
    parameters[27] = clamp_safe(parameters[27], 0.01, 1.0); // base_weight1
    parameters[28] = clamp_safe(parameters[28], 0.1, 1.0); // base_weight2
    parameters[29] = clamp_safe(parameters[29], 0.0, 0.9); // s_weight_power1
    parameters[30] = clamp_safe(parameters[30], 0.1, 1.1); // s_weight_power2

    // 31..33 difficulty/stability modulation of the curve. ALL-POSITIVE CONVENTION: stored
    // shifted so the range starts at 0; the model formulas offset back (w-0.5 / w-0.3).
    parameters[31] = clamp_safe(parameters[31], 0.0, 1.0); // d_weight  (effective = w - 0.5)
    parameters[32] = clamp_safe(parameters[32], 0.0, 0.6); // d_decay   (effective = w - 0.3)
    parameters[33] = clamp_safe(parameters[33], 0.0, 0.6); // s_decay1  (effective = w - 0.3)

    // Cross-parameter monotonicity, applied AFTER the box clamps: initial stability
    // non-decreasing in rating, base2 >= base1.
    parameters[1] = parameters[1].max(parameters[0]);
    parameters[2] = parameters[2].max(parameters[1]);
    parameters[3] = parameters[3].max(parameters[2]);
    parameters[26] = parameters[26].max(parameters[25]);
}

/// clip_parameters without the allocation (the per-Adam-step clip in train()).
pub(crate) fn clip_parameters_in_place(parameters: &mut [f32]) {
    clip_fsrs7_parameters(parameters);
}

pub(crate) fn clip_parameters(parameters: &Parameters) -> Vec<f32> {
    let mut parameters = parameters.to_vec();
    clip_fsrs7_parameters(&mut parameters);
    parameters
}

pub(crate) fn parameters_to_model<B: Backend>(
    parameters: &Parameters,
    device: &B::Device,
) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::new_with_device(config.clone(), device);
    let clipped = clip_parameters(parameters);
    model.w = Param::from_tensor(Tensor::from_floats(
        TensorData::new(
            clipped.clone(),
            Shape {
                dims: vec![clipped.len()],
            },
        ),
        device,
    ));
    model.version = ModelVersion::Fsrs7;
    model
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = if parameters.is_empty() {
        DEFAULT_PARAMETERS.to_vec()
    } else if parameters.len() == model_v7::PARAM_LEN {
        parameters.to_vec()
    } else {
        return Err(FSRSError::InvalidParameters);
    };
    if parameters.iter().any(|&w| !w.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}
