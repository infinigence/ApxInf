use apxinf_core::Backend;
use apxinf_core::{DType, Device, KvCache, Shape, Tensor};
use apxinf_cuda::kernels::qwen35_gdn::{gated_rms_norm_bf16, Qwen35GdnLayout, Qwen35GdnState};
use apxinf_cuda::kernels::qwen35_w4::{Qwen35W4DeviceProjection, Qwen35W4Layout};
use apxinf_cuda::CudaContext;
use half::bf16;

use super::loader::{
    Bf16TensorPayload, GdnLayerPayload, LinearPayload, PackedLinearPayload,
    Qwen35CheckpointInventory,
};

/// Device-owned W4 projection loaded from one checkpoint prefix.
pub struct Qwen35CheckpointProjection {
    layout: Qwen35W4Layout,
    projection: Qwen35W4DeviceProjection,
}

/// Dense BF16 projection retaining checkpoint row-major `[out_features,
/// in_features]` storage. Projection uses a transposed cuBLAS operand, so
/// startup does not allocate a second host-side matrix.
pub struct Qwen35Bf16Projection {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
}

pub enum Qwen35MixedProjection {
    Packed(Qwen35CheckpointProjection),
    Bf16(Qwen35Bf16Projection),
}

/// Request-local mutable state for one CUDA GDN layer.
pub struct Qwen35CudaGdnState {
    state: Qwen35GdnState,
    device_id: usize,
}

impl Qwen35CudaGdnState {
    pub fn new(
        backend: &apxinf_cuda::CudaBackend,
        layout: Qwen35GdnLayout,
    ) -> Result<Self, String> {
        let device_id = backend.device_id();
        Ok(Self {
            state: Qwen35GdnState::new(backend.context(), layout)
                .map_err(|error| error.to_string())?,
            device_id,
        })
    }

    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    pub const fn position(&self) -> usize {
        self.state.position()
    }

    pub fn reset(&mut self, backend: &apxinf_cuda::CudaBackend) -> Result<(), String> {
        self.state
            .reset(backend.context())
            .map_err(|error| error.to_string())
    }

    /// Refill this state in place from `source` (recycled-session fast path).
    pub fn copy_state_from(
        &mut self,
        source: &Self,
        backend: &apxinf_cuda::CudaBackend,
    ) -> Result<(), String> {
        self.state
            .copy_state_from(&source.state, backend.context())
            .map_err(|error| error.to_string())
    }

    /// Deep-clone the per-request GDN state (bit-identical device copies,
    /// stream-ordered; the session-level fork synchronizes once at the end).
    pub fn deep_clone(&self, backend: &apxinf_cuda::CudaBackend) -> Result<Self, String> {
        Ok(Self {
            state: self
                .state
                .deep_clone(backend.context())
                .map_err(|error| error.to_string())?,
            device_id: self.device_id,
        })
    }
}

/// Device-owned weights and execution for one real Qwen3.5 GDN layer.
pub struct Qwen35CudaGdnLayer {
    layer_index: usize,
    device_id: usize,
    dimensions: Qwen35GdnLayout,
    hidden_size: usize,
    rms_epsilon: f32,
    input_norm: Tensor,
    in_proj_qkv: Qwen35CheckpointProjection,
    in_proj_z: Qwen35CheckpointProjection,
    in_proj_a: Qwen35Bf16Projection,
    in_proj_b: Qwen35Bf16Projection,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out_proj: Qwen35MixedProjection,
    post_attention_norm: Tensor,
    mlp_gate_proj: Qwen35CheckpointProjection,
    mlp_up_proj: Qwen35CheckpointProjection,
    mlp_down_proj: Qwen35CheckpointProjection,
}

impl Qwen35CudaGdnLayer {
    pub fn from_inventory(
        ctx: &CudaContext,
        inventory: &Qwen35CheckpointInventory,
        layer_index: usize,
    ) -> Result<Self, String> {
        if inventory.config.layer_types.get(layer_index) != Some(&super::config::LayerType::Gdn) {
            return Err(format!("layer {layer_index} is not GDN"));
        }
        let payload = inventory
            .read_gdn_layer_payload(layer_index)
            .map_err(|error| error.to_string())?;
        let dimensions = Qwen35GdnLayout::new(
            inventory.config.linear_conv_kernel_dim,
            inventory.config.linear_key_heads,
            inventory.config.linear_value_heads,
            inventory.config.linear_head_dim,
            inventory.config.linear_head_dim,
        )
        .map_err(|error| error.to_string())?;
        Self::from_payload(
            ctx,
            layer_index,
            inventory.config.hidden_size,
            inventory.config.rms_norm_eps,
            dimensions,
            payload,
        )
    }

    fn from_payload(
        ctx: &CudaContext,
        layer_index: usize,
        hidden_size: usize,
        rms_epsilon: f32,
        dimensions: Qwen35GdnLayout,
        payload: GdnLayerPayload,
    ) -> Result<Self, String> {
        if payload.layer_index != layer_index {
            return Err(format!(
                "GDN payload layer {} does not match requested layer {layer_index}",
                payload.layer_index
            ));
        }
        if hidden_size == 0 || !rms_epsilon.is_finite() || rms_epsilon <= 0.0 {
            return Err("GDN hidden size and RMS epsilon must be valid".into());
        }
        let input_norm =
            upload_standard_norm_payload(ctx, &payload.input_norm, hidden_size, "input norm")?;
        let post_attention_norm = upload_standard_norm_payload(
            ctx,
            &payload.post_attention_norm,
            hidden_size,
            "post-attention norm",
        )?;
        let conv_weight = upload_bf16_payload(
            ctx,
            &payload.conv1d_weight,
            vec![dimensions.conv_channels(), 1, dimensions.conv_kernel],
            "GDN convolution weight",
        )?;
        let a_log = upload_bf16_payload(
            ctx,
            &payload.a_log,
            vec![dimensions.value_heads],
            "GDN A_log",
        )?;
        let dt_bias = upload_bf16_payload(
            ctx,
            &payload.dt_bias,
            vec![dimensions.value_heads],
            "GDN dt_bias",
        )?;
        let norm = upload_bf16_payload(
            ctx,
            &payload.norm,
            vec![dimensions.value_dim],
            "GDN gated norm",
        )?;

        Ok(Self {
            layer_index,
            device_id: ctx.device_id(),
            dimensions,
            hidden_size,
            rms_epsilon,
            input_norm,
            in_proj_qkv: Qwen35CheckpointProjection::from_packed_payload(
                ctx,
                &payload.in_proj_qkv,
            )?,
            in_proj_z: Qwen35CheckpointProjection::from_packed_payload(ctx, &payload.in_proj_z)?,
            in_proj_a: Qwen35Bf16Projection::from_payload(ctx, &payload.in_proj_a)?,
            in_proj_b: Qwen35Bf16Projection::from_payload(ctx, &payload.in_proj_b)?,
            conv_weight,
            a_log,
            dt_bias,
            norm,
            out_proj: Qwen35MixedProjection::from_payload(ctx, payload.out_proj)?,
            post_attention_norm,
            mlp_gate_proj: Qwen35CheckpointProjection::from_packed_payload(
                ctx,
                &payload.mlp_gate_proj,
            )?,
            mlp_up_proj: Qwen35CheckpointProjection::from_packed_payload(
                ctx,
                &payload.mlp_up_proj,
            )?,
            mlp_down_proj: Qwen35CheckpointProjection::from_packed_payload(
                ctx,
                &payload.mlp_down_proj,
            )?,
        })
    }

    pub const fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    pub const fn dimensions(&self) -> Qwen35GdnLayout {
        self.dimensions
    }

    pub fn prefill(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        state: &mut Qwen35CudaGdnState,
    ) -> Result<Tensor, String> {
        let expected_device = Device::Cuda(self.device_id);
        let hidden_dims = hidden.shape().dims();
        if backend.device() != expected_device
            || state.device_id != self.device_id
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || hidden_dims.len() != 2
            || hidden_dims[0] == 0
            || hidden_dims[1] != self.hidden_size
        {
            return Err(format!(
                "GDN prefill requires CUDA{} BF16 [rows,{}] hidden with rows > 0 and matching state/backend, got {:?} {} on {:?}",
                self.device_id,
                self.hidden_size,
                hidden_dims,
                hidden.dtype(),
                hidden.device()
            ));
        }
        if state.state.layout() != self.dimensions {
            return Err("GDN state layout does not match layer".into());
        }
        let qkv_width = self.dimensions.conv_channels();
        let query_width = self.dimensions.query_width();
        let value_width = self.dimensions.value_width();
        if self.in_proj_qkv.layout.out_features != qkv_width
            || self.in_proj_z.layout.out_features != value_width
            || self.in_proj_a.in_features() != self.hidden_size
            || self.in_proj_a.out_features() != self.dimensions.value_heads
            || self.in_proj_b.in_features() != self.hidden_size
            || self.in_proj_b.out_features() != self.dimensions.value_heads
            || self.out_proj.in_features() != value_width
            || self.out_proj.out_features() != self.hidden_size
            || self.mlp_gate_proj.layout.in_features != self.hidden_size
            || self.mlp_up_proj.layout.in_features != self.hidden_size
            || self.mlp_gate_proj.layout.out_features != self.mlp_up_proj.layout.out_features
            || self.mlp_down_proj.layout.in_features != self.mlp_gate_proj.layout.out_features
            || self.mlp_down_proj.layout.out_features != self.hidden_size
        {
            return Err("GDN layer projection dimensions do not match layer layout".into());
        }

        let ctx = backend.context();
        let mut convolution_committed = false;
        let mut recurrent_committed = false;
        let result = (|| -> Result<Tensor, String> {
            let normalized = backend
                .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
                .map_err(|error| error.to_string())?;
            let qkv = self
                .in_proj_qkv
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let z = self
                .in_proj_z
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let a = self
                .in_proj_a
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let b = self
                .in_proj_b
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let convolved = state
                .state
                .causal_conv_silu_prefill(ctx, &qkv, &self.conv_weight)
                .map_err(|error| error.to_string())?;
            convolution_committed = true;
            let key_width = self.dimensions.key_heads * self.dimensions.key_dim;
            let query = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                ctx,
                &convolved,
                0,
                query_width,
            )
            .map_err(|error| error.to_string())?;
            let key = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                ctx,
                &convolved,
                query_width,
                query_width,
            )
            .map_err(|error| error.to_string())?;
            let value = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                ctx,
                &convolved,
                key_width * 2,
                value_width,
            )
            .map_err(|error| error.to_string())?;
            let recurrent = state
                .state
                .gated_delta_prefill(
                    ctx,
                    &query,
                    &key,
                    &value,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                )
                .map_err(|error| error.to_string())?;
            recurrent_committed = true;
            let gated = gated_rms_norm_bf16(
                ctx,
                &recurrent,
                &z,
                &self.norm,
                self.dimensions.value_heads,
                self.dimensions.value_dim,
                self.rms_epsilon,
            )
            .map_err(|error| error.to_string())?;
            let attention_update = self
                .out_proj
                .project(ctx, &gated)
                .map_err(|error| error.to_string())?;
            let residual = backend
                .add(hidden, &attention_update)
                .map_err(|error| error.to_string())?;
            let mlp_input = backend
                .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
                .map_err(|error| error.to_string())?;
            let mlp_gate = self
                .mlp_gate_proj
                .project(ctx, &mlp_input)
                .map_err(|error| error.to_string())?;
            let mlp_up = self
                .mlp_up_proj
                .project(ctx, &mlp_input)
                .map_err(|error| error.to_string())?;
            let mlp_hidden = backend
                .mul(
                    &backend.silu(&mlp_gate).map_err(|error| error.to_string())?,
                    &mlp_up,
                )
                .map_err(|error| error.to_string())?;
            let mlp_update = self
                .mlp_down_proj
                .project(ctx, &mlp_hidden)
                .map_err(|error| error.to_string())?;
            let output = backend
                .add(&residual, &mlp_update)
                .map_err(|error| error.to_string())?;
            apxinf_cuda::kernels::qwen35_gdn::require_finite_bf16(
                ctx,
                &output,
                "GDN prefill output",
            )
            .map_err(|error| error.to_string())?;
            Ok(output)
        })();

        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                let mut rollback_errors = Vec::new();
                if recurrent_committed {
                    if let Err(rollback) = state.state.rollback_last_recurrent() {
                        rollback_errors.push(rollback.to_string());
                    }
                }
                if convolution_committed {
                    if let Err(rollback) = state.state.rollback_last_convolution() {
                        rollback_errors.push(rollback.to_string());
                    }
                }
                if rollback_errors.is_empty() {
                    Err(error)
                } else {
                    Err(format!(
                        "{error}; GDN state rollback failed: {}",
                        rollback_errors.join("; ")
                    ))
                }
            }
        }
    }

    pub fn decode_token(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        state: &mut Qwen35CudaGdnState,
    ) -> Result<Tensor, String> {
        let expected_device = Device::Cuda(self.device_id);
        if backend.device() != expected_device
            || state.device_id != self.device_id
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims() != [1, self.hidden_size]
        {
            return Err(format!(
                "GDN decode requires CUDA{} BF16 [1,{}] hidden with matching state/backend, got {:?} {} on {:?}",
                self.device_id,
                self.hidden_size,
                hidden.shape().dims(),
                hidden.dtype(),
                hidden.device()
            ));
        }
        if state.state.layout() != self.dimensions {
            return Err("GDN state layout does not match layer".into());
        }

        let ctx = backend.context();
        let mut convolution_committed = false;
        let mut recurrent_committed = false;
        let result = (|| -> Result<Tensor, String> {
            let normalized = backend
                .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
                .map_err(|error| error.to_string())?;
            let qkv = self
                .in_proj_qkv
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let z = self
                .in_proj_z
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let a = self
                .in_proj_a
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let b = self
                .in_proj_b
                .project(ctx, &normalized)
                .map_err(|error| error.to_string())?;
            let convolved = state
                .state
                .causal_conv_silu(ctx, &qkv, &self.conv_weight)
                .map_err(|error| error.to_string())?;
            convolution_committed = true;
            let key_width = self.dimensions.key_heads * self.dimensions.key_dim;
            let value_width = self.dimensions.value_width();
            let query = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                ctx, &convolved, 0, key_width,
            )
            .map_err(|error| error.to_string())?;
            let key = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                ctx, &convolved, key_width, key_width,
            )
            .map_err(|error| error.to_string())?;
            let value = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                ctx,
                &convolved,
                key_width * 2,
                value_width,
            )
            .map_err(|error| error.to_string())?;
            let recurrent = state
                .state
                .gated_delta_step(
                    ctx,
                    &query,
                    &key,
                    &value,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                )
                .map_err(|error| error.to_string())?;
            recurrent_committed = true;
            let gated = gated_rms_norm_bf16(
                ctx,
                &recurrent,
                &z,
                &self.norm,
                self.dimensions.value_heads,
                self.dimensions.value_dim,
                self.rms_epsilon,
            )
            .map_err(|error| error.to_string())?;
            let attention_update = self
                .out_proj
                .project(ctx, &gated)
                .map_err(|error| error.to_string())?;
            let residual = backend
                .add(hidden, &attention_update)
                .map_err(|error| error.to_string())?;
            let mlp_input = backend
                .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
                .map_err(|error| error.to_string())?;
            let mlp_gate = self
                .mlp_gate_proj
                .project(ctx, &mlp_input)
                .map_err(|error| error.to_string())?;
            let mlp_up = self
                .mlp_up_proj
                .project(ctx, &mlp_input)
                .map_err(|error| error.to_string())?;
            let mlp_hidden = backend
                .mul(
                    &backend.silu(&mlp_gate).map_err(|error| error.to_string())?,
                    &mlp_up,
                )
                .map_err(|error| error.to_string())?;
            let mlp_update = self
                .mlp_down_proj
                .project(ctx, &mlp_hidden)
                .map_err(|error| error.to_string())?;
            let output = backend
                .add(&residual, &mlp_update)
                .map_err(|error| error.to_string())?;
            apxinf_cuda::kernels::qwen35_gdn::require_finite_bf16(
                ctx,
                &output,
                "GDN decode output",
            )
            .map_err(|error| error.to_string())?;
            Ok(output)
        })();

        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                let mut rollback_errors = Vec::new();
                if recurrent_committed {
                    if let Err(rollback) = state.state.rollback_last_recurrent() {
                        rollback_errors.push(rollback.to_string());
                    }
                }
                if convolution_committed {
                    if let Err(rollback) = state.state.rollback_last_convolution() {
                        rollback_errors.push(rollback.to_string());
                    }
                }
                if rollback_errors.is_empty() {
                    Err(error)
                } else {
                    Err(format!(
                        "{error}; GDN state rollback failed: {}",
                        rollback_errors.join("; ")
                    ))
                }
            }
        }
    }

    /// Batched decode: advance `states.len()` independent requests by one
    /// token each. RMSNorm, the qkv/z/a/b projections and the MLP run as one
    /// `[M, ...]` batch (per-row bit-identical kernels); the convolution,
    /// gated-delta recurrence and gated norm run per request on row slices.
    /// Outputs are bit-identical to M sequential `decode_token` calls.
    ///
    /// Error semantics: a failure poisons the whole batch (earlier rows may
    /// have committed conv/recurrent state); callers must drop every session
    /// in the batch. No per-row rollback is attempted.
    pub fn decode_token_batch(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        states: &mut [&mut Qwen35CudaGdnState],
    ) -> Result<Tensor, String> {
        let batch = states.len();
        let expected_device = Device::Cuda(self.device_id);
        if batch == 0
            || backend.device() != expected_device
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims() != [batch, self.hidden_size]
        {
            return Err(format!(
                "GDN batched decode requires CUDA{} BF16 [{batch},{}] hidden",
                self.device_id, self.hidden_size
            ));
        }
        for (index, state) in states.iter().enumerate() {
            if state.device_id != self.device_id || state.state.layout() != self.dimensions {
                return Err(format!("GDN batched decode state {index} mismatch"));
            }
        }

        let ctx = backend.context();
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        // The packed-W4 projections assign one CUDA block per (row, out)
        // pair, so a batched call is bit-identical per row and safe to share.
        // The BF16 in_proj_a/in_proj_b run through cuBLAS, whose kernel
        // choice (and per-row accumulation order) changes with M — those run
        // per request on the normalized row to stay bit-identical.
        let qkv_all = self
            .in_proj_qkv
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let z_all = self
            .in_proj_z
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;

        let key_width = self.dimensions.key_heads * self.dimensions.key_dim;
        let value_width = self.dimensions.value_width();
        // a/b run through cuBLAS; keep them per request (bit-identical to
        // the serial path) and stack the [1, heads] rows for the batched
        // recurrence kernel.
        let mut a_rows: Vec<Tensor> = Vec::with_capacity(batch);
        let mut b_rows: Vec<Tensor> = Vec::with_capacity(batch);
        for index in 0..batch {
            let normalized_row = matrix_row(ctx, &normalized, index)?;
            a_rows.push(
                self.in_proj_a
                    .project(ctx, &normalized_row)
                    .map_err(|error| error.to_string())?,
            );
            b_rows.push(
                self.in_proj_b
                    .project(ctx, &normalized_row)
                    .map_err(|error| error.to_string())?,
            );
        }
        let a_refs: Vec<&Tensor> = a_rows.iter().collect();
        let b_refs: Vec<&Tensor> = b_rows.iter().collect();
        let a_all = apxinf_cuda::kernels::elementwise::stack_rows_bf16(ctx, &a_refs)
            .map_err(|error| error.to_string())?;
        let b_all = apxinf_cuda::kernels::elementwise::stack_rows_bf16(ctx, &b_refs)
            .map_err(|error| error.to_string())?;

        // Batched convolution/recurrence/gated-norm: one kernel per stage for
        // the whole batch, with per-request ring/state pointer arrays. The
        // per-request math inside each kernel is copied verbatim from the
        // serial kernels, so outputs and state updates stay bit-identical.
        let mut gdn_states: Vec<&mut apxinf_cuda::kernels::qwen35_gdn::Qwen35GdnState> =
            states.iter_mut().map(|state| &mut state.state).collect();
        let convolved = apxinf_cuda::kernels::qwen35_gdn::causal_conv_silu_batch(
            ctx,
            &mut gdn_states,
            &qkv_all,
            &self.conv_weight,
        )
        .map_err(|error| error.to_string())?;
        let query =
            apxinf_cuda::kernels::elementwise::slice_columns_bf16(ctx, &convolved, 0, key_width)
                .map_err(|error| error.to_string())?;
        let key = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx, &convolved, key_width, key_width,
        )
        .map_err(|error| error.to_string())?;
        let value = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &convolved,
            key_width * 2,
            value_width,
        )
        .map_err(|error| error.to_string())?;
        let recurrent = apxinf_cuda::kernels::qwen35_gdn::gated_delta_step_batch(
            ctx,
            &mut gdn_states,
            &query,
            &key,
            &value,
            &a_all,
            &b_all,
            &self.a_log,
            &self.dt_bias,
        )
        .map_err(|error| error.to_string())?;
        let gated = gated_rms_norm_bf16(
            ctx,
            &recurrent,
            &z_all,
            &self.norm,
            self.dimensions.value_heads,
            self.dimensions.value_dim,
            self.rms_epsilon,
        )
        .map_err(|error| error.to_string())?;
        // out_proj may be a BF16 (cuBLAS) matrix on this checkpoint; keep it
        // per-row for bit equality in that case, batch when packed W4.
        let attention_update = match &self.out_proj {
            Qwen35MixedProjection::Packed(projection) => projection
                .project(ctx, &gated)
                .map_err(|error| error.to_string())?,
            Qwen35MixedProjection::Bf16(projection) => {
                let mut update_rows: Vec<Tensor> = Vec::with_capacity(batch);
                for index in 0..batch {
                    let gated_row = matrix_row(ctx, &gated, index)?;
                    update_rows.push(
                        projection
                            .project(ctx, &gated_row)
                            .map_err(|error| error.to_string())?,
                    );
                }
                let update_refs: Vec<&Tensor> = update_rows.iter().collect();
                apxinf_cuda::kernels::elementwise::stack_rows_bf16(ctx, &update_refs)
                    .map_err(|error| error.to_string())?
            }
        };
        let residual = backend
            .add(hidden, &attention_update)
            .map_err(|error| error.to_string())?;
        let mlp_input = backend
            .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let mlp_gate = self
            .mlp_gate_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_up = self
            .mlp_up_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_hidden = backend
            .mul(
                &backend.silu(&mlp_gate).map_err(|error| error.to_string())?,
                &mlp_up,
            )
            .map_err(|error| error.to_string())?;
        let mlp_update = self
            .mlp_down_proj
            .project(ctx, &mlp_hidden)
            .map_err(|error| error.to_string())?;
        let output = backend
            .add(&residual, &mlp_update)
            .map_err(|error| error.to_string())?;
        apxinf_cuda::kernels::qwen35_gdn::require_finite_bf16(
            ctx,
            &output,
            "GDN batched decode output",
        )
        .map_err(|error| error.to_string())?;
        Ok(output)
    }
}

/// Zero-copy view of row `index` of a `[rows, width]` BF16 device matrix as
/// `[1, width]`. Shares storage with the batch tensor, so the row is the
/// batch entry itself (trivially bit-identical) and per-request consumers
/// pay no copy or allocation.
fn matrix_row(_ctx: &CudaContext, matrix: &Tensor, index: usize) -> Result<Tensor, String> {
    apxinf_cuda::kernels::elementwise::row_view_bf16(matrix, index)
        .map_err(|error| format!("matrix row view: {error}"))
}

fn upload_bf16_payload(
    ctx: &CudaContext,
    payload: &Bf16TensorPayload,
    expected_shape: Vec<usize>,
    label: &str,
) -> Result<Tensor, String> {
    if payload.shape != expected_shape {
        return Err(format!(
            "{label} shape {:?} does not match expected {:?}",
            payload.shape, expected_shape
        ));
    }
    let host = Tensor::from_bf16(Shape::new(expected_shape), &payload.values)
        .map_err(|error| format!("create {label}: {error}"))?;
    apxinf_cuda::transfers::to_cuda(&host, ctx.device_id())
        .map_err(|error| format!("upload {label}: {error}"))
}

pub(crate) fn upload_standard_norm_payload(
    ctx: &CudaContext,
    payload: &Bf16TensorPayload,
    width: usize,
    label: &str,
) -> Result<Tensor, String> {
    if payload.shape != [width] {
        return Err(format!(
            "{label} shape {:?} does not match expected [{width}]",
            payload.shape
        ));
    }
    let values = standard_rms_norm_weights(&payload.values);
    let host = Tensor::from_bf16(Shape::new(vec![width]), &values)
        .map_err(|error| format!("create {label}: {error}"))?;
    apxinf_cuda::transfers::to_cuda(&host, ctx.device_id())
        .map_err(|error| format!("upload {label}: {error}"))
}

impl Qwen35Bf16Projection {
    pub(crate) fn from_payload(
        ctx: &CudaContext,
        payload: &Bf16TensorPayload,
    ) -> Result<Self, String> {
        if payload.shape.len() != 2 || payload.shape[0] == 0 || payload.shape[1] == 0 {
            return Err(format!(
                "BF16 projection must have non-empty rank-2 shape, got {:?}",
                payload.shape
            ));
        }
        let out_features = payload.shape[0];
        let in_features = payload.shape[1];
        let expected = out_features
            .checked_mul(in_features)
            .ok_or_else(|| "BF16 projection element count overflow".to_string())?;
        if payload.values.len() != expected {
            return Err(format!(
                "BF16 projection payload has {} values, expected {expected}",
                payload.values.len()
            ));
        }
        let host = Tensor::from_bf16(Shape::new(vec![out_features, in_features]), &payload.values)
            .map_err(|error| format!("create BF16 projection tensor: {error}"))?;
        let weight = apxinf_cuda::transfers::to_cuda(&host, ctx.device_id())
            .map_err(|error| format!("upload BF16 projection tensor: {error}"))?;
        Ok(Self {
            in_features,
            out_features,
            weight,
        })
    }

    pub(crate) fn project(
        &self,
        ctx: &CudaContext,
        activation: &Tensor,
    ) -> apxinf_core::Result<Tensor> {
        if activation.device() != Device::Cuda(ctx.device_id())
            || activation.dtype() != DType::BF16
            || activation.shape().dims().len() != 2
            || activation.shape().dims()[1] != self.in_features
        {
            return Err(apxinf_core::Error::Other(format!(
                "BF16 projection activation must be CUDA BF16 [rows, {}], got {:?} {} on {:?}",
                self.in_features,
                activation.shape().dims(),
                activation.dtype(),
                activation.device()
            )));
        }
        apxinf_cuda::kernels::gemm::project_checkpoint_bf16(ctx, activation, &self.weight)
    }

    const fn in_features(&self) -> usize {
        self.in_features
    }

    const fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Qwen35MixedProjection {
    fn from_payload(ctx: &CudaContext, payload: LinearPayload) -> Result<Self, String> {
        match payload {
            LinearPayload::Packed(payload) => {
                Qwen35CheckpointProjection::from_packed_payload(ctx, &payload).map(Self::Packed)
            }
            LinearPayload::Bf16(payload) => {
                Qwen35Bf16Projection::from_payload(ctx, &payload).map(Self::Bf16)
            }
        }
    }

    fn project(&self, ctx: &CudaContext, activation: &Tensor) -> apxinf_core::Result<Tensor> {
        match self {
            Self::Packed(projection) => projection.project(ctx, activation),
            Self::Bf16(projection) => projection.project(ctx, activation),
        }
    }

    const fn in_features(&self) -> usize {
        match self {
            Self::Packed(projection) => projection.layout.in_features,
            Self::Bf16(projection) => projection.in_features(),
        }
    }

    const fn out_features(&self) -> usize {
        match self {
            Self::Packed(projection) => projection.layout.out_features,
            Self::Bf16(projection) => projection.out_features(),
        }
    }
}

fn standard_rms_norm_weights(values: &[bf16]) -> Vec<bf16> {
    values
        .iter()
        .map(|value| bf16::from_f32(1.0 + value.to_f32()))
        .collect()
}

fn upload_standard_rms_norm(
    ctx: &CudaContext,
    values: &[bf16],
    width: usize,
    label: &str,
) -> Result<Tensor, String> {
    if values.len() != width {
        return Err(format!(
            "{label} has {} values, expected {width}",
            values.len()
        ));
    }
    let values = standard_rms_norm_weights(values);
    let host =
        Tensor::from_bf16(Shape::new(vec![width]), &values).map_err(|error| error.to_string())?;
    apxinf_cuda::transfers::to_cuda(&host, ctx.device_id()).map_err(|error| error.to_string())
}

/// Device-owned weights for one real full-attention layer.
///
/// The packed projections remain in their checkpoint layout on device. Norm
/// vectors are uploaded as BF16 tensors without creating an expanded model
/// copy. This is a weight bundle only; execution is added by the model owner.
pub struct Qwen35CudaFullAttentionLayer {
    layer_index: usize,
    device_id: usize,
    q_proj: Qwen35CheckpointProjection,
    k_proj: Qwen35CheckpointProjection,
    v_proj: Qwen35CheckpointProjection,
    o_proj: Qwen35CheckpointProjection,
    gate_proj: Qwen35CheckpointProjection,
    up_proj: Qwen35CheckpointProjection,
    down_proj: Qwen35CheckpointProjection,
    input_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_norm: Tensor,
    n_query_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rms_epsilon: f32,
}

/// Request-local state for one CUDA full-attention layer.
///
/// The cache is deliberately one-layer and BF16-only. The complete model
/// executor will own one such cache per full-attention layer; this standalone
/// state keeps the layer smoke and its position semantics independently
/// testable without introducing a CPU mirror.
pub struct Qwen35CudaFullAttentionState {
    cache: apxinf_cuda::CudaKVCache,
    max_seq_len: usize,
    device_id: usize,
}

impl Qwen35CudaFullAttentionState {
    pub fn new(backend: &apxinf_cuda::CudaBackend, max_seq_len: usize) -> Result<Self, String> {
        if max_seq_len == 0 {
            return Err("full-attention max sequence length must be non-zero".into());
        }
        let device_id = backend.device_id();
        let cache = apxinf_cuda::CudaKVCache::new_bf16(device_id, 1, 4, 256, max_seq_len)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            cache,
            max_seq_len,
            device_id,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.cache.seq_len()
    }

    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn clear(&mut self) -> Result<(), String> {
        apxinf_core::KvCache::clear(&mut self.cache).map_err(|error| error.to_string())
    }

    /// Refill this state in place from `source` (recycled-session fast path).
    pub fn copy_state_from(
        &mut self,
        source: &Self,
        backend: &apxinf_cuda::CudaBackend,
    ) -> Result<(), String> {
        if self.max_seq_len != source.max_seq_len {
            return Err("KV state refill requires identical capacity".into());
        }
        self.cache
            .copy_from(&source.cache, backend.context())
            .map_err(|error| error.to_string())
    }

    /// Deep-clone the per-request KV state (bit-identical device copies,
    /// stream-ordered; the session-level fork synchronizes once at the end).
    pub fn deep_clone(&self, backend: &apxinf_cuda::CudaBackend) -> Result<Self, String> {
        Ok(Self {
            cache: self
                .cache
                .deep_clone(backend.context())
                .map_err(|error| error.to_string())?,
            max_seq_len: self.max_seq_len,
            device_id: self.device_id,
        })
    }
}

pub struct Qwen35AttentionProjectionTensors {
    pub q_gate: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

impl Qwen35CudaFullAttentionLayer {
    pub fn from_inventory(
        ctx: &CudaContext,
        inventory: &Qwen35CheckpointInventory,
        layer_index: usize,
    ) -> Result<Self, String> {
        if inventory.config.layer_types.get(layer_index)
            != Some(&super::config::LayerType::FullAttention)
        {
            return Err(format!("layer {layer_index} is not full-attention"));
        }
        let prefix = format!("model.language_model.layers.{layer_index}");
        let projection = |name: &str| {
            Qwen35CheckpointProjection::from_inventory(ctx, inventory, &format!("{prefix}.{name}"))
        };
        let norm = |name: &str, width: usize| -> Result<Tensor, String> {
            let label = format!("{prefix}.{name}");
            let values = inventory
                .read_tensor_bf16_values(&label)
                .map_err(|error| error.to_string())?;
            upload_standard_rms_norm(ctx, &values, width, &label)
        };

        Ok(Self {
            layer_index,
            device_id: ctx.device_id(),
            q_proj: projection("self_attn.q_proj")?,
            k_proj: projection("self_attn.k_proj")?,
            v_proj: projection("self_attn.v_proj")?,
            o_proj: projection("self_attn.o_proj")?,
            gate_proj: projection("mlp.gate_proj")?,
            up_proj: projection("mlp.up_proj")?,
            down_proj: projection("mlp.down_proj")?,
            input_norm: norm("input_layernorm.weight", inventory.config.hidden_size)?,
            q_norm: norm(
                "self_attn.q_norm.weight",
                inventory.config.full_attention_head_dim,
            )?,
            k_norm: norm(
                "self_attn.k_norm.weight",
                inventory.config.full_attention_head_dim,
            )?,
            post_attention_norm: norm(
                "post_attention_layernorm.weight",
                inventory.config.hidden_size,
            )?,
            n_query_heads: inventory.config.full_attention_heads,
            n_kv_heads: inventory.config.full_attention_kv_heads,
            head_dim: inventory.config.full_attention_head_dim,
            rotary_dim: inventory.config.partial_rotary_dim(),
            rope_theta: super::attention::QWEN35_ROPE_THETA,
            rms_epsilon: inventory.config.rms_norm_eps,
        })
    }

    pub const fn layer_index(&self) -> usize {
        self.layer_index
    }
    pub const fn device_id(&self) -> usize {
        self.device_id
    }
    pub const fn q_layout(&self) -> Qwen35W4Layout {
        self.q_proj.layout
    }
    pub const fn k_layout(&self) -> Qwen35W4Layout {
        self.k_proj.layout
    }
    pub const fn v_layout(&self) -> Qwen35W4Layout {
        self.v_proj.layout
    }
    pub const fn o_layout(&self) -> Qwen35W4Layout {
        self.o_proj.layout
    }
    pub const fn gate_layout(&self) -> Qwen35W4Layout {
        self.gate_proj.layout
    }
    pub const fn up_layout(&self) -> Qwen35W4Layout {
        self.up_proj.layout
    }
    pub const fn down_layout(&self) -> Qwen35W4Layout {
        self.down_proj.layout
    }
    pub fn norm_shapes(&self) -> [[usize; 1]; 4] {
        [
            [self.input_norm.shape().dims()[0]],
            [self.q_norm.shape().dims()[0]],
            [self.k_norm.shape().dims()[0]],
            [self.post_attention_norm.shape().dims()[0]],
        ]
    }

    /// Run the real layer-3 input normalization and packed q/k/v projections.
    /// The result stays on the selected CUDA device and preserves the q/gate
    /// concatenation emitted by the checkpoint's q projection.
    pub fn project_attention_inputs(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
    ) -> Result<Qwen35AttentionProjectionTensors, String> {
        if hidden.device() != Device::Cuda(self.device_id)
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims().len() != 2
            || hidden.shape().dims()[1] != self.q_layout().in_features
        {
            return Err(format!(
                "full-attention hidden must be CUDA BF16 [rows, {}], got {:?} {} on {:?}",
                self.q_layout().in_features,
                hidden.shape().dims(),
                hidden.dtype(),
                hidden.device()
            ));
        }
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        Ok(Qwen35AttentionProjectionTensors {
            q_gate: self
                .q_proj
                .project(backend.context(), &normalized)
                .map_err(|error| error.to_string())?,
            k: self
                .k_proj
                .project(backend.context(), &normalized)
                .map_err(|error| error.to_string())?,
            v: self
                .v_proj
                .project(backend.context(), &normalized)
                .map_err(|error| error.to_string())?,
        })
    }

    /// Execute one causal CUDA BF16 token step, including the complete
    /// full-attention and SwiGLU layer order from the reference semantics.
    pub fn decode_token(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        position: usize,
        state: &mut Qwen35CudaFullAttentionState,
    ) -> Result<Tensor, String> {
        let expected_device = Device::Cuda(self.device_id);
        if backend.device() != expected_device
            || state.device_id != self.device_id
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims() != [1, self.input_norm.shape().dims()[0]]
        {
            return Err(format!(
                "full-attention decode requires CUDA{} BF16 [1,{}] hidden, matching state/backend; got {:?} {} on {:?}",
                self.device_id,
                self.input_norm.shape().dims()[0],
                hidden.shape().dims(),
                hidden.dtype(),
                hidden.device()
            ));
        }
        if position != state.seq_len() {
            return Err(format!(
                "full-attention position {position} does not match local KV length {}",
                state.seq_len()
            ));
        }
        if position >= state.max_seq_len {
            return Err(format!(
                "full-attention position {position} exceeds max sequence length {}",
                state.max_seq_len
            ));
        }
        if position > u32::MAX as usize {
            return Err("full-attention position exceeds CUDA RoPE range".into());
        }
        if self.n_query_heads != 24
            || self.n_kv_heads != 4
            || self.head_dim != 256
            || self.rotary_dim != 64
            || self.q_layout().out_features != 2 * self.n_query_heads * self.head_dim
            || self.k_layout().out_features != self.n_kv_heads * self.head_dim
            || self.v_layout().out_features != self.n_kv_heads * self.head_dim
            || self.o_layout().out_features != self.input_norm.shape().dims()[0]
            || self.gate_layout().out_features != self.up_layout().out_features
            || self.down_layout().out_features != self.input_norm.shape().dims()[0]
        {
            return Err(
                "checkpoint full-attention dimensions violate the CUDA layer contract".into(),
            );
        }

        let ctx = backend.context();
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let q_gate = self
            .q_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let k = self
            .k_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let v = self
            .v_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;

        let q_width = self.n_query_heads * self.head_dim;
        let q_gate_heads = q_gate
            .reshape(vec![1, self.n_query_heads, self.head_dim * 2])
            .map_err(|error| error.to_string())?;
        let q_gate_heads = q_gate_heads
            .reshape(vec![self.n_query_heads, self.head_dim * 2])
            .map_err(|error| error.to_string())?;
        let q = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &q_gate_heads,
            0,
            self.head_dim,
        )
        .map_err(|error| error.to_string())?;
        let gate = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &q_gate_heads,
            self.head_dim,
            self.head_dim,
        )
        .map_err(|error| error.to_string())?;
        let q = q
            .reshape(vec![1, q_width])
            .map_err(|error| error.to_string())?;
        let gate = gate
            .reshape(vec![1, q_width])
            .map_err(|error| error.to_string())?;
        let k = k
            .reshape(vec![1, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let v = v
            .reshape(vec![1, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;

        let q = q
            .reshape(vec![self.n_query_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.q_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![1, self.n_query_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let k = k
            .reshape(vec![self.n_kv_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.k_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![1, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let q = apxinf_cuda::kernels::rope::apply_partial_batched(
            ctx,
            &q,
            self.n_query_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            position as u32,
        )
        .map_err(|error| error.to_string())?;
        let k = apxinf_cuda::kernels::rope::apply_partial_batched(
            ctx,
            &k,
            self.n_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            position as u32,
        )
        .map_err(|error| error.to_string())?;

        state
            .cache
            .append(ctx, 0, &k, &v, 1)
            .map_err(|error| error.to_string())?;
        let attended = backend
            .sdpa_decode(
                &q,
                &mut state.cache,
                0,
                self.n_query_heads,
                self.n_kv_heads,
                self.head_dim,
                position + 1,
                state.max_seq_len,
            )
            .map_err(|error| error.to_string())?;
        let gate = apxinf_cuda::kernels::activation::sigmoid(ctx, &gate)
            .map_err(|error| error.to_string())?;
        let gated = backend
            .mul(&attended, &gate)
            .map_err(|error| error.to_string())?;
        let attention_update = self
            .o_proj
            .project(ctx, &gated)
            .map_err(|error| error.to_string())?;
        let residual = backend
            .add(hidden, &attention_update)
            .map_err(|error| error.to_string())?;
        let mlp_input = backend
            .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let mlp_gate = self
            .gate_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_up = self
            .up_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_gate = backend.silu(&mlp_gate).map_err(|error| error.to_string())?;
        let mlp_hidden = backend
            .mul(&mlp_gate, &mlp_up)
            .map_err(|error| error.to_string())?;
        let mlp_update = self
            .down_proj
            .project(ctx, &mlp_hidden)
            .map_err(|error| error.to_string())?;
        let output = backend
            .add(&residual, &mlp_update)
            .map_err(|error| error.to_string())?;
        apxinf_cuda::kernels::qwen35_gdn::require_finite_bf16(
            ctx,
            &output,
            "full-attention decode output",
        )
        .map_err(|error| error.to_string())?;
        state.cache.advance(1);
        Ok(output)
    }

    /// Batched decode: advance `states.len()` independent requests by one
    /// token each. The shared weight-bound work (RMSNorm, all W4/BF16
    /// projections, the MLP) runs as one `[M, ...]` batch whose per-row
    /// kernels are bit-identical to the single-request path; the per-request
    /// segment (head reshape, QK norm, RoPE at each request's own position,
    /// KV append, flash decode, sigmoid gate) runs as an M-iteration loop on
    /// row slices. Outputs are therefore bit-identical to M sequential
    /// `decode_token` calls.
    ///
    /// Error semantics: any failure poisons the whole batch (KV appends may
    /// have committed for earlier rows); callers must drop every session in
    /// the batch, mirroring the single-request drop-on-failure rule.
    pub fn decode_token_batch(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        positions: &[usize],
        states: &mut [&mut Qwen35CudaFullAttentionState],
    ) -> Result<Tensor, String> {
        let batch = states.len();
        let hidden_size = self.input_norm.shape().dims()[0];
        let expected_device = Device::Cuda(self.device_id);
        if batch == 0
            || positions.len() != batch
            || backend.device() != expected_device
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || hidden.shape().dims() != [batch, hidden_size]
        {
            return Err(format!(
                "full-attention batched decode requires CUDA{} BF16 [{batch},{hidden_size}] hidden and one position per state",
                self.device_id
            ));
        }
        for (index, state) in states.iter().enumerate() {
            if state.device_id != self.device_id {
                return Err(format!(
                    "batched decode state {index} is on the wrong device"
                ));
            }
            if positions[index] != state.seq_len() {
                return Err(format!(
                    "batched decode position {} does not match state {index} KV length {}",
                    positions[index],
                    state.seq_len()
                ));
            }
            if positions[index] >= state.max_seq_len || positions[index] > u32::MAX as usize {
                return Err(format!(
                    "batched decode position {} out of range for state {index}",
                    positions[index]
                ));
            }
        }

        let ctx = backend.context();
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let q_gate_all = self
            .q_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let k_all = self
            .k_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let v_all = self
            .v_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;

        let q_width = self.n_query_heads * self.head_dim;
        let k_width = self.n_kv_heads * self.head_dim;
        // Shared, per-row-independent stages run once for the whole batch:
        // the q/gate column split ([batch*heads, 2*dim] view is the same
        // memory layout the serial path reshapes through), the QK RMSNorms
        // (row-independent kernel), the per-row-position partial RoPE, and
        // the sigmoid gate. Only KV append and flash decode stay per request.
        let q_gate_heads = q_gate_all
            .reshape(vec![batch * self.n_query_heads, self.head_dim * 2])
            .map_err(|error| error.to_string())?;
        let q_flat = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &q_gate_heads,
            0,
            self.head_dim,
        )
        .map_err(|error| error.to_string())?;
        let gate_flat = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &q_gate_heads,
            self.head_dim,
            self.head_dim,
        )
        .map_err(|error| error.to_string())?;
        let gate_all = gate_flat
            .reshape(vec![batch, q_width])
            .map_err(|error| error.to_string())?;
        let q_all = backend
            .rms_norm(&q_flat, &self.q_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?
            .reshape(vec![batch, self.n_query_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let k_all = k_all
            .reshape(vec![batch * self.n_kv_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.k_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![batch, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let rope_positions: Vec<u32> = positions.iter().map(|value| *value as u32).collect();
        let q_all = apxinf_cuda::kernels::rope::apply_partial_positions(
            ctx,
            &q_all,
            self.n_query_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            &rope_positions,
        )
        .map_err(|error| error.to_string())?;
        let k_all = apxinf_cuda::kernels::rope::apply_partial_positions(
            ctx,
            &k_all,
            self.n_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            &rope_positions,
        )
        .map_err(|error| error.to_string())?;
        let q_all = q_all
            .reshape(vec![batch, q_width])
            .map_err(|error| error.to_string())?;
        let k_all = k_all
            .reshape(vec![batch, k_width])
            .map_err(|error| error.to_string())?;

        let mut attended_rows: Vec<Tensor> = Vec::with_capacity(batch);
        for (index, state) in states.iter_mut().enumerate() {
            let position = positions[index];
            let q = matrix_row(ctx, &q_all, index)?
                .reshape(vec![1, self.n_query_heads, self.head_dim])
                .map_err(|error| error.to_string())?;
            let k = matrix_row(ctx, &k_all, index)?
                .reshape(vec![1, self.n_kv_heads, self.head_dim])
                .map_err(|error| error.to_string())?;
            let v = matrix_row(ctx, &v_all, index)?
                .reshape(vec![1, self.n_kv_heads, self.head_dim])
                .map_err(|error| error.to_string())?;
            state
                .cache
                .append(ctx, 0, &k, &v, 1)
                .map_err(|error| error.to_string())?;
            let attended = backend
                .sdpa_decode(
                    &q,
                    &mut state.cache,
                    0,
                    self.n_query_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    position + 1,
                    state.max_seq_len,
                )
                .map_err(|error| error.to_string())?;
            attended_rows.push(attended);
        }
        let attended_refs: Vec<&Tensor> = attended_rows.iter().collect();
        let attended = apxinf_cuda::kernels::elementwise::stack_rows_bf16(ctx, &attended_refs)
            .map_err(|error| error.to_string())?;
        let gate = apxinf_cuda::kernels::activation::sigmoid(ctx, &gate_all)
            .map_err(|error| error.to_string())?;
        let gated = backend
            .mul(&attended, &gate)
            .map_err(|error| error.to_string())?;

        let attention_update = self
            .o_proj
            .project(ctx, &gated)
            .map_err(|error| error.to_string())?;
        let residual = backend
            .add(hidden, &attention_update)
            .map_err(|error| error.to_string())?;
        let mlp_input = backend
            .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let mlp_gate = self
            .gate_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_up = self
            .up_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_gate = backend.silu(&mlp_gate).map_err(|error| error.to_string())?;
        let mlp_hidden = backend
            .mul(&mlp_gate, &mlp_up)
            .map_err(|error| error.to_string())?;
        let mlp_update = self
            .down_proj
            .project(ctx, &mlp_hidden)
            .map_err(|error| error.to_string())?;
        let output = backend
            .add(&residual, &mlp_update)
            .map_err(|error| error.to_string())?;
        apxinf_cuda::kernels::qwen35_gdn::require_finite_bf16(
            ctx,
            &output,
            "full-attention batched decode output",
        )
        .map_err(|error| error.to_string())?;
        for state in states.iter_mut() {
            state.cache.advance(1);
        }
        Ok(output)
    }

    /// Execute a causal CUDA BF16 prefill over a contiguous sequence of rows.
    /// KV data is appended at `position` and the logical cache length advances
    /// only after attention, gating, projection, and MLP/residual output all
    /// succeed.
    pub fn prefill(
        &self,
        backend: &apxinf_cuda::CudaBackend,
        hidden: &Tensor,
        position: usize,
        state: &mut Qwen35CudaFullAttentionState,
    ) -> Result<Tensor, String> {
        let expected_device = Device::Cuda(self.device_id);
        let dims = hidden.shape().dims();
        if backend.device() != expected_device
            || state.device_id != self.device_id
            || hidden.device() != expected_device
            || hidden.dtype() != DType::BF16
            || dims.len() != 2
            || dims[1] != self.input_norm.shape().dims()[0]
        {
            return Err(format!(
                "full-attention prefill requires CUDA{} BF16 [rows,{}] hidden, matching state/backend; got {:?} {} on {:?}",
                self.device_id,
                self.input_norm.shape().dims()[0],
                dims,
                hidden.dtype(),
                hidden.device()
            ));
        }
        let rows = dims[0];
        if rows == 0 {
            return Err("full-attention prefill requires at least one row".into());
        }
        if position != state.seq_len() {
            return Err(format!(
                "full-attention position {position} does not match local KV length {}",
                state.seq_len()
            ));
        }
        if rows > state.max_seq_len.saturating_sub(position) {
            return Err(format!(
                "full-attention prefill of {rows} rows at position {position} exceeds max sequence length {}",
                state.max_seq_len
            ));
        }
        if position > u32::MAX as usize
            || position.checked_add(rows - 1).unwrap_or(usize::MAX) > u32::MAX as usize
        {
            return Err("full-attention positions exceed CUDA RoPE range".into());
        }
        if self.n_query_heads != 24
            || self.n_kv_heads != 4
            || self.head_dim != 256
            || self.rotary_dim != 64
            || self.q_layout().out_features != 2 * self.n_query_heads * self.head_dim
            || self.k_layout().out_features != self.n_kv_heads * self.head_dim
            || self.v_layout().out_features != self.n_kv_heads * self.head_dim
            || self.o_layout().out_features != self.input_norm.shape().dims()[0]
            || self.gate_layout().out_features != self.up_layout().out_features
            || self.down_layout().out_features != self.input_norm.shape().dims()[0]
        {
            return Err(
                "checkpoint full-attention dimensions violate the CUDA layer contract".into(),
            );
        }

        let ctx = backend.context();
        let normalized = backend
            .rms_norm(hidden, &self.input_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let q_gate = self
            .q_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let k = self
            .k_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;
        let v = self
            .v_proj
            .project(ctx, &normalized)
            .map_err(|error| error.to_string())?;

        let q_width = self.n_query_heads * self.head_dim;
        let q_gate_heads = q_gate
            .reshape(vec![rows * self.n_query_heads, self.head_dim * 2])
            .map_err(|error| error.to_string())?;
        let q = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &q_gate_heads,
            0,
            self.head_dim,
        )
        .map_err(|error| error.to_string())?
        .reshape(vec![rows, q_width])
        .map_err(|error| error.to_string())?;
        let gate = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx,
            &q_gate_heads,
            self.head_dim,
            self.head_dim,
        )
        .map_err(|error| error.to_string())?
        .reshape(vec![rows, q_width])
        .map_err(|error| error.to_string())?;
        let k = k
            .reshape(vec![rows, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let v = v
            .reshape(vec![rows, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;

        let q = q
            .reshape(vec![rows * self.n_query_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.q_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![rows, self.n_query_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let k = k
            .reshape(vec![rows * self.n_kv_heads, self.head_dim])
            .and_then(|value| backend.rms_norm(&value, &self.k_norm, self.rms_epsilon))
            .map_err(|error| error.to_string())?
            .reshape(vec![rows, self.n_kv_heads, self.head_dim])
            .map_err(|error| error.to_string())?;
        let q = apxinf_cuda::kernels::rope::apply_partial_batched(
            ctx,
            &q,
            self.n_query_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            position as u32,
        )
        .map_err(|error| error.to_string())?;
        let k = apxinf_cuda::kernels::rope::apply_partial_batched(
            ctx,
            &k,
            self.n_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            position as u32,
        )
        .map_err(|error| error.to_string())?;

        state
            .cache
            .append(ctx, 0, &k, &v, rows)
            .map_err(|error| error.to_string())?;
        let attended = backend
            .sdpa_prefill(
                &q,
                &mut state.cache,
                0,
                self.n_query_heads,
                self.n_kv_heads,
                self.head_dim,
                position + rows,
                state.max_seq_len,
            )
            .map_err(|error| error.to_string())?;
        let gate = apxinf_cuda::kernels::activation::sigmoid(ctx, &gate)
            .map_err(|error| error.to_string())?;
        let gated = backend
            .mul(&attended, &gate)
            .map_err(|error| error.to_string())?;
        let attention_update = self
            .o_proj
            .project(ctx, &gated)
            .map_err(|error| error.to_string())?;
        let residual = backend
            .add(hidden, &attention_update)
            .map_err(|error| error.to_string())?;
        let mlp_input = backend
            .rms_norm(&residual, &self.post_attention_norm, self.rms_epsilon)
            .map_err(|error| error.to_string())?;
        let mlp_gate = self
            .gate_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_up = self
            .up_proj
            .project(ctx, &mlp_input)
            .map_err(|error| error.to_string())?;
        let mlp_gate = backend.silu(&mlp_gate).map_err(|error| error.to_string())?;
        let mlp_hidden = backend
            .mul(&mlp_gate, &mlp_up)
            .map_err(|error| error.to_string())?;
        let mlp_update = self
            .down_proj
            .project(ctx, &mlp_hidden)
            .map_err(|error| error.to_string())?;
        let output = backend
            .add(&residual, &mlp_update)
            .map_err(|error| error.to_string())?;
        apxinf_cuda::kernels::qwen35_gdn::require_finite_bf16(
            ctx,
            &output,
            "full-attention prefill output",
        )
        .map_err(|error| error.to_string())?;
        state.cache.advance(rows);
        Ok(output)
    }
}

impl Qwen35CheckpointProjection {
    pub fn from_inventory(
        ctx: &CudaContext,
        inventory: &Qwen35CheckpointInventory,
        base: &str,
    ) -> Result<Self, String> {
        let host = inventory
            .read_packed_linear_payload(base)
            .map_err(|error| format!("load Qwen3.5 projection `{base}`: {error}"))?;
        Self::from_packed_payload(ctx, &host)
    }

    fn from_packed_payload(ctx: &CudaContext, host: &PackedLinearPayload) -> Result<Self, String> {
        let layout = Qwen35W4Layout::new(
            host.layout.out_features,
            host.layout.in_features,
            host.layout.group_size,
        )
        .map_err(|error| format!("create Qwen3.5 projection layout: {error}"))?;
        let packed_bytes = host
            .weight_packed
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let scales = Tensor::from_bf16(
            Shape::new(vec![layout.out_features, layout.groups()]),
            &host.scales_bf16,
        )
        .map_err(|error| format!("create Qwen3.5 BF16 scales: {error}"))?;
        let zero_point_bytes = host
            .zero_points
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let projection = Qwen35W4DeviceProjection::upload(
            ctx,
            layout,
            &packed_bytes,
            &scales,
            &zero_point_bytes,
        )
        .map_err(|error| format!("upload Qwen3.5 packed projection: {error}"))?;
        Ok(Self { layout, projection })
    }

    pub const fn layout(&self) -> Qwen35W4Layout {
        self.layout
    }

    pub fn project(&self, ctx: &CudaContext, activation: &Tensor) -> apxinf_core::Result<Tensor> {
        self.projection.project(ctx, activation)
    }

    pub fn project_host_f32(
        &self,
        ctx: &CudaContext,
        activation: &[f32],
    ) -> apxinf_core::Result<Vec<f32>> {
        if activation.len() != self.layout.in_features {
            return Err(apxinf_core::Error::Other(format!(
                "Qwen3.5 projection activation requires {}, got {}",
                self.layout.in_features,
                activation.len()
            )));
        }
        let rounded = activation
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let cpu = Tensor::from_bf16(Shape::new(vec![1, self.layout.in_features]), &rounded)?;
        let gpu = apxinf_cuda::transfers::to_cuda(&cpu, ctx.device_id())?;
        let output = self.project(ctx, &gpu)?;
        let output = apxinf_cuda::transfers::to_cpu(&output)?;
        if output.device() != Device::Cpu || output.dtype() != DType::BF16 {
            return Err(apxinf_core::Error::Other(
                "Qwen3.5 projection returned an invalid output tensor".into(),
            ));
        }
        let bytes = output.storage().as_cpu().ok_or_else(|| {
            apxinf_core::Error::Other("projection output is not CPU storage".into())
        })?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_loader::QWEN35_MODEL_REVISION;

    fn download_bf16(tensor: &Tensor) -> Vec<f32> {
        apxinf_cuda::transfers::to_cpu(tensor)
            .unwrap()
            .to_f32_vec()
            .unwrap()
    }

    fn bf16_payload(shape: Vec<usize>, value: f32) -> Bf16TensorPayload {
        let len = shape.iter().product();
        Bf16TensorPayload {
            shape,
            values: vec![bf16::from_f32(value); len],
        }
    }

    fn zero_packed(out_features: usize, in_features: usize) -> PackedLinearPayload {
        let layout = super::super::weights::PackedLinearLayout::new(out_features, in_features, 32);
        PackedLinearPayload {
            layout,
            weight_packed: vec![0; out_features * layout.packed_k_columns()],
            scales_bf16: vec![bf16::from_f32(1.0); out_features * layout.groups()],
            zero_points: vec![0; layout.packed_n_rows() * layout.groups()],
        }
    }

    fn tiny_gdn_payload(a_weight: f32) -> super::super::loader::GdnLayerPayload {
        super::super::loader::GdnLayerPayload {
            layer_index: 0,
            input_norm: bf16_payload(vec![2], 0.0),
            in_proj_qkv: zero_packed(6, 2),
            in_proj_z: zero_packed(2, 2),
            in_proj_a: bf16_payload(vec![1, 2], a_weight),
            in_proj_b: bf16_payload(vec![1, 2], 0.0),
            conv1d_weight: bf16_payload(vec![6, 1, 2], 1.0),
            a_log: bf16_payload(vec![1], 0.0),
            dt_bias: bf16_payload(vec![1], 0.0),
            norm: bf16_payload(vec![2], 1.0),
            out_proj: LinearPayload::Bf16(bf16_payload(vec![2, 2], 0.0)),
            post_attention_norm: bf16_payload(vec![2], 0.0),
            mlp_gate_proj: zero_packed(3, 2),
            mlp_up_proj: zero_packed(3, 2),
            mlp_down_proj: zero_packed(2, 3),
        }
    }

    /// Batched decode must be bit-identical to running the same requests
    /// serially: the shared projections are per-row independent kernels and
    /// the per-request segment reuses the serial code path, so any deviation
    /// is an orchestration bug. Verified on the real checkpoint layers with
    /// three requests over three steps, including divergent attention
    /// positions.
    #[test]
    #[ignore = "requires GPU and the pinned Qwen3.5 checkpoint"]
    fn real_layers_batched_decode_bit_matches_serial() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4")
            });
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let gdn = Qwen35CudaGdnLayer::from_inventory(&ctx, &inventory, 0).unwrap();
        let attention = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        let batch = 3usize;
        let steps = 3usize;
        // Deterministic pseudo-random step inputs, distinct per (step, row).
        let step_hidden = |step: usize| -> Vec<half::bf16> {
            (0..batch * 5120)
                .map(|index| {
                    let raw = (index * 2654435761 + step * 40503) % 1013;
                    half::bf16::from_f32(((raw as f32) - 506.0) * 0.0004)
                })
                .collect()
        };
        let upload_matrix = |values: &[half::bf16], rows: usize| -> Tensor {
            let host = Tensor::from_bf16(Shape::new(vec![rows, 5120]), values).unwrap();
            apxinf_cuda::transfers::to_cuda(&host, device).unwrap()
        };
        let bits = |tensor: &Tensor| -> Vec<u16> {
            apxinf_cuda::transfers::to_cpu(tensor)
                .unwrap()
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        };

        // ── GDN layer ────────────────────────────────────────────────
        let mut serial_states: Vec<Qwen35CudaGdnState> = (0..batch)
            .map(|_| Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap())
            .collect();
        let mut batch_states: Vec<Qwen35CudaGdnState> = (0..batch)
            .map(|_| Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap())
            .collect();
        for step in 0..steps {
            let values = step_hidden(step);
            let mut serial_outputs: Vec<u16> = Vec::new();
            for (row, state) in serial_states.iter_mut().enumerate() {
                let row_values = &values[row * 5120..(row + 1) * 5120];
                let hidden = upload_matrix(row_values, 1);
                let output = gdn.decode_token(&backend, &hidden, state).unwrap();
                serial_outputs.extend(bits(&output));
            }
            let hidden = upload_matrix(&values, batch);
            let mut refs: Vec<&mut Qwen35CudaGdnState> = batch_states.iter_mut().collect();
            let output = gdn
                .decode_token_batch(&backend, &hidden, &mut refs)
                .unwrap();
            assert_eq!(
                bits(&output),
                serial_outputs,
                "GDN batched step {step} diverged from serial"
            );
        }

        // ── Full-attention layer, including divergent positions ──────
        let capacity = 16usize;
        let mut serial_states: Vec<Qwen35CudaFullAttentionState> = (0..batch)
            .map(|_| Qwen35CudaFullAttentionState::new(&backend, capacity).unwrap())
            .collect();
        let mut batch_states: Vec<Qwen35CudaFullAttentionState> = (0..batch)
            .map(|_| Qwen35CudaFullAttentionState::new(&backend, capacity).unwrap())
            .collect();
        // Give request 2 a one-step head start on both sides so the batched
        // call sees genuinely different positions.
        {
            let values = step_hidden(99);
            let row_values = &values[2 * 5120..3 * 5120];
            let hidden = upload_matrix(row_values, 1);
            let serial_out = attention
                .decode_token(&backend, &hidden, 0, &mut serial_states[2])
                .unwrap();
            let batch_out = attention
                .decode_token(&backend, &hidden, 0, &mut batch_states[2])
                .unwrap();
            assert_eq!(bits(&serial_out), bits(&batch_out));
        }
        for step in 0..steps {
            let values = step_hidden(step);
            let mut serial_outputs: Vec<u16> = Vec::new();
            for (row, state) in serial_states.iter_mut().enumerate() {
                let row_values = &values[row * 5120..(row + 1) * 5120];
                let hidden = upload_matrix(row_values, 1);
                let position = state.seq_len();
                let output = attention
                    .decode_token(&backend, &hidden, position, state)
                    .unwrap();
                serial_outputs.extend(bits(&output));
            }
            let hidden = upload_matrix(&values, batch);
            let positions: Vec<usize> = batch_states.iter().map(|state| state.seq_len()).collect();
            assert_ne!(positions[0], positions[2], "positions must diverge");
            let mut refs: Vec<&mut Qwen35CudaFullAttentionState> =
                batch_states.iter_mut().collect();
            let output = attention
                .decode_token_batch(&backend, &hidden, &positions, &mut refs)
                .unwrap();
            assert_eq!(
                bits(&output),
                serial_outputs,
                "attention batched step {step} diverged from serial"
            );
        }
    }

    #[test]
    fn standard_rms_norm_upload_rounds_one_plus_checkpoint_weight() {
        let weights = [-0.5, 0.0, 0.25]
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let uploaded = standard_rms_norm_weights(&weights);
        assert_eq!(
            uploaded.into_iter().map(bf16::to_f32).collect::<Vec<_>>(),
            vec![0.5, 1.0, 1.25]
        );
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn mixed_bf16_projection_constructs_and_projects_checkpoint_matrix() {
        let ctx = CudaContext::new(0).expect("CUDA device required");
        let payload = Bf16TensorPayload {
            shape: vec![2, 3],
            values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                .into_iter()
                .map(bf16::from_f32)
                .collect(),
        };
        let projection =
            Qwen35MixedProjection::from_payload(&ctx, LinearPayload::Bf16(payload)).unwrap();
        assert_eq!(projection.in_features(), 3);
        assert_eq!(projection.out_features(), 2);

        let activation =
            Tensor::from_bf16(Shape::new(vec![1, 3]), &[bf16::from_f32(1.0); 3]).unwrap();
        let activation = apxinf_cuda::transfers::to_cuda(&activation, 0).unwrap();
        let output = projection.project(&ctx, &activation).unwrap();
        assert_eq!(download_bf16(&output), vec![6.0, 15.0]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn standard_rms_norm_upload_applies_one_plus_on_device() {
        let ctx = CudaContext::new(0).expect("CUDA device required");
        let checkpoint = [-0.5, 0.0, 0.25]
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let uploaded = upload_standard_rms_norm(&ctx, &checkpoint, 3, "synthetic norm").unwrap();
        assert_eq!(download_bf16(&uploaded), vec![0.5, 1.0, 1.25]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn synthetic_gdn_layer_runs_complete_zero_update_and_advances_state() {
        let backend = apxinf_cuda::CudaBackend::new(0).expect("CUDA device required");
        let layout = Qwen35GdnLayout::new(2, 1, 1, 2, 2).unwrap();
        let layer = Qwen35CudaGdnLayer::from_payload(
            backend.context(),
            0,
            2,
            1e-6,
            layout,
            tiny_gdn_payload(0.0),
        )
        .unwrap();
        let mut state = Qwen35CudaGdnState::new(&backend, layout).unwrap();
        let hidden = Tensor::from_bf16(
            Shape::new(vec![1, 2]),
            &[bf16::from_f32(3.0), bf16::from_f32(4.0)],
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&hidden, 0).unwrap();

        let output = layer.decode_token(&backend, &hidden, &mut state).unwrap();
        assert_eq!(download_bf16(&output), vec![3.0, 4.0]);
        assert_eq!(state.position(), 1);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn synthetic_gdn_layer_prefill_preserves_rows_and_advances_by_sequence_length() {
        let backend = apxinf_cuda::CudaBackend::new(0).expect("CUDA device required");
        let layout = Qwen35GdnLayout::new(2, 1, 1, 2, 2).unwrap();
        let layer = Qwen35CudaGdnLayer::from_payload(
            backend.context(),
            0,
            2,
            1e-6,
            layout,
            tiny_gdn_payload(0.0),
        )
        .unwrap();
        let mut state = Qwen35CudaGdnState::new(&backend, layout).unwrap();
        let hidden = Tensor::from_bf16(
            Shape::new(vec![2, 2]),
            &[
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
                bf16::from_f32(5.0),
                bf16::from_f32(12.0),
            ],
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&hidden, 0).unwrap();

        let output = layer.prefill(&backend, &hidden, &mut state).unwrap();

        assert_eq!(output.shape().dims(), &[2, 2]);
        assert_eq!(download_bf16(&output), vec![3.0, 4.0, 5.0, 12.0]);
        assert_eq!(state.position(), 2);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn synthetic_gdn_layer_prefill_rolls_back_state_after_recurrent_failure() {
        let backend = apxinf_cuda::CudaBackend::new(0).expect("CUDA device required");
        let layout = Qwen35GdnLayout::new(2, 1, 1, 2, 2).unwrap();
        let layer = Qwen35CudaGdnLayer::from_payload(
            backend.context(),
            0,
            2,
            1e-6,
            layout,
            tiny_gdn_payload(f32::NAN),
        )
        .unwrap();
        let mut state = Qwen35CudaGdnState::new(&backend, layout).unwrap();
        let hidden = Tensor::from_bf16(
            Shape::new(vec![2, 2]),
            &[
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
                bf16::from_f32(5.0),
                bf16::from_f32(12.0),
            ],
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&hidden, 0).unwrap();

        let error = layer.prefill(&backend, &hidden, &mut state).unwrap_err();

        assert!(error.contains("non-finite"));
        assert_eq!(state.position(), 0);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn synthetic_gdn_layer_resets_state_after_post_convolution_failure() {
        let backend = apxinf_cuda::CudaBackend::new(0).expect("CUDA device required");
        let layout = Qwen35GdnLayout::new(2, 1, 1, 2, 2).unwrap();
        let layer = Qwen35CudaGdnLayer::from_payload(
            backend.context(),
            0,
            2,
            1e-6,
            layout,
            tiny_gdn_payload(f32::NAN),
        )
        .unwrap();
        let mut state = Qwen35CudaGdnState::new(&backend, layout).unwrap();
        let hidden = Tensor::from_bf16(
            Shape::new(vec![1, 2]),
            &[bf16::from_f32(3.0), bf16::from_f32(4.0)],
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&hidden, 0).unwrap();

        let error = layer
            .decode_token(&backend, &hidden, &mut state)
            .unwrap_err();
        assert!(error.contains("non-finite"));
        assert_eq!(state.position(), 0);
    }

    /// Profiling harness for layer-level prefill/decode cost attribution on a
    /// development GPU. Not a correctness test: it loads one real GDN layer
    /// and one real full-attention layer, then loops prefill (rows from
    /// `APXINF_PROFILE_ROWS`, default 64) and decode so `nvprof`/`ncu` can
    /// attribute time per kernel. Wall-clock per call is printed for
    /// cross-checking against the measured per-block service cost.
    #[test]
    #[ignore = "profiling harness; requires a development GPU and the pinned checkpoint"]
    fn real_layer_profile_harness_prefill_and_decode() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let rows: usize = std::env::var("APXINF_PROFILE_ROWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);
        let iters: usize = std::env::var("APXINF_PROFILE_ITERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8);
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let gdn = Qwen35CudaGdnLayer::from_inventory(&ctx, &inventory, 0).unwrap();
        let attention = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        let values: Vec<half::bf16> = (0..rows * 5120)
            .map(|index| half::bf16::from_f32((((index % 61) as f32) - 30.0) * 0.001))
            .collect();
        let host = Tensor::from_bf16(Shape::new(vec![rows, 5120]), &values).unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
        let single = Tensor::from_bf16(Shape::new(vec![1, 5120]), &values[..5120]).unwrap();
        let single = apxinf_cuda::transfers::to_cuda(&single, device).unwrap();

        let time_section = |label: &str, mut body: Box<dyn FnMut() + '_>| {
            // Warm up every shape/tactic (cuBLAS heuristics, allocator pools)
            // before the timed loop so the first section is not charged for
            // one-time initialization.
            body();
            ctx.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..iters {
                body();
            }
            ctx.synchronize().unwrap();
            let elapsed = started.elapsed();
            eprintln!(
                "profile {label}: rows={rows} iters={iters} total={elapsed:?} per_call={:?}",
                elapsed / iters as u32
            );
        };

        time_section(
            "gdn_prefill",
            Box::new(|| {
                let mut state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
                gdn.prefill(&backend, &hidden, &mut state).unwrap();
            }),
        );
        time_section(
            "attention_prefill",
            Box::new(|| {
                let mut state = Qwen35CudaFullAttentionState::new(&backend, rows + 2).unwrap();
                attention.prefill(&backend, &hidden, 0, &mut state).unwrap();
            }),
        );
        // Stage-level attribution inside the GDN prefill, replaying the same
        // call sequence the layer makes so each stage can be timed alone.
        let ctx_ref = backend.context();
        let normalized = backend
            .rms_norm(&hidden, &gdn.input_norm, gdn.rms_epsilon)
            .unwrap();
        time_section(
            "gdn_stage_projections_in",
            Box::new(|| {
                gdn.in_proj_qkv.project(ctx_ref, &normalized).unwrap();
                gdn.in_proj_z.project(ctx_ref, &normalized).unwrap();
                gdn.in_proj_a.project(ctx_ref, &normalized).unwrap();
                gdn.in_proj_b.project(ctx_ref, &normalized).unwrap();
            }),
        );
        let qkv = gdn.in_proj_qkv.project(ctx_ref, &normalized).unwrap();
        let a = gdn.in_proj_a.project(ctx_ref, &normalized).unwrap();
        let b = gdn.in_proj_b.project(ctx_ref, &normalized).unwrap();
        time_section(
            "gdn_stage_conv_prefill",
            Box::new(|| {
                let mut state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
                state
                    .state
                    .causal_conv_silu_prefill(ctx_ref, &qkv, &gdn.conv_weight)
                    .unwrap();
            }),
        );
        let mut conv_state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
        let convolved = conv_state
            .state
            .causal_conv_silu_prefill(ctx_ref, &qkv, &gdn.conv_weight)
            .unwrap();
        let query_width = gdn.dimensions().query_width();
        let value_width = gdn.dimensions().value_width();
        let query = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx_ref,
            &convolved,
            0,
            query_width,
        )
        .unwrap();
        let key = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx_ref,
            &convolved,
            query_width,
            query_width,
        )
        .unwrap();
        let value = apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            ctx_ref,
            &convolved,
            query_width * 2,
            value_width,
        )
        .unwrap();
        time_section(
            "gdn_stage_delta_prefill",
            Box::new(|| {
                let mut state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
                state
                    .state
                    .gated_delta_prefill(
                        ctx_ref,
                        &query,
                        &key,
                        &value,
                        &a,
                        &b,
                        &gdn.a_log,
                        &gdn.dt_bias,
                    )
                    .unwrap();
            }),
        );
        let z = gdn.in_proj_z.project(ctx_ref, &normalized).unwrap();
        let mut delta_state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
        let recurrent = delta_state
            .state
            .gated_delta_prefill(
                ctx_ref,
                &query,
                &key,
                &value,
                &a,
                &b,
                &gdn.a_log,
                &gdn.dt_bias,
            )
            .unwrap();
        time_section(
            "gdn_stage_norm_out_mlp",
            Box::new(|| {
                let gated = gated_rms_norm_bf16(
                    ctx_ref,
                    &recurrent,
                    &z,
                    &gdn.norm,
                    gdn.dimensions().value_heads,
                    gdn.dimensions().value_dim,
                    gdn.rms_epsilon,
                )
                .unwrap();
                let update = gdn.out_proj.project(ctx_ref, &gated).unwrap();
                let residual = backend.add(&hidden, &update).unwrap();
                let mlp_input = backend
                    .rms_norm(&residual, &gdn.post_attention_norm, gdn.rms_epsilon)
                    .unwrap();
                let mlp_gate = gdn.mlp_gate_proj.project(ctx_ref, &mlp_input).unwrap();
                let mlp_up = gdn.mlp_up_proj.project(ctx_ref, &mlp_input).unwrap();
                let mlp_hidden = backend
                    .mul(&backend.silu(&mlp_gate).unwrap(), &mlp_up)
                    .unwrap();
                gdn.mlp_down_proj.project(ctx_ref, &mlp_hidden).unwrap();
            }),
        );

        // Micro-benchmarks isolating the allocation cost hypothesis: every
        // W4 projection currently cudaMallocs a fresh dequant scratch (up to
        // 178 MB for the MLP shapes) plus output and flags buffers per call.
        time_section(
            "micro_cuda_malloc_free_178mb",
            Box::new(|| {
                let buffer = apxinf_cuda::CudaBuffer::alloc(178 * 1024 * 1024, device).unwrap();
                drop(buffer);
            }),
        );
        time_section(
            "micro_cuda_malloc_free_4b",
            Box::new(|| {
                let buffer = apxinf_cuda::CudaBuffer::alloc(4, device).unwrap();
                drop(buffer);
            }),
        );
        time_section(
            "micro_mlp_gate_project_only",
            Box::new(|| {
                gdn.mlp_gate_proj.project(ctx_ref, &normalized).unwrap();
            }),
        );

        time_section(
            "gdn_decode",
            Box::new(|| {
                let mut state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
                gdn.decode_token(&backend, &single, &mut state).unwrap();
            }),
        );
        time_section(
            "attention_decode",
            Box::new(|| {
                let mut state = Qwen35CudaFullAttentionState::new(&backend, rows + 2).unwrap();
                attention.prefill(&backend, &hidden, 0, &mut state).unwrap();
                attention
                    .decode_token(&backend, &single, rows, &mut state)
                    .unwrap();
            }),
        );

        // Service-shaped decode: the request state is created once and reused
        // across steps, exactly like `Qwen35CudaSession`, so state-allocation
        // cost is excluded and per-step cost is directly comparable to the
        // measured service TPOT.
        let mut gdn_state = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
        time_section(
            "gdn_decode_reused_state",
            Box::new(|| {
                gdn.decode_token(&backend, &single, &mut gdn_state).unwrap();
            }),
        );
        let capacity = rows + (iters + 1) * 4 + 8;
        let mut attention_state = Qwen35CudaFullAttentionState::new(&backend, capacity).unwrap();
        attention
            .prefill(&backend, &hidden, 0, &mut attention_state)
            .unwrap();
        let mut attention_position = rows;
        time_section(
            "attn_decode_reused_state",
            Box::new(|| {
                attention
                    .decode_token(&backend, &single, attention_position, &mut attention_state)
                    .unwrap();
                attention_position += 1;
            }),
        );

        // Attention prefill stage attribution: the score GEMMs, the softmax
        // and the value GEMMs are timed separately at a long kv length so the
        // softmax kernel's per-block full-row rescan is visible.
        {
            let kv_len = std::env::var("APXINF_PROFILE_KV")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4096);
            let heads = 24usize;
            let scores_rows = rows * heads;
            let values: Vec<half::bf16> = (0..scores_rows * kv_len)
                .map(|index| half::bf16::from_f32((((index % 97) as f32) - 48.0) * 0.01))
                .collect();
            let host = Tensor::from_bf16(Shape::new(vec![scores_rows, kv_len]), &values).unwrap();
            let scores = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
            time_section(
                "attn_stage_softmax_only",
                Box::new(|| {
                    apxinf_cuda::kernels::attention::softmax_causal(
                        ctx_ref,
                        &scores,
                        0,
                        heads as u32,
                    )
                    .unwrap();
                }),
            );
            eprintln!(
                "profile attn_softmax_shape: rows={scores_rows} cols={kv_len} \
                 elements={} redundant_block_rescans={}",
                scores_rows * kv_len,
                (kv_len + 255) / 256
            );
        }

        // Attention prefill at a REALISTIC kv length. The plain
        // `attention_prefill` section above starts from an empty cache, so its
        // kv_len equals `rows`; in a real 16K prompt the later blocks attend
        // over a kv cache of up to 16512, and the score/value GEMMs scale with
        // rows x kv_len. This section pre-fills the cache and then times one
        // additional block, which is what the service actually pays.
        {
            let preload = std::env::var("APXINF_PROFILE_PRELOAD")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            if preload > 0 {
                // Time the whole staircase once: every block from 0 to
                // `preload`, exactly as the service walks a long prompt. The
                // per-block average then reflects the real growing kv length
                // instead of a fixed short one. A single pass is used because
                // the cache cursor only moves forward.
                let mut state =
                    Qwen35CudaFullAttentionState::new(&backend, preload + rows + 8).unwrap();
                ctx.synchronize().unwrap();
                let started = std::time::Instant::now();
                let mut position = 0usize;
                let mut blocks = 0usize;
                while position + rows <= preload {
                    attention
                        .prefill(&backend, &hidden, position, &mut state)
                        .unwrap();
                    position += rows;
                    blocks += 1;
                }
                ctx.synchronize().unwrap();
                let elapsed = started.elapsed();
                eprintln!(
                    "profile attn_prefill_staircase: preload={preload} block_rows={rows} \
                     blocks={blocks} final_kv={position} total={elapsed:?} \
                     per_block_avg={:?}",
                    elapsed / blocks.max(1) as u32
                );

                // Stage split for ONE block at the final (longest) kv length,
                // which is where the staircase spends most of its time.
                let heads = 24usize;
                let kv_len = position;
                let q_values: Vec<half::bf16> = (0..rows * heads * 256)
                    .map(|i| half::bf16::from_f32((((i % 71) as f32) - 35.0) * 0.01))
                    .collect();
                let q_host =
                    Tensor::from_bf16(Shape::new(vec![rows, heads, 256]), &q_values).unwrap();
                let q_gpu = apxinf_cuda::transfers::to_cuda(&q_host, device).unwrap();
                time_section(
                    "attn_stage_full_sdpa_at_final_kv",
                    Box::new(|| {
                        backend
                            .sdpa_prefill(
                                &q_gpu,
                                &mut state.cache,
                                0,
                                heads,
                                4,
                                256,
                                kv_len,
                                position + rows + 8,
                            )
                            .unwrap();
                    }),
                );
                let scores_values: Vec<half::bf16> = (0..rows * heads * kv_len)
                    .map(|i| half::bf16::from_f32((((i % 53) as f32) - 26.0) * 0.02))
                    .collect();
                let scores_host =
                    Tensor::from_bf16(Shape::new(vec![rows * heads, kv_len]), &scores_values)
                        .unwrap();
                let scores_gpu = apxinf_cuda::transfers::to_cuda(&scores_host, device).unwrap();
                time_section(
                    "attn_stage_scale_only",
                    Box::new(|| {
                        apxinf_cuda::kernels::elementwise::scale(ctx_ref, &scores_gpu, 0.0625)
                            .unwrap();
                    }),
                );
                time_section(
                    "attn_stage_softmax_at_final_kv",
                    Box::new(|| {
                        apxinf_cuda::kernels::attention::softmax_causal(
                            ctx_ref,
                            &scores_gpu,
                            0,
                            heads as u32,
                        )
                        .unwrap();
                    }),
                );
                let scores_bytes = rows * heads * kv_len * 2;
                time_section(
                    "attn_stage_alloc_scores_buffer",
                    Box::new(|| {
                        let buffer = apxinf_cuda::CudaBuffer::alloc(scores_bytes, device).unwrap();
                        drop(buffer);
                    }),
                );
                eprintln!(
                    "profile attn_stage_shapes: rows={rows} heads={heads} kv={kv_len} \
                     scores_buffer={:.0} MB",
                    scores_bytes as f64 / (1024.0 * 1024.0)
                );

                // Isolate the output-stride hypothesis: the same score GEMM
                // written to an interleaved [seq, head, kv] layout
                // (ldc = heads*kv, one row every 768 KB) versus a contiguous
                // per-head [head, seq, kv] layout (ldc = kv).
                {
                    use apxinf_cuda::cublas::CublasTranspose;
                    let q_buf = apxinf_cuda::CudaBuffer::from_tensor(&q_gpu).unwrap();
                    let k_buf = apxinf_cuda::CudaBuffer::alloc(kv_len * 256 * 2, device).unwrap();
                    let c_buf =
                        apxinf_cuda::CudaBuffer::alloc(rows * heads * kv_len * 2, device).unwrap();
                    let gqa = heads / 4;
                    let run = |ldc: i32, stride_c: i64, label: &str| {
                        time_section(
                            label,
                            Box::new(|| {
                                ctx_ref
                                    .cublas()
                                    .batched_gemm_ex(
                                        DType::BF16,
                                        CublasTranspose::None,
                                        CublasTranspose::Transpose,
                                        rows,
                                        kv_len,
                                        256,
                                        1.0,
                                        &q_buf,
                                        (heads * 256) as i32,
                                        256,
                                        &k_buf,
                                        256,
                                        0,
                                        0.0,
                                        &c_buf,
                                        ldc,
                                        stride_c,
                                        gqa as i32,
                                    )
                                    .unwrap();
                            }),
                        );
                    };
                    run(
                        (heads * kv_len) as i32,
                        kv_len as i64,
                        "attn_micro_gemm_interleaved_ldc",
                    );
                    run(
                        kv_len as i32,
                        (rows * kv_len) as i64,
                        "attn_micro_gemm_contiguous_ldc",
                    );
                }
            }
        }

        // Decode-shape stage attribution inside one GDN layer.
        let normalized_1 = backend
            .rms_norm(&single, &gdn.input_norm, gdn.rms_epsilon)
            .unwrap();
        time_section(
            "dec_stage_in_projections",
            Box::new(|| {
                gdn.in_proj_qkv.project(ctx_ref, &normalized_1).unwrap();
                gdn.in_proj_z.project(ctx_ref, &normalized_1).unwrap();
                gdn.in_proj_a.project(ctx_ref, &normalized_1).unwrap();
                gdn.in_proj_b.project(ctx_ref, &normalized_1).unwrap();
            }),
        );
        time_section(
            "dec_stage_mlp_projections",
            Box::new(|| {
                let g = gdn.mlp_gate_proj.project(ctx_ref, &normalized_1).unwrap();
                let u = gdn.mlp_up_proj.project(ctx_ref, &normalized_1).unwrap();
                let h = backend.mul(&backend.silu(&g).unwrap(), &u).unwrap();
                gdn.mlp_down_proj.project(ctx_ref, &h).unwrap();
            }),
        );
        let mlp_hidden_1 = {
            let g = gdn.mlp_gate_proj.project(ctx_ref, &normalized_1).unwrap();
            let u = gdn.mlp_up_proj.project(ctx_ref, &normalized_1).unwrap();
            backend.mul(&backend.silu(&g).unwrap(), &u).unwrap()
        };
        time_section(
            "dec_stage_mlp_down_only",
            Box::new(|| {
                gdn.mlp_down_proj.project(ctx_ref, &mlp_hidden_1).unwrap();
            }),
        );
        time_section(
            "dec_stage_mlp_gate_only",
            Box::new(|| {
                gdn.mlp_gate_proj.project(ctx_ref, &normalized_1).unwrap();
            }),
        );
        let mut conv_state_1 = Qwen35CudaGdnState::new(&backend, gdn.dimensions()).unwrap();
        let qkv_1 = gdn.in_proj_qkv.project(ctx_ref, &normalized_1).unwrap();
        time_section(
            "dec_stage_conv_step",
            Box::new(|| {
                conv_state_1
                    .state
                    .causal_conv_silu(ctx_ref, &qkv_1, &gdn.conv_weight)
                    .unwrap();
            }),
        );

        // LM head projection at decode shape [1, 5120] -> [1, 248320]: BF16
        // GEMV over ~2.4 GiB of weights, plus the D2H logits copy and host
        // argmax the session performs per token.
        let lm_payload = inventory
            .read_bf16_tensor_payload(
                "lm_head.weight",
                &[inventory.config.vocab_size, inventory.config.hidden_size],
            )
            .unwrap();
        let lm_head = Qwen35Bf16Projection::from_payload(ctx_ref, &lm_payload).unwrap();
        time_section(
            "lm_head_project_only",
            Box::new(|| {
                lm_head.project(ctx_ref, &single).unwrap();
            }),
        );
        time_section(
            "lm_head_project_d2h_argmax",
            Box::new(|| {
                let logits = lm_head.project(ctx_ref, &single).unwrap();
                let cpu = apxinf_cuda::transfers::to_cpu(&logits).unwrap();
                let values = cpu.to_f32_vec().unwrap();
                let _ = crate::qwen35::model::greedy_argmax(&values, values.len()).unwrap();
            }),
        );
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_owns_all_checkpoint_weights_on_cuda() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();

        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        assert_eq!(layer.layer_index(), 3);
        assert_eq!(layer.device_id(), device);
        assert_eq!(layer.q_layout().in_features, 5120);
        assert_eq!(layer.q_layout().out_features, 12288);
        assert_eq!(layer.k_layout().out_features, 1024);
        assert_eq!(layer.v_layout().out_features, 1024);
        assert_eq!(layer.o_layout().out_features, 5120);
        assert_eq!(layer.gate_layout().out_features, 17408);
        assert_eq!(layer.up_layout().out_features, 17408);
        assert_eq!(layer.down_layout().out_features, 5120);
        assert_eq!(layer.norm_shapes(), [[5120], [256], [256], [5120]]);
        assert_eq!(layer.rope_theta, 10_000_000.0);
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_projects_cuda_qkv_from_oracle_embedding() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();
        let oracle = std::path::PathBuf::from(
            "/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts/embedding.f32.bin",
        );
        let bytes = std::fs::read(oracle).unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let host = Tensor::from_bf16(
            Shape::new(vec![8, 5120]),
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
        let projected = layer.project_attention_inputs(&backend, &hidden).unwrap();
        assert_eq!(projected.q_gate.shape().dims(), &[8, 12288]);
        assert_eq!(projected.k.shape().dims(), &[8, 1024]);
        assert_eq!(projected.v.shape().dims(), &[8, 1024]);
        assert_eq!(projected.q_gate.device(), Device::Cuda(device));
        assert!(apxinf_cuda::transfers::to_cpu(&projected.q_gate).is_ok());
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_runs_one_cuda_token_step_and_advances_local_kv() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        let oracle = std::path::PathBuf::from(
            "/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts/embedding.f32.bin",
        );
        let bytes = std::fs::read(oracle).unwrap();
        let values = bytes
            .chunks_exact(4)
            .take(5120)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        let host = Tensor::from_bf16(Shape::new(vec![1, 5120]), &values).unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
        let mut state = Qwen35CudaFullAttentionState::new(&backend, 8).unwrap();

        let output = layer
            .decode_token(&backend, &hidden, 0, &mut state)
            .unwrap();

        assert_eq!(output.shape().dims(), &[1, 5120]);
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.device(), Device::Cuda(device));
        let output_cpu = apxinf_cuda::transfers::to_cpu(&output).unwrap();
        let output_values = output_cpu.to_f32_vec().unwrap();
        assert!(output_values.iter().all(|value| value.is_finite()));
        assert!(output_values.iter().any(|value| value.abs() > 1e-6));
        assert_eq!(state.cache.storage_dtype(), DType::BF16);
        assert_eq!(state.cache.dtype().unwrap(), Some(DType::BF16));
        assert_eq!(state.seq_len(), 1);
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_full_attention_layer_three_prefills_multiple_cuda_rows_and_advances_by_rows() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let layer = Qwen35CudaFullAttentionLayer::from_inventory(&ctx, &inventory, 3).unwrap();

        let oracle = std::path::PathBuf::from(
            "/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts/embedding.f32.bin",
        );
        let bytes = std::fs::read(oracle).unwrap();
        let values = bytes
            .chunks_exact(4)
            .take(2 * 5120)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        let host = Tensor::from_bf16(Shape::new(vec![2, 5120]), &values).unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&host, device).unwrap();
        let mut state = Qwen35CudaFullAttentionState::new(&backend, 8).unwrap();

        let output = layer.prefill(&backend, &hidden, 0, &mut state).unwrap();

        assert_eq!(output.shape().dims(), &[2, 5120]);
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.device(), Device::Cuda(device));
        let output_cpu = apxinf_cuda::transfers::to_cpu(&output).unwrap();
        assert!(output_cpu
            .to_f32_vec()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(state.seq_len(), 2);
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint payload"]
    fn real_layer_zero_projection_matches_cpu_for_selected_outputs() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let ctx = CudaContext::new(device).expect("CUDA device required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let projection =
            Qwen35CheckpointProjection::from_inventory(&ctx, &inventory, base).unwrap();
        let host = inventory.read_packed_linear(base).unwrap();
        let activation = (0..host.layout.in_features)
            .map(|index| (index as f32 * 0.00031).sin() * 0.02)
            .collect::<Vec<_>>();
        let output = projection.project_host_f32(&ctx, &activation).unwrap();
        assert_eq!(output.len(), host.layout.out_features);
        for out in 0..4 {
            let expected = (0..host.layout.in_features)
                .map(|k| {
                    let value = bf16::from_f32(activation[k]).to_f32();
                    let weight = host
                        .layout
                        .dequantize_value(
                            &host.weight_packed,
                            &host.scales,
                            &host.zero_points,
                            out,
                            k,
                        )
                        .unwrap();
                    value * weight
                })
                .sum::<f32>();
            let expected = bf16::from_f32(expected).to_f32();
            let delta = (output[out] - expected).abs();
            assert!(
                delta <= 0.02 * expected.abs().max(1.0),
                "out={out} delta={delta}"
            );
        }
    }

    #[test]
    #[ignore = "requires GPU2 and the pinned Qwen3.5 checkpoint/oracle payload"]
    fn real_layer_zero_prefill_first_row_matches_oracle_hidden() {
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .expect("APXINF_CUDA_DEVICE must select a non-formal development GPU")
            .parse::<usize>()
            .unwrap();
        let backend = apxinf_cuda::CudaBackend::new(device).expect("CUDA backend required");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                .unwrap();
        let layer = Qwen35CudaGdnLayer::from_inventory(backend.context(), &inventory, 0).unwrap();
        let oracle = std::path::PathBuf::from(
            "/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts",
        );
        let read_row = |name: &str, row: usize| {
            std::fs::read(oracle.join(name))
                .unwrap()
                .chunks_exact(4)
                .skip(row * inventory.config.hidden_size)
                .take(inventory.config.hidden_size)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let embedding = std::fs::read(oracle.join("embedding.f32.bin"))
            .unwrap()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let hidden = Tensor::from_bf16(
            Shape::new(vec![8, inventory.config.hidden_size]),
            &embedding
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let hidden = apxinf_cuda::transfers::to_cuda(&hidden, device).unwrap();
        let mut state = Qwen35CudaGdnState::new(&backend, layer.dimensions()).unwrap();

        let actual = layer.prefill(&backend, &hidden, &mut state).unwrap();
        let actual = apxinf_cuda::transfers::to_cpu(&actual)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let expected = read_row("layer-000-hidden.f32.bin", 0);
        let actual = &actual[..inventory.config.hidden_size];
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut sum_squared = 0.0f64;
        let mut exact = 0usize;
        let mut first_mismatch = None;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let delta = (actual - expected).abs();
            max_abs = max_abs.max(delta);
            sum_abs += f64::from(delta);
            sum_squared += f64::from(delta) * f64::from(delta);
            if actual == expected {
                exact += 1;
            } else if first_mismatch.is_none() {
                first_mismatch = Some((index, actual, expected, delta));
            }
        }
        let len = actual.len();
        let mae = sum_abs / len as f64;
        let rmse = (sum_squared / len as f64).sqrt();
        // The approved oracle schema allows absolute/relative error of 0.01.
        // CUDA reduction order (especially packed-W4 GEMM tree reductions) is
        // not required to reproduce Transformers' exact BF16 bit pattern.
        let violations = actual
            .iter()
            .zip(&expected)
            .filter(|(actual, expected)| {
                (**actual - **expected).abs() > 0.01 + 0.01 * expected.abs()
            })
            .count();
        assert!(
            violations == 0,
            "layer 0 token 0 mismatch: max_abs={max_abs} mae={mae} rmse={rmse} \
             exact={exact}/{len} violations={violations} first_mismatch={first_mismatch:?}"
        );
    }
}
