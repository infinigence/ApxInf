# M1-R1 Layer-3 CUDA Evidence

Date: 2026-08-24
Branch: `integrate/member2`
Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`
Checkpoint: `/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4`
Development GPU: `GPU-f4efcc89-d74e-d37b-caf1-52cde9f0582e` (GPU2 logical lane)

The new `Qwen35CudaFullAttentionLayer` owns all seven layer-3 packed W4
projections and four native BF16 norm vectors on the selected CUDA device.
The projection smoke consumes the existing oracle `embedding.f32.bin` and
executes BF16 RMSNorm followed by q/gate, k, and v packed projections. No
full-BF16 checkpoint copy, CPU fallback, or token output is involved.

Commands:

```text
CUDA_VISIBLE_DEVICES=GPU-f4efcc89-d74e-d37b-caf1-52cde9f0582e \
APXINF_CUDA_DEVICE=0 \
APXINF_QWEN35_CHECKPOINT=/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
cargo test -p apxinf-model --features cuda \
  qwen35::cuda::tests::real_full_attention_layer_three_owns_all_checkpoint_weights_on_cuda \
  --lib -- --ignored --nocapture

CUDA_VISIBLE_DEVICES=GPU-f4efcc89-d74e-d37b-caf1-52cde9f0582e \
APXINF_CUDA_DEVICE=0 \
APXINF_QWEN35_CHECKPOINT=/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
cargo test -p apxinf-model --features cuda \
  qwen35::cuda::tests::real_full_attention_layer_three_projects_cuda_qkv_from_oracle_embedding \
  --lib -- --ignored --nocapture
```

Both tests passed on 2026-08-24. The second test verified device shapes
`[8,12288]`, `[8,1024]`, `[8,1024]` and successful device-to-host readback.
This is layer/projection evidence only; it does not establish a complete
64-layer executor, `RUNTIME_READY`, `BASE_CORRECT`, or production serve.

The oracle input used by the projection smoke is covered by the existing
oracle `manifest.sha256`; the evaluator directory was not modified.
