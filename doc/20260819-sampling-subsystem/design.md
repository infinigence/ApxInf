# Sampling Subsystem Design

Date: 2026-08-19
Status: Implemented

## Goals

- Keep logits, filtering, and selection on the model backend.
- Give LLMs and VLMs one categorical sampling contract.
- Give continuous VLA models a reusable device normal generator without
  pretending their regression output is categorical.
- Make seeded results reproducible across CPU and CUDA.
- Keep model implementations responsible only for producing logits or action
  tensors; policy and RNG state belong outside model architecture code.
- Preserve the historical greedy generation API as a compatibility wrapper.

## Non-goals

- Batched or paged request scheduling.
- Beam search, speculative decoding, or grammar-constrained decoding.
- A universal VLA `ActionHeadKind` hierarchy. Current PI0.5 inference needs a
  standard-normal initial latent, not categorical, deterministic-regression,
  or Gaussian-regression dispatch.
- Image preprocessing or LLM/VLM modality dispatch. Those are separate model
  frontend concerns.

## Architecture

```text
Autoregressive LLM/VLM

GenerationRequest
    │
    ├─ LlmTrait::prefill / forward
    │       └─ device-resident [..., vocab] logits
    │
    └─ Backend::create_token_sampler
            ├─ prompt/generated-token history
            ├─ penalties and filtering
            ├─ RngKey stream
            └─ TokenSample { token_id, logprob }

Continuous VLA

VlaRequest
    ├─ InitialLatent::Provided(&Tensor) ── exact compatibility path
    └─ InitialLatent::Generate { rng }
            └─ Backend::create_normal_generator
                    └─ fill stable model-owned latent tensor in place
```

The categorical and continuous paths share only backend ownership and the
counter-based `RngKey`. `TokenSampler` is not called by PI0.5.

## Core categorical API

The public types live in `apxinf-core/src/sampling.rs` and are re-exported by
`apxinf-core`.

### Policy types

`TokenSelection` chooses between:

- `Greedy`; and
- `Random { temperature, top_k, top_p }`.

`TokenPenalties` contains:

- `repetition`, where `1.0` disables the penalty;
- `frequency`, multiplied by the occurrence count; and
- `presence`, applied once to every seen token.

`TokenSamplingParams` combines selection, penalties, and
`return_logprob`. `TokenSamplingParams::greedy()` is the compatibility default.

### Stream identity

```rust
pub struct RngKey {
    pub seed: u64,
    pub sequence: u64,
    pub draw: u64,
}
```

The output is a pure function of the key and element index. `sequence`
separates logical requests, while `draw` identifies a generation step. The
token sampler advances `draw` only after a successful sample. The normal
generator accepts an explicit key on each call, leaving advancement policy to
its caller.

### Allocation and request lifecycle

`TokenSamplingSpec { vocab_size, max_sequence_len }` is the fixed allocation
contract. A backend creates one stateful `TokenSampler` for a request. The
generation driver then calls:

1. `begin(TokenSamplingInit)` once with prompt IDs, policy, and initial RNG;
2. `sample(NextTokenLogits)` once per generated token.

The sampler owns occurrence counts, sequence length, RNG state, output storage,
and backend-specific workspaces. Models never own or mutate sampling history.

`NextTokenLogits` is a checked view selecting an explicit row or the final row
of a tensor whose last dimension is the vocabulary. This prevents every model
from reimplementing last-row offset logic.

## Sampling semantics

For every token, operations occur in this order:

1. select the requested logits row;
2. convert arithmetic to f32;
3. map NaN to negative infinity and positive infinity to finite `f32::MAX`;
4. apply repetition penalty to seen tokens: negative logits are multiplied,
   non-negative logits are divided;
5. subtract `frequency * occurrence_count` and then `presence`;
6. for random selection, divide by temperature;
7. sort descending with token ID as the deterministic tie-break;
8. apply top-k, then retain the smallest top-p prefix whose cumulative mass
   reaches the threshold;
9. sample from that retained distribution and optionally return its
   post-filter log-probability; and
10. add the selected token to history and advance the draw counter.

Greedy selection chooses the lowest token ID on equal adjusted logits. When a
greedy log-probability is requested, it is calculated over the complete
post-penalty vocabulary distribution.

Invalid policy parameters, tensor/device mismatches, out-of-vocabulary prompt
IDs, exhausted sequence capacity, invalid probability mass, and all-invalid
logits return errors rather than silently selecting a token.

## LLM and VLM integration

`GenerationOptions` is the public partial-settings layer: every field is
optional so model defaults, deployment overrides, and request overrides can use
the same type. `ResolvedGenerationOptions` is crate-private and carries the
complete maximum output length, EOS IDs, sampling policy, and `RngKey` consumed
by the generation driver. Resolution applies layers in this order:

```text
ApxInf defaults < generation_config.json < deployment overrides < request
```

`GenerationRequest` pairs public options with the existing `LlmInput`. The
shared driver resolves them and then:

1. validates modality and sampling parameters before model work;
2. creates and initializes the sampler from `LlmTrait::backend()`;
3. resets and prewarms the model;
4. samples the final prefill row; and
5. repeats token-only `forward` plus sampling until length or EOS termination.

`GeneratedToken` contains the token ID and optional log-probability.
`GenerationOutput` contains all generated tokens and the existing timing
profile. `generate_streaming` remains available and constructs greedy options;
new callers use `generate_streaming_with_options`.

Example:

```rust
let options = GenerationOptions {
    max_new_tokens: Some(50),
    eos_token_ids: Some(vec![eos_token_id]),
    sampling_mode: Some(SamplingMode::Random),
    temperature: Some(0.8),
    top_k: Some(40),
    top_p: Some(0.95),
    repetition_penalty: Some(1.1),
    seed: Some(42),
    return_logprob: Some(true),
    ..GenerationOptions::default()
};

let output = model.generate_streaming_with_options(input, &options, on_token)?;
```

## VLA integration

`Observation` describes environment data only: vision input and token IDs.
Generation-specific state is carried by:

```rust
pub enum InitialLatent<'a> {
    Generate { rng: RngKey },
    Provided(&'a Tensor),
}

pub struct VlaRequest<'a> {
    pub observation: &'a Observation,
    pub initial_latent: InitialLatent<'a>,
}
```

`Provided` preserves exact reference and OpenPI-compatible behavior.
`Generate` asks the prepared runtime's `NormalGenerator` to overwrite the same
stable device tensor captured by the graph. This removes the host allocation
and H2D latent copy without changing PI0.5's action-head mathematics.

The Python binding exposes one compatibility-preserving entry point:
`infer_rgb(..., noise=None)`. A provided array is copied exactly; omission draws
from a model-owned counter-based stream and fills the stable device buffer. The
stream advances once per implicit inference and can be reset with
`reset_sampling(seed)`. `infer_rgb_seeded(..., seed, sequence=0, draw=0)` remains
available for an explicitly keyed replay. The private L0 test binding follows
the same optional-noise rule and retains `_infer_patches_seeded`.

At L2, `Pi05Policy.infer(observation, noise=...)` accepts an exact external
latent. The keyword takes precedence over `observation["noise"]` and a custom
pipeline sampler. With no source of external noise, the policy passes `None` to
L1 so sampling stays inside the runtime rather than allocating and uploading a
host array.

## Ownership and concurrency

- A `TokenSampler` belongs to one logical sequence and is mutable.
- A prepared PI0.5 plan owns one normal generator bound to its stable latent
  allocation.
- A Python model handle owns the seed and implicit draw counter; exact provided
  noise bypasses that stream and does not advance it.
- Backend operations for model forward, normal fill, and token sampling use the
  same CUDA stream, so ordering does not require an intermediate host sync.
- The selected token must reach the host because the current generation driver
  invokes callbacks, checks EOS, and writes the next token control input there.
  Only the compact `TokenSample` crosses that boundary.
