# Metal serving experiment plan

Status: experiment specification, 2026-10-09. This document contains no performance results.
The [initial matrix report](metal-results-20261009.md) records results and output differences from the first nine-profile run.

The initial target is an Apple M4 with 16 GiB of unified memory.
The worker uses Python 3.14.3, MLX 0.32.1, and MLX-LM 0.31.3.
The complete package pins remain in `scripts/apxinf_mlx_generate.py`.
The worker checks those pins before it imports MLX.

## Hardware basis

Apple specifies 120 GB/s memory bandwidth for this M4 model.
That specification describes a hardware limit, not measured serving bandwidth.
The [Mac mini specification](https://support.apple.com/en-ie/121555) identifies the memory and GPU configuration.

Apple GPUs share system memory with the CPU.
Resident weights, active state, allocator reserve, and desktop applications therefore compete for the same physical capacity.
The [Apple resource guide](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus) describes that storage model.

Apple documents a tradeoff between submission frequency and CPU-to-GPU synchronization overhead.
The [command buffer guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html) supplies this architectural basis.
Our output batches combine protocol events, not Metal command buffers.
They do not prove fewer GPU submissions.
Only device traces can establish that lower-level effect.

The local 2B configuration contains six full-attention layers and eighteen linear-attention layers.
Its full-attention configuration uses two KV heads with a head dimension of 256.
At BF16, ordinary full-attention KV storage grows by 12288 bytes per cached token across those six layers.
At 16384 tokens, that component alone requires 192 MiB.
This calculation excludes recurrent state, convolution history, temporary arrays, allocation padding, and weights.
It is a sizing estimate from the local configuration, not a measured process footprint.

A later capacity model should separate linear KV growth from fixed recurrent state and chunk-dependent temporary arrays.
That model should include system memory pressure and observed swap changes.
It should not treat all available unified memory as a GPU-only budget.

## Execution boundary

The first profile uses one resident text model and one active request.
The main worker thread constructs the model and owns every MLX operation.
Independent reader and writer threads handle protocol bytes.
The worker calls public dependency APIs. It does not implement a new Metal kernel.

`generate_step` controls prefill and decode through its public API.
The worker checks controls through the prompt progress callback and between generated tokens.
An active GPU operation can delay cancellation until the next safe boundary.

The pinned generator consumes each yielded token before the worker receives it.
The worker records this consumed position separately from reported output tokens.
The worker performs device synchronization before it acknowledges resource settlement.

## Two independent variables

| Variable | Candidate values | Expected effect to test |
| --- | --- | --- |
| Prefill step size | 64, 256, 1024 tokens | Smaller chunks can shorten control latency but add evaluation and dispatch overhead. |
| Output batch size | 1, 4, 8 tokens | Larger batches can reduce protocol overhead but increase token delivery delay. |

The first generated token always creates an immediate output event.
Later events contain up to the configured output batch size.
The worker flushes partial batches at completion or a safe stop boundary.
This experiment does not change the model weights or greedy sampler.

The full experiment contains nine configurations.
Use identical model artifacts across all nine configurations.
Record a separate model revision for each configuration.
The current model manifest includes execution settings in that revision.
Do not compare different model quantizations as the same execution experiment.

## Workload matrix

| Workload | Input | Output allowance | Observation |
| --- | --- | --- | --- |
| Interactive | A fixed short chat prompt | 128 tokens | First output delay and protocol overhead |
| Long input | Fixed rendered inputs near 2K and 8K tokens | 128 tokens | Prefill duration and memory peak |
| Decode | A fixed short prompt | 512 tokens | Token delivery intervals and total duration |
| Prefill cancellation | A fixed input near 8K tokens | 512 tokens | Cancel acknowledgement and resource settlement |
| Decode cancellation | A fixed short prompt | 512 tokens | Stop latency after a measured output boundary |
| Queue interference | One long request and one short request | Fixed per request | Queue delay under the serial profile |
| Slow reader | The decode workload with a throttled client | 512 tokens | Bounded output memory and cancellation |

Queue interference does not demonstrate continuous batching.
The serial profile queues the second request until the first request settles.
A later batch profile needs its own mixed prefill and decode experiment.

## Measurement procedure

1. Reserve exclusive use of the device for the experiment.
2. Record the hardware, OS, model manifest, runtime pins, and worker identity.
3. Load one model and complete one untimed warmup request.
4. Save the authoritative rendered token IDs for each workload.
5. Run each configuration with identical input IDs and output limits.
6. Repeat requests in a recorded, balanced order.
7. Record sample counts with every latency percentile.
8. Record output tokens and compare them before making performance claims.
9. Close the worker before the next device user starts.

Do not use a simultaneous Metal workload as a baseline.
Record any background memory pressure or swap activity.
Keep cold load measurements separate from warm request measurements.

Capture these timestamps at the gateway:

- Client request arrival.
- Admission and worker submission.
- First token event and first visible output.
- Each output event.
- Control submission and command acknowledgement.
- Terminal result and resource settlement.

Capture worker elapsed time, first-token time, and MLX peak allocation bytes.
The client trace determines observable token latency.
Worker metrics do not replace client timing or OS memory measurements.

## Memory controls

The worker passes `--memory-limit-bytes` to public `mlx.core.set_memory_limit` before loading the model.
The pinned API describes this limit as a guideline during graph evaluation.
It can still use RAM and available swap beyond that value.
It is not an OS memory quota or a physical memory guarantee.

The worker does not increase the wired-memory limit.
The supervisor separately checks resident allocation and sequence reservations.
Those estimates do not replace process footprint, memory pressure, and swap observations.

The initial worker caps output bytes and reserves capacity for terminal events.
The worker retains command digests instead of completed prompt copies.
Device settlement releases active state before capacity leases return to the coordinator.
Allocator reserve remains a separate service budget item.

The current supervisor checks resident allocation at startup.
Its request reservation metric excludes the resident model.
Its peak metric does not report current process memory.
A future memory controller needs current allocation, process footprint, and pressure measurements after each settlement.

## Proposed latency-budget policy

Status: design proposal. The current worker does not implement this policy.
This proposal combines measured profiles, a resident model, and a time limit for host output batches.
It does not require a new Metal kernel.

A latency budget is an allowed delay for one observable request phase.
Set separate budgets for first visible output, output delivery, and cancellation settlement.
Keep queue delay separate from model execution delay.
The serial profile cannot protect short requests from an active long request.

The initial selector runs before worker startup.
It chooses one tested prefill size and output batch size for a declared workload mix.
It rejects candidates that exceed any accepted latency or memory threshold.
It chooses the remaining candidate with the highest measured completed-token rate.
If no candidate passes, retain the reference profile and report the unmet budgets.

Store the following evidence with each profile:

- Hardware and OS identity.
- Model artifact hashes, runtime pins, and execution identity.
- Input length bands and output allowances.
- Sample counts, token comparisons, and observed latency distributions.
- Peak allocation, process footprint, memory pressure, and swap observations.
- Thresholds, accepted workload mix, and selection result.

Do not transfer an M4 result to a different device without another measurement.
Do not claim a percentile bound from a small demonstration run.
Use separate training and validation samples when selecting a profile.

The next output policy adds a maximum wait for computed tokens.
The owner starts that wait when the first token enters an empty pending batch.
It sends the batch when the token limit or time limit occurs at a safe boundary.
The first token and terminal events remain immediate.
Cancellation retains the existing token-range and terminal contract.
The independent writer can send queued bytes while the owner evaluates the next token.

This timer controls host batching delay after computation.
It cannot interrupt an active MLX operation.
The observed delay can exceed the timer by one device step and subsequent transport delay.
Measure that excess before choosing a budget.
Do not present the timer as a hard deadline.

Use measured prefill callback intervals to select a chunk size.
Include control delivery, device synchronization, and cleanup in the cancellation budget.
Changing a chunk size can change numerical execution order.
Compare generated token IDs for every candidate before accepting its latency result.

The current worker fixes execution settings in its startup manifest.
A future policy must define its configuration and identity fields before implementation.
Changing settings within a worker requires a new explicit capability contract.
The selector must not silently change precision, model artifacts, or dependency versions.

This proposal tests a concrete serving improvement: fewer host output events within explicit observed latency budgets.
The nine fixed profiles provide its first evidence.
If those measurements show no useful tradeoff, retain immediate token delivery.

## Matrix driver

`benchmarks/serving/metal_matrix.py` runs the nine fixed profiles in separate processes.
It requires an existing Metal lock that another process holds.
The caller must acknowledge that reservation with `--lock-owned-by-parent`.
The caller must unload its previous model before starting the driver.
The driver does not acquire a second long-term device lock.

The driver uses port 8081 and the `apxinf-local` model alias.
Each profile contains two warmup requests and sixteen measured requests.
The warmup covers one long input and one short input.
Measured requests use the mixed workload, concurrency one, and an output allowance of 32 tokens.
These samples support exploratory means and medians, not a performance guarantee.

The driver checks artifact records, runtime identity, and execution settings before each measurement.
It saves readiness, startup duration, warmup results, raw request results, process identifiers, logs, and swap observations.
Startup duration includes process creation, model loading, and identity inspection.
It compares visible output hashes by request index.
The HTTP trace does not expose token IDs, so this comparison cannot establish token parity.

Each server starts in a new process group.
The driver signals only that group during cleanup.
It waits until that group has no live processes before starting another profile.
Process inspection requires the host execution permissions used for actual Metal measurements.
If cleanup cannot establish exit, the driver stops the matrix.

Example after the parent reserves the device and unloads its previous model:

```sh
python3 benchmarks/serving/metal_matrix.py \
  --model /absolute/path/to/local/model \
  --output-dir benchmarks/serving/results/metal-matrix-run \
  --lock-owned-by-parent
```

The output directory must not exist before the run.
The driver refuses a free lock or an occupied port.
Its fixed profile order can introduce thermal or temporal bias.
Repeat promising configurations in a balanced order before choosing a default.

## Acceptance and reporting

No candidate becomes the default solely because it reduces worker elapsed time.
Compare first output latency, delivery intervals, cancellation settlement, memory, and completed output together.
Investigate any token divergence before accepting the candidate profile.
Use a fixed reference profile for the model quality comparison.

The initial values of 256 prefill tokens and one output token are provisional settings.
They are not measured optimum values.
Record accepted thresholds before choosing a performance default.

## Static source checks

The pinned Qwen3.5 MLX model constructs a text model from `text_config`.
Its weight adapter excludes vision weights from that text model.
The local Qwen3.5-2B configuration uses this model family.
These source checks establish an available loader path, not a successful model run.

The public loader can honor a `model_file` entry in the model configuration.
The worker rejects `model_file` and `auto_map` before loading.
The worker also disables remote tokenizer code and external model downloads.

Primary references: [MLX memory API](https://ml-explore.github.io/mlx/build/html/python/memory_management.html),
[fixed MLX-LM generation API](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/generate.py),
and [fixed Qwen3.5 text model](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/qwen3_5.py).
The installed MLX 0.32.1 type stub supplies the memory-limit wording used above.
