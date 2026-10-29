# Native prepared-code diagnostics

These tools compare direct and HTTP generation with externally recorded executable identities.
They produce diagnostic evidence, not a performance acceptance decision.
Both paths select ordinary MiniCPM5 BF16 with 256-token prefill chunks.
The process budget is 10 GiB and the allocator cache limit is 2 GiB.
The prepared profile reserves 1 GiB for the next request.
Both paths reject an input and output reservation above 16384 tokens.

## Preparation

Use the same complete checkpoint, dependency SDK and source for both paths.
Build the native worker and probe against MLX 0.32.3.
The tools add no external runtime dependency.
The probe uses existing model and tokenizer interfaces.
The comparison client uses the standard library and the existing Metal lock check.

```sh
cargo build --release --locked -p apxinf-serving --features native-minicpm5
cargo build --release --locked -p apxinf-model --features mlx --example metal_text_probe
```

Supply an original JSON corpus with this shape:

```json
{"cases":[{"id":"cache-question","messages":[{"role":"user","content":"Explain why a cache can reduce repeated work."}],"max_tokens":64}]}
```

Use unique case IDs.
Keep identical prompt and output settings for a same-shape repetition.
Include different real input lengths to examine profile replacement.
Record actual rendered input counts instead of assuming character counts equal tokens.
The example above does not constitute long-context coverage.

## Execution ownership

Run all model work inside the host's existing bounded Metal lease.
The owner must acquire every shared lock required by that host.
Do not load the direct model while another resident service owns the device.
Finish the probe before starting the HTTP model.
The service and its client belong to the same owned job.
Stop and reap the service before releasing the lease.
The comparison client checks the selected lock's contention and requires an ownership acknowledgement.
That observation does not prove the external owner's full resource policy.

From a clean source checkout, run the probe inside that lease:

```sh
target/release/examples/metal_text_probe /model/snapshot /run/corpus.json 3 retain-code \
  > /run/direct.jsonl
```

Use `release-all` for a direct cleanup control.
The probe records its executable digest, observed checkout, actual library, model directory and memory policy.
The checkout fields describe the process working directory, not the source used to compile the executable.
Each request records complete token IDs, text, stop reason and engine timings.
Preparation remains included in engine TTFT.
The probe reports cleanup and retained allocation separately.
The probe does not identify itself as a numerically independent oracle.

The build owner must retain a build receipt before claiming source binding.
The receipt records `source_commit`, clean `source_dirty`, and a `binaries` object.
Clean `source_dirty` is Boolean false or the empty Git status string.
Binary entries `metal_text_probe` and `apxinf-native-worker` contain actual `bytes` and lowercase `sha256` values.
Record these observations after building the selected clean source.
The receipt remains external evidence, not an executable's self-reported compiler history.

Start the selected service after the probe exits:

```sh
target/release/apxinf-serve --provider native-minicpm5-prepared \
  --native-worker "$PWD/target/release/apxinf-native-worker" \
  --metal-library "$MLX_ROOT/lib/libmlx.dylib" --model /model/snapshot \
  --model-id minicpm5 --max-context 16384 --max-output-tokens 256 \
  --allocator-cache-limit-bytes 2147483648
```

Wait for `/readyz` inside the owned job.
Use a new output path for each comparison:

```sh
python benchmarks/serving/native_runtime_compare.py --base-url http://127.0.0.1:8080 \
  --corpus /run/corpus.json --direct-jsonl /run/direct.jsonl \
  --native-worker "$PWD/target/release/apxinf-native-worker" \
  --native-probe "$PWD/target/release/examples/metal_text_probe" --build-receipt /run/build-receipt.json \
  --metal-lock "$HOME/.cache/enginetailor/metal-measure.lock" --lock-owned-by-parent \
  --output /run/comparison.json
```

The client checks the ready profile, both executable digests, model path and memory policy.
With a build receipt, it checks both actual files and probe metadata against the recorded clean source.
Without that receipt, the report marks source binding unverified.
The receipt covers the selected probe and native worker files.
The external owner must separately bind the running server process to its build.
Matching model directory paths does not establish model artifact equality.
The report records that limitation instead of claiming independent weight verification.
It checks complete response text, final token usage and stop reason.
A mismatch remains in the output report and produces a nonzero exit status.
An existing report is never overwritten.

Retain the service stderr with the comparison.
Its `native_request_settled` records include preparation, text decoding, cleanup and retained memory.
Combine those observations with client TTFT, wall time and the existing service queue metrics.
For a performance claim, add prospectively fixed paired repeats, host health observations and the numerical gate.
