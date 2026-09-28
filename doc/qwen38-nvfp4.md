# Qwen3.8-27B-NVFP4

Text LLM with 64 layers — 48 Gated DeltaNet (linear attention) + 16 full
attention — quantized to mixed NVFP4/FP8 by NVIDIA ModelOpt. ApxInf runs it
on Jetson AGX Thor (sm_110): batch 1, BF16 KV cache, CUDA-graph decode.

Implementation: `crates/apxinf-model/src/qwen38/`, organized per
[Model Layer Architecture](model-layer-architecture.md).

## Requirements

- Jetson AGX Thor (sm_110) with the CUDA toolkit; ~20 GiB free device memory.
- The checkpoint directory as published (safetensors shards, `config.json`,
  tokenizer assets). No conversion needed — `AutoModel` detects the model
  from `config.json`.

```bash
cargo build --features cuda --release --bin apxinf
```

## Inference with the CLI

```bash
./target/release/apxinf generate \
    --model <path-to-Qwen3.8-27B-NVFP4> \
    --device cuda \
    --prompt "What is the capital of France? Answer in one sentence."
```

```
<think>
The user asks for the capital of France ... This is a straightforward
factual question.
</think>

The capital of France is Paris.

=== Generation Profile ===
TTFT:             752.4 ms
TPOT:             77.4 ms/token
Generation TPS:   12.92 tok/s
=========================
```

The prompt is rendered with the checkpoint's own `chat_template.jinja`; the
model reasons inside `<think>…</think>` before answering.

| Flag | Effect |
|---|---|
| `--max-tokens N` | generation budget |
| `--system TEXT` | override the system message |
| `--no-eos-stop` | fixed-length generation |
| `--sample --temperature T --top-k K --top-p P --seed S` | categorical sampling |

Latency notes: the first process on a machine additionally pays one-time
kernel autotuning (recipes persist under `/tmp/apxinf-qwen38-*-recipes`);
each CLI invocation pays weight loading and one CUDA-graph capture. A
resident process reaches steady state from its second generation.

## Benchmarking

```bash
cargo run -p apxinf-model --features cuda --release --example qwen38_bench -- \
    <path-to-Qwen3.8-27B-NVFP4> --prompt-len 2048 --max-new 128 --repeats 3
```

Deterministic prompt through the public entry (`AutoModel` →
`generate_streaming`); JSON per repeat plus a summary. Repeat 0 is the
warm-up and excluded from the means. The run asserts identical tokens
across repeats.

Jetson AGX Thor, locked clocks (1575 MHz GPC):

| Metric | Steady state |
|---|---:|
| TTFT, 2048-token prompt | 534 ms |
| Decode | 78.1 ms/token (12.8 tok/s) |

## Correctness

- Quantizer contract suites: `qwen38_fp8_quant_contract`,
  `qwen38_nvfp4_quant_contract` (`apxinf-cuda-new`).

## Limitations

- Batch 1, text only; prompt + generation bounded by the KV capacity fixed
  at load (`max_seq_len`, minimum 4096).
- Validated on sm_110 only.
