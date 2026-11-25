# PI0.5 INT8 on an 8 GB GPU

`int8_dynamic` (W8A8) holds the PI0.5 weights in 4024 MiB, so the model fits an
8 GB card. What does not fit by default is the static workspace, which this guide
shows how to size.

Tested on one device: an RTX 4060 (8 GB, sm_89, CUDA 13.2) with
`lerobot/pi05_libero_base`, two views, an action horizon of 10 and 10 prompt
tokens. BF16 loads 6642 MiB of weights on the same card and does not run there.

The checkpoint's own `config.json` sets `chunk_size` to 50. The LIBERO serving
command in the [root README](../README.md#using-apxinf-from-python) uses
`--action-horizon 10`, which is the configuration measured here. A horizon of 50
reserves more: 9144 MiB at 21 tokens, against 7752 MiB at horizon 10.

## Why the default fails

Each PI0.5 session reserves one static workspace for every intermediate of a
request. The reservation is a conservative bound that counts all intermediates as
live at once (`Pi05Config::cuda_graph_workspace_bytes_int8_dynamic`, twice the BF16
bound). For the configuration above it is 7660 MiB. With the weights resident the
reservation cannot succeed on 8 GB, and the first request fails with
`CUDA error 2: out of memory`.

## `APXINF_PI05_ARENA_MIB`

`APXINF_PI05_ARENA_MIB=<MiB>` replaces the capacity of the static workspace of the
prepared (CUDA Graph) sessions and of the eager session that `PI05` runners create
for a request. The variable is read when a session is created, so set it before the
process prepares a model. `pi05_bench` captures its graphs through the same path and
honours the variable too.

- Unset keeps the bound, so deployments that do not set it are unchanged.
- A value that is not a positive whole number of MiB, or that overflows in bytes,
  is an error that names the variable. It is never ignored.
- A capacity below what a request needs fails that request with
  `static inference workspace exhausted: need N contiguous bytes, capacity is M bytes`.
  `N` is the size of the one allocation that did not fit, not the total workspace
  the request needs.

Measured on the device above, horizon 10, 10 prompt tokens, CUDA Graph, autotune off
(recipes already present), whole-GPU memory sampled every 0.2 s:

| `APXINF_PI05_ARENA_MIB` | Outcome | GPU peak | p50 of 50 replays |
| --- | --- | --- | --- |
| unset | `CUDA error 2: out of memory` before the first request | 4024 MiB | - |
| 48 | `workspace exhausted: need 34209792 contiguous bytes, capacity is 50331648 bytes` | 4024 MiB | - |
| 64 | `workspace exhausted: need 17104896 contiguous bytes, capacity is 67108864 bytes` | 4024 MiB | - |
| 80 | `workspace exhausted: need 8552448 contiguous bytes, capacity is 83886080 bytes` | 4024 MiB | - |
| 96 | runs | 4452 MiB | 63.56 ms |
| 128 | runs | 4484 MiB | 63.65 ms |
| 256 | runs | 4612 MiB | 63.56 ms |
| `abc` | `APXINF_PI05_ARENA_MIB="abc" is not a positive number of MiB that fits in the address space` | 4024 MiB | - |

The smallest capacity that runs is 92 MiB: 91 MiB fails and 92 MiB runs, each
reproduced twice. The failing allocation changes with the capacity (8552448 bytes
at 80, 88 and 90 MiB; 2138112 bytes at 91 MiB), so the boundary depends on how the
allocator packs the request, not on a single total. Pick a capacity from a run of
your own input, start well above the boundary, and leave room for longer prompts.
92 MiB is not a safe margin.

The first failing value in each row is the amount the request needed in the
contiguous block that failed, not the full workspace need. Read it as a hint for
the next value to try.

## GEMM recipes

The variable moves only the reservation. A GEMM shape without a recipe resolves to
the cuBLAS fallback, and on this card the first request then fails with
`ApxInf native status 3: out of memory` even at 256 MiB. One run with autotune
enabled writes the missing recipes, and later runs reuse them. The recipes are kept
under `configs/tuning/<vendor>/<device>-sm<version>/`; see
[GEMM tuning databases](../configs/tuning/README.md#cuda-new-recipes). The
allocation behind the fallback failure was not traced.

Enable autotune with `LoadOptions.autotune`. The websocket server exposes it as
`--autotune`.

## Serving and evaluating on the 8 GB card

Build the `apxinf_py` binding with CUDA (`maturin build --release --features cuda
-m crates/apxinf-py/Cargo.toml`), install it with `python/apxinf[serving]`, and
start the server with the workspace sized and autotune on, following the
[LIBERO evaluation](../README.md#libero-evaluation) setup for the checkpoint and
its `norm_stats.json`:

```bash
export APXINF_PI05_ARENA_MIB=256
python scripts/pi05_openpi_websocket_server.py --model-variant int8_dynamic \
    --model-dir <path-to-model> --norm-stats <path-to-model>/norm_stats.json \
    --autotune --action-dim 7 --action-horizon 10 --num-views 2 \
    --image-keys observation/image,observation/wrist_image \
    --state-key observation/state --host 127.0.0.1 --port 8000
python scripts/eval_libero.py --backend websocket --precision int8 \
    --suite libero_10 --tasks all --trials-per-task 10 \
    --results-jsonl <out>/results.jsonl --summary-json <out>/summary.json
```

Measured on the card above, built from this tree with no architecture override
and no local changes: the server starts with the GEMM recipes empty, the first
request succeeds (model p50 63.1 ms over 6 requests, whole-GPU peak 4618 MiB),
and `pi05_remote_client.py` and `eval_libero.py` run against it. LIBERO-10 with
10 episodes per task finished 91 of 100, with a call-weighted model time of
63.56 ms and a request round trip of 65.61 ms over 5533 calls; the serving
process held 4580 MiB. Ten episodes per task are too few to compare with the
published success rate.

## Build

The CUTLASS W8A8 kernel has to be built for sm_89. That build comes from the
ConvRot INT8 commit this change sits on, not from this one. The build targets the
visible GPU; set `APXINF_CUDA_ARCH=sm_89` only when building on another machine
(see [Build ApxInf](../README.md#build-apxinf)). Without the kernel, the unaligned
language GEMMs take the cuBLAS fallback described above.

## Limits

- One GPU was tested. Orin and other cards were not.
- The 92 MiB boundary is for horizon 10 and 10 tokens. Other horizons or longer
  prompts need a larger workspace.
- BF16 does not fit this card, so the variable does not help there.
- The bound itself is unchanged. Deriving a tighter bound from the captured graph
  would remove the need for the variable and is separate work.
