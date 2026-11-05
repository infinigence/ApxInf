# apxinf-musa

MUSA backend for ApxInf, with an independent Moore Threads runtime derived
from the former `apxinf-cuda` backend.

Installing the MUSA driver and SDK on an M1000 is covered by Moore Threads'
[environment setup guide](https://docs.mthreads.com/vllm-musa-m1000/vllm-musa-m1000-doc-online/AIModule/AIModule1.4.1+vllm1.4.2/environment_setup/).

| Piece | State |
|---|---|
| Build contract (`build.rs`, `MUSA_PATH`, optional toolkit) | in place |
| `musa` / `mttx` features | in place |
| Device runtime (context, stream, buffer, transfers, workspace, graph) | in place |
| Native kernel facades and W8A8 GEMM | in place |
| Typed-operator facade (`ops`) | in place |
| PI0.5 BF16 port | in place (`crates/apxinf-model/src/musa_pi05/`) |
| PI0.5 INT8 (W8A8) routes | in place: `int8_dynamic` and `int8_dynamic_no_encode`, fused GeGLU quantization, NN/NT decode images |
| Fixed-shape preparation and capture | in place: per-stream vendor handles, same-stream warm-up, per-session shape coverage |

This file is the status summary. The architecture, the relationship to the
current CUDA runtime and its predecessor, the INT8 routes, capture, the tuning
database, serving and the accuracy gate live in
[the backend document](../../doc/musa-backend.md).

Capture is the contract to read before changing any operator: vendor state is
created on first use per shape and per stream, and only outside capture. An
`ExecutionSession::prepare` traversal is what creates it; `ExecutionSession::run`
refuses a GEMM or attention shape the prepare pass did not reach, and
`prepared_shapes()` lists what that pass covered.

`cargo run -p apxinf-musa --example device_report` reports the target the
build produced and round-trips a buffer through the device.

## Where the numbers in these files come from

Kernel comments cite measurements taken on an M1000 with probe programs that are
not part of this repository. They state the shape, the device and the result,
and they stop there: a comment never carries a path to the probe. The working
directory `devlocal/` is ignored by Git, so a path into it dangles for every
other checkout and cannot be checked by reading the tree. A check that has to
be repeatable belongs in a `#[test]` or a committed tool, not in a comment.
