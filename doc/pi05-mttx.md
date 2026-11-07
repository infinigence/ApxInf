# Profiling PI0.5 on MUSA with MTTX

MUSA's profiler marker API is **MTTX** (`mt-tools-extension`), the counterpart
of CUDA's NVTX — in fact the same client protocol under another name: a
header-only client `dlopen`s `$MTTX_INJECTION64_PATH` on its first API call and
hands the process's export table to whatever is behind it. `msys` is the tool
that consumes the result.

Model code never names it. It calls
`crate::profiling::trace::range("name")`, and that name resolves to NVTX on a
CUDA build, to MTTX on a MUSA build, and to an empty guard otherwise:

```rust
let _range = crate::profiling::trace::range("vision_encoder");
```

The guard pops on drop, so a range is a lexical scope.

## Recording a trace

Three things have to be true, and each fails silently on its own:

1. **The crate is built with the `mttx` feature** *and* a `mt-tools-extension`
   checkout was found at build time — `APXINF_MTTX_SRC` or `MTTX_SRC`, else
   `$MUSA_PATH/mt-tools-extension`, and nothing else is searched. Without
   either, the module compiles to no-op stubs: the surface stays, the labels do
   not. `apxinf_musa::MTTX_AVAILABLE` reports which build a process got.
   Finding the headers but failing to compile them is a hard error, because the
   failure mode of a marker API is silence and a build that claims success is
   worse.

   `apxinf-musa`'s own `default = ["mttx"]` only applies when that crate is
   built directly: `apxinf-model` and `apxinf-py` both depend on it with
   `default-features = false`. A wheel built with `--features musa` therefore
   has no markers — pass `--features musa-mttx` to link them.

   ```bash
   # absolute, or $PWD/<checkout>: the build script runs with the crate as its
   # working directory, so a relative path resolves from there
   MTTX_SRC=$PWD/<path-to-mt-tools-extension> cargo build -p apxinf-musa --features mttx
   ```
2. **`$MTTX_INJECTION64_PATH` names the injection library** when the process
   runs. The handshake happens once, on the first range, and **never retries**:
   a process started without the variable is unlabelled for its whole life, and
   nothing in it reports that.
3. **The capture is taken with `msys`.** A report with no `type=3` events has no
   ranges in it, whatever the program pushed.

`crates/apxinf-musa/src/mttx.rs` carries the reasoning for all three; it is the
only place the client is touched.

## Reading the recording

Two defects are known on this stack, and both change what a number means rather
than whether the trace exists:

- **Timestamps can come out flattened** — every range reporting the same start,
  or durations collapsing to a constant. The cause was a driver merge commit, not
  this backend; setting `MUSA_INFLIGHT_SUBMISSION_LIMIT=9999` before the process
  restores them. If a timeline looks like a comb, check that first.
- **The summed per-kernel time is not stable between collections.** Two captures
  of the same binary have differed by 12 ms in the total while the individual
  kernels were fine. So the total is not a measurement — use it to rank, and read
  **call counts** and **µs per call** instead, which are conserved across
  collections.

With no tool attached a range costs one relaxed atomic load per call and
returns; with one attached it allocates its name and makes two foreign calls
(push on construction, pop on drop). That is still cheap beside a kernel, which
is what makes labelling thousands of operators per inference affordable — but it
also means a labelled region adds no samples of its own. Ranges partition time;
they do not measure it.

## What is labelled today

The model families that place ranges (`gr00t`, `qwen3vl`) get their usual spans.
PI0.5 does not, on either backend: `pi05/` places none, so there is no upstream
call site to copy. PI0.5's MUSA port does not place its own yet either — what a
trace shows for it today is the runtime's kernels, which is enough to attribute
time to GEMM, attention and the elementwise tail, and not enough to separate the
prefix from the denoising loop by label. Adding them means calling
`profiling::trace::range` at the port's own stage boundaries in
`musa_pi05/model/`.

Keep the ranges out of `adapters/`, `kernels/custom/` and `kernels/asm/` when
adding them: those trees are hashed into `APXINF_KERNEL_BUILD_ID`, and a
host-side marker that moves the id retires a valid device cache and buries a
false difference in every later "the id changed, so the kernels changed"
reading.
