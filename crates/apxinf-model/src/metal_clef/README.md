# Native Clef on MLX

This Metal port of the Clef decision family reads a game frame, answers the
typed questions a caller attaches to it, and returns per-question probabilities.
It calls safe `apxinf-mlx` array operations; there is no Python and no external
model provider. Registration name: `metal_clef`. Target:
`mlx-community/clef-flash-4bit` (Qwen3.5 hybrid backbone, 4-bit affine),
single image, Apple Silicon macOS.

## What this port owns

| File | Responsibility |
| --- | --- |
| `config.rs` | Admit the checkpoint's text, **vision** and joint-head geometry; nothing is inferred |
| `weights.rs` | Load the backbone, the vision tower and the joint head through one checked loader |
| `preprocess.rs` | Bind the checkpoint's preprocessing contract to the shared `vision_preprocess` |
| `vision.rs` | The 27-block tower, its merger and the mRoPE position builder |
| `model.rs` | Forward composition, composition selection and the image splice |
| `encode.rs` | Tokenize the state and questions; expand image placeholders |
| `ane.rs`, `ane_admission.rs` | GPU+ANE splits (columns on M1 Pro/M2, neurons on M4) |

Image container decoding is **not** here: `vision_preprocess` documents that the
application owns container formats and the loaded model owns layout, dtype and
image-token expansion. Callers hand `DecodedImage` (RGB8 + extents) to
`SchemaDecisionRuntime::decide_schema_with_images`.

## Request path

1. `encode` tokenizes the state and the typed questions, and emits one image
   placeholder per merged patch row.
2. `preprocess` resolves the checkpoint's contract once, then patchifies each
   image to `[rows, 1536]` FP32 rows in request order.
3. `vision` runs the tower and returns `[rows, hidden]` features.
4. `model` splices those features into the placeholder rows and runs the
   backbone with three-axis mRoPE positions derived from the grids.
5. The joint head yields one probability row per question, in request order.

A request without images takes the same path with equal axes, so the text
result is unchanged.

## Compositions

`auto` selects from the Metal architecture; `reference` is always available, and
`gpu-ane-columns` / `gpu-ane-neurons` are explicit variants behind `coreml`.

## Tests

```sh
cargo test -p apxinf-model --features coreml --lib metal_clef::
```

The tower and preprocessing are covered by device tests; the single-image
parity fingerprint is `enemy.noul = 0.023870831355452538` against the reference
decide-server, and the text fingerprint is unchanged at `0.020645929500460625`.
