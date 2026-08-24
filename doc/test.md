# ApxInf Test Documentation

## Running Tests

```bash
# Run the portable library test suite
cargo test -p apxinf-core -p apxinf-loader -p apxinf-model -p apxinf-tokenizer --lib

# Compile all workspace library, binary, and test targets
cargo check --workspace --lib --bins --tests

# Run tests for a specific crate
cargo test -p apxinf-core
cargo test -p apxinf-loader
cargo test -p apxinf-model
cargo test -p apxinf-tokenizer

# Run a specific test
cargo test -p apxinf-core test_matmul_cpu
```

## Test Summary

The portable suite currently contains 65 tests across `apxinf-core`,
`apxinf-loader`, `apxinf-model`, and `apxinf-tokenizer`. CUDA compilation is
covered by the workspace check above; CUDA-linked tests and benchmarks run on
NVIDIA devices.

---

## apxinf-core

### Shape Tests (`shape::tests`)

#### `test_numel`
Tests that `Shape::numel()` correctly computes the total number of elements.
```rust
Shape::new(vec![2, 3, 4]).numel() == 24
Shape::new(vec![1]).numel() == 1
Shape::scalar().numel() == 1
```

#### `test_strides`
Tests that row-major strides are computed correctly.
```rust
Shape::new(vec![2, 3, 4]).strides() == vec![12, 4, 1]
```

#### `test_dim_negative`
Tests negative indexing for dimension access (Python-style).
```rust
shape.dim(-1) == last dimension
shape.dim(-2) == second-to-last dimension
```

#### `test_display`
Tests the `Display` trait implementation for `Shape`.

#### `test_matmul_shape`
Tests shape inference for 2D matrix multiplication.
```rust
[2, 3] @ [3, 4] -> [2, 4]
```

#### `test_matmul_shape_batch`
Tests shape inference for batched matrix multiplication.
```rust
[1, 2, 3] @ [1, 3, 4] -> [1, 2, 4]
```

#### `test_matmul_dim_mismatch`
Tests that mismatched dimensions return an error.
```rust
[2, 3] @ [4, 5] -> Error (3 != 4)
```

### Tensor Tests (`tensor::tests`)

#### `test_from_f32`
Tests creating a tensor from f32 data.

#### `test_from_bf16`
Tests creating a tensor from bf16 data.

#### `test_to_f32_vec_from_bf16`
Tests converting bf16 tensor data back to f32.

#### `test_zeros`
Tests creating a zero-filled tensor.

#### `test_reshape`
Tests reshaping a tensor to a compatible shape.

#### `test_reshape_mismatch`
Tests that reshaping to incompatible shape returns an error.

#### `test_transpose_2d`
Tests 2D tensor transposition.

#### `test_matmul_cpu`
Tests CPU matrix multiplication.
```rust
[2, 3] @ [3, 2] -> [2, 2]
```

#### `test_display`
Tests the `Display` trait implementation for `Tensor`.

---

## apxinf-loader

### SafeTensors Tests (`safetensors::tests`)

#### `test_load_f32_tensor`
Tests loading a single f32 tensor from SafeTensors format.

#### `test_load_bf16_tensor`
Tests loading a bf16 tensor and converting to f32.

#### `test_load_multiple_tensors`
Tests loading multiple tensors from a single SafeTensors file.

#### `test_unsupported_dtype_skipped`
Tests that unsupported dtypes (e.g., INT8) return an error.

### GGUF Tests (`gguf::tests`)

#### `test_load_f32_tensor`
Tests loading a single f32 tensor from GGUF format.

#### `test_load_metadata`
Tests parsing GGUF metadata key-value pairs.

#### `test_config_from_metadata`
Tests extracting `ModelConfig` from GGUF metadata.

#### `test_multiple_tensors`
Tests loading multiple tensors from a GGUF file.

#### `test_bad_magic`
Tests that files with incorrect magic number return an error.

---

## apxinf-model

### Llama Model Tests (`llama::tests`)

#### `test_model_construction`
Tests building a LlamaModel from weight tensors:
- Verifies all weight names are correct
- Tests weight transposition (PyTorch -> matmul format)
- Verifies model can be constructed without errors

#### `test_kv_cache`
Tests KV cache operations:
- Creating empty cache
- Verifying cache dimensions match config

#### `test_generation`
Tests end-to-end token generation:
- Build model from weights
- Encode prompt tokens
- Run autoregressive generation
- Verify output token count

---

## Running TinyLlama Model

### Prerequisites

1. Download TinyLlama 1.1B Chat model files:
```bash
mkdir -p models/tinyllama
cd models/tinyllama

# Download tokenizer files
curl -L -o tokenizer.json \
  "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json"
curl -L -o tokenizer_config.json \
  "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json"

# Download model weights
curl -L -o model.safetensors \
  "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"
```

2. Build the release binary:
```bash
cargo build --release
```

### Running Inference

```bash
./target/release/apxinf generate \
  -m models/tinyllama/model.safetensors \
  -t models/tinyllama/tokenizer.json \
  -p "Once upon a time" \
  --max-tokens 50
```

### CLI Options

```
apxinf generate [OPTIONS]

Options:
  -m, --model <PATH>       Path to model weights (SafeTensors format)
  -t, --tokenizer <PATH>   Path to tokenizer file (tokenizer.json)
  -p, --prompt <TEXT>      Input prompt text
  --max-tokens <N>         Maximum new tokens to generate [default: 50]
```

### Quick Test

```bash
# Run the built-in CPU matmul verification
./target/release/apxinf test
```

### Expected Output

```
apxinf — LLM inference engine

Loading tokenizer from "models/tinyllama/tokenizer.json"...
Vocab size: 32000
Loading model from "models/tinyllama/model.safetensors"...
Loaded 201 tensors
Model config: hidden=2048, layers=22, heads=32
Building model...
Model ready.

Prompt: Once upon a time
Prompt tokens: [9682, 682, 451, 825]

Generating up to 50 tokens...

Output: Once upon a time...
```

### Notes

- The model runs on CPU by default
- Build with `--features cuda` and select `--device cuda` for NVIDIA GPU inference
- Loads model defaults from `generation_config.json`; missing settings retain
  deterministic greedy compatibility behavior. `--generation-config apxinf`
  ignores model defaults.
- `--sample` enables backend sampling with temperature, top-k, top-p,
  repetition/frequency/presence penalties, and a reproducible seed. On CUDA,
  the logits pipeline stays on the GPU and returns only the sampled result.
- The complete sampling test matrix and Thor commands are in
  [`doc/20260819-sampling-subsystem/test.md`](20260819-sampling-subsystem/test.md).
