# ApxInf LLM Inference Engine - System Design

```mermaid
graph TB
    subgraph CLI["CLI Entry Point"]
        MAIN["apxinf (main.rs)"]
        MAIN --> |"--model"| LOAD["Model Loading"]
        MAIN --> |"--tokenizer"| TOK["Tokenizer"]
        MAIN --> |"--prompt"| GEN["Generation"]
        MAIN --> |"--device cuda/cpu"| DEV["Device Selection"]
    end

    subgraph Core["apxinf-core"]
        TENSOR["Tensor"]
        SHAPE["Shape"]
        DTYPE["DType (F32/BF16)"]
        STORAGE["Storage (Cpu/Gpu)"]
        DEVICE["Device Enum"]
        OPS["CPU Ops (matmul, blas)"]

        TENSOR --> SHAPE
        TENSOR --> DTYPE
        TENSOR --> STORAGE
        TENSOR --> DEVICE
        STORAGE --> |"CpuStorage"| CPU_MEM["Vec<u8>"]
        STORAGE --> |"GpuStorageHandle"| GPU_MEM["Arc GPU Buffer"]
    end

    subgraph CUDA["apxinf-cuda"]
        CUDA_CTX["CudaContext"]
        CUDA_BUF["CudaBuffer"]
        CUBLAS["cuBLAS Handle"]
    end

    subgraph Loader["apxinf-loader"]
        SAFETENSORS["SafeTensors Loader"]
        CONFIG["ModelConfig"]
        SAFETENSORS --> |"parse metadata"| CONFIG
        SAFETENSORS --> |"HashMap<String, Tensor>"| WEIGHTS["Weight Tensors"]
    end

    subgraph Model["apxinf-model"]
        LLAMA["LlamaModel"]
        WEIGHTS_STRUCT["LlamaWeights"]
        LAYER["TransformerLayer"]
        KV_CACHE["KVCache"]
        FORWARD["forward_single()"]
        GENERATE["generate_streaming()"]

        LLAMA --> WEIGHTS_STRUCT
        WEIGHTS_STRUCT --> |"Vec"| LAYER
        LLAMA --> KV_CACHE
        LLAMA --> FORWARD
        LLAMA --> GENERATE
        FORWARD --> |"ops"| RMS_N["rms_norm"]
        FORWARD --> MATMUL_N["matmul"]
        FORWARD --> ROPE_N["rope"]
        FORWARD --> SILU_N["silu"]
        FORWARD --> ATTENTION["attention"]
        FORWARD --> MLP["mlp"]
    end

    subgraph Tokenizer["apxinf-tokenizer"]
        HF_TOK["HfTokenizer"]
        CHAT_MSG["ChatMessage"]
        CHAT_TPL["Chat Template (minijinja)"]
        ENCODE["encode()"]
        DECODE["decode()"]
        EOS["eos_token_id"]

        HF_TOK --> ENCODE
        HF_TOK --> DECODE
        HF_TOK --> CHAT_TPL
        CHAT_TPL --> |"Jinja2"| CHAT_MSG
    end

    %% Relationships
    MAIN --> |"uses"| LLAMA
    MAIN --> |"uses"| TOK

    LLAMA --> |"depends on"| TENSOR
    LLAMA --> |"cfg(feature=cuda)"| CUDA_CTX

    TENSOR --> |"Device::Cuda"| CUDA_CTX
    TENSOR --> |"Device::Cpu"| OPS

    SAFETENSORS --> |"creates"| TENSOR

    DEVICE --> |"Cuda(0)"| CUDA_CTX
    DEVICE --> |"Cpu"| OPS

    style MAIN fill:#f9f,stroke:#333,stroke-width:2px
    style LLAMA fill:#bbf,stroke:#333,stroke-width:2px
    style CUDA_CTX fill:#9f9,stroke:#333,stroke-width:2px
    style TENSOR fill:#ff9,stroke:#333,stroke-width:2px
```

## Crate Structure

| Crate | Purpose | Key Components |
|-------|---------|----------------|
| `apxinf` | CLI entry point | argparse, generation loop, streaming output |
| `apxinf-core` | Core tensor infrastructure | Tensor, Shape, DType, Storage, Device enum |
| `apxinf-cuda` | CUDA GPU backend | CudaContext, CudaBuffer, cuBLAS and custom kernels |
| `apxinf-loader` | Model weight loading | SafeTensors parser, ModelConfig |
| `apxinf-model` | Model architecture | LlamaModel, TransformerLayer, KVCache, forward pass |
| `apxinf-tokenizer` | Text encoding/decoding | HfTokenizer wrapper, chat template support |

## Data Flow

1. **Loading**: SafeTensors → HashMap<String, Tensor> → LlamaModel
2. **Device Transfer**: CPU Tensor → CUDA Tensor
3. **Forward Pass**:
   - Embedding lookup → [1, hidden_size]
   - For each layer: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual
   - Final RMSNorm → Output projection → [1, vocab_size]
4. **Generation**: argmax(logits) → token_id → decode → text

## GPU Backend Dispatch

```rust
match device {
    Device::Cpu => cpu_op(...),
    Device::Cuda(_) => cuda_op(...),
}
```
