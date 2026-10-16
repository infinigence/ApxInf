// Adapted from the EngineTailor Qwen3-TTS predictor MLP frontier 944fd18d.
// Copyright (c) 2026 Haiyan Qin. MIT; see NOTICE.
// BF16 gate/up rounding, FP32 SiLU with precise exp, BF16 product.
    struct QwenTTSActivateMultiply {
      bfloat16_t operator()(float gate_acc, float up_acc) thread {
        const bfloat16_t gate = static_cast<bfloat16_t>(gate_acc);
        const bfloat16_t up = static_cast<bfloat16_t>(up_acc);
        const float f = static_cast<float>(gate);
        const float exponent = metal::precise::exp(-f);
        const float silu_f = f / (1.0f + exponent);
        const bfloat16_t silu = static_cast<bfloat16_t>(silu_f);
        return static_cast<bfloat16_t>(silu * up);
      }
    };
