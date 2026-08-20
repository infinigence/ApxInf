#!/usr/bin/env python3
"""Roofline performance analysis for TinyLlama 1.1B inference on Apple M2 GPU.

Analyzes arithmetic intensity, memory-bandwidth vs compute bounds, and
theoretical peak throughput for FP32, FP16/BF16, INT8, and INT4 precisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════
# 1. Model Specification
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int

    @property
    def weight_params_per_layer(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        return h*h + h*self.n_kv_heads*self.head_dim + h*self.n_kv_heads*self.head_dim + h*h + h*i + h*i + i*h

    @property
    def total_weight_params(self) -> int:
        return self.n_layers * self.weight_params_per_layer + self.hidden_size * self.vocab_size

    @property
    def norm_params_per_layer(self) -> int:
        return 2 * self.hidden_size  # attn_norm + ffn_norm

    @property
    def total_norm_params(self) -> int:
        return self.n_layers * self.norm_params_per_layer + self.hidden_size  # + output_norm

    @property
    def total_params(self) -> int:
        return self.total_weight_params + self.total_norm_params + self.vocab_size * self.hidden_size


TINYLLAMA = ModelSpec(
    name="TinyLlama-1.1B",
    hidden_size=2048,
    intermediate_size=5632,
    n_layers=22,
    n_heads=32,
    n_kv_heads=4,
    head_dim=64,
    vocab_size=32000,
    max_seq_len=2048,
)

# Per-layer matmul operations: (name, K, N)
# For decode: M=1, input activation is [1, K], weight is [K, N]
# For prefill: M=S
LAYER_MATMULS = [
    ("wq",     2048, 2048),
    ("wk",     2048,  256),   # n_kv_heads * head_dim = 4*64
    ("wv",     2048,  256),
    ("wo",     2048, 2048),
    ("w_gate", 2048, 5632),
    ("w_up",   2048, 5632),
    ("w_down", 5632, 2048),
]
LM_HEAD = ("lm_head", 2048, 32000)

# Element-wise ops per layer: (name, elements, ops_per_element, num_input_reads, has_weight)
ELEMWISE_OPS = [
    ("rms_norm_attn", 2048, 4, 1, True),   # sum_sq + div + mul_weight ≈ 4 ops/elem
    ("rms_norm_ffn",  2048, 4, 1, True),
    ("silu",          5632, 2, 1, False),    # exp + mul ≈ 2 ops/elem
    ("rope_q",        2048, 3, 1, False),    # freq + trig + mul ≈ 3 ops/elem
    ("rope_k",         256, 3, 1, False),
    ("add_attn",      2048, 1, 2, False),    # 1 add
    ("add_ffn",       2048, 1, 2, False),
    ("mul_gated",     5632, 1, 2, False),    # 1 mul
]


# ═══════════════════════════════════════════════════════════════════════
# 2. Hardware Specification
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GPUSpec:
    name: str
    gpu_cores: int
    max_clock_ghz: float
    alus_per_core: int
    memory_bandwidth_gbps: float  # GB/s

    @property
    def peak_f32_gflops(self) -> float:
        return self.gpu_cores * self.alus_per_core * self.max_clock_ghz * 2  # FMA


M2 = GPUSpec(
    name="Apple M2",
    gpu_cores=10,
    max_clock_ghz=1.4,
    alus_per_core=128,
    memory_bandwidth_gbps=100.0,
)


@dataclass
class PrecisionSpec:
    key: str
    label: str
    bytes_per_weight: float
    bytes_per_activation: float
    bytes_per_kv: float
    peak_gflops: float
    effective_gflops: float  # accounting for mixed-precision accumulation limits
    color: str


PRECISIONS = [
    PrecisionSpec("f32",  "FP32",               4.0, 4.0, 4.0,  3600,  3600, "#1f77b4"),
    PrecisionSpec("bf16", "BF16/FP16",          2.0, 2.0, 2.0,  7200,  7200, "#2ca02c"),
    PrecisionSpec("int8", "INT8 (FP16 act)",    1.0, 2.0, 2.0, 14400,  7200, "#d62728"),
    PrecisionSpec("int4", "INT4 (FP16 act)",    0.5, 2.0, 2.0, 28800,  7200, "#9467bd"),
]


# ═══════════════════════════════════════════════════════════════════════
# 3. Arithmetic Intensity Functions
# ═══════════════════════════════════════════════════════════════════════

def matmul_flops(M: int, K: int, N: int) -> int:
    return 2 * M * K * N

def matmul_bytes(M: int, K: int, N: int, bw: float, ba: float) -> float:
    """Total bytes accessed for matmul [M,K] x [K,N]."""
    return M * K * ba + K * N * bw + M * N * ba  # input + weight + output

def matmul_ai(M: int, K: int, N: int, bw: float, ba: float) -> float:
    """Arithmetic intensity (ops/byte) for matmul."""
    flops = matmul_flops(M, K, N)
    bytes_accessed = matmul_bytes(M, K, N, bw, ba)
    return flops / bytes_accessed if bytes_accessed > 0 else 0

def elemwise_ai(elements: int, ops_per_elem: int, ba: float,
                num_input_reads: int = 1, has_weight: bool = False) -> float:
    """Arithmetic intensity for element-wise operation."""
    flops = elements * ops_per_elem
    read_bytes = num_input_reads * elements * ba
    if has_weight:
        read_bytes += elements * ba
    write_bytes = elements * ba
    return flops / (read_bytes + write_bytes)

def sdpa_decode_ai(n_heads: int, n_kv_heads: int, ba_kv: float) -> float:
    """Arithmetic intensity for SDPA decode (constant, independent of position).

    FLOPs ≈ 2 * n_heads * P * head_dim (QK + AV for P KV positions)
    Bytes ≈ 2 * n_kv_heads * P * head_dim * ba_kv (K and V cache read)
    AI = n_heads / (n_kv_heads * ba_kv)
    """
    return n_heads / (n_kv_heads * ba_kv)


# ═══════════════════════════════════════════════════════════════════════
# 4. Decode Throughput Analysis
# ═══════════════════════════════════════════════════════════════════════

def total_weight_bytes_gb(model: ModelSpec, prec: PrecisionSpec) -> float:
    return model.total_weight_params * prec.bytes_per_weight / 1e9

def kv_bytes_per_position(model: ModelSpec, prec: PrecisionSpec) -> float:
    """Bytes added to KV cache traffic per decode position (all layers)."""
    return model.n_layers * 2 * model.n_kv_heads * model.head_dim * prec.bytes_per_kv

def activation_bytes_per_token(model: ModelSpec, prec: PrecisionSpec) -> float:
    """Rough estimate of intermediate activation bytes per decode token."""
    h, i = model.hidden_size, model.intermediate_size
    ba = prec.bytes_per_activation
    # Per layer: ~10 [1,h] tensors + 3 [1,i] tensors for intermediates
    return model.n_layers * (10 * h + 3 * i) * ba

def total_decode_bytes_gb(model: ModelSpec, prec: PrecisionSpec, position: int) -> float:
    """Total memory traffic per decode token at a given position."""
    weight_gb = total_weight_bytes_gb(model, prec)
    kv_gb = kv_bytes_per_position(model, prec) * position / 1e9
    act_gb = activation_bytes_per_token(model, prec) / 1e9
    return weight_gb + kv_gb + act_gb

def decode_peak_tps(gpu: GPUSpec, total_bytes_gb: float) -> float:
    """Theoretical peak decode tokens/second (memory-bound)."""
    return gpu.memory_bandwidth_gbps / total_bytes_gb

def decode_flops_per_token(model: ModelSpec) -> int:
    """Total matmul FLOPs per decode token (M=1)."""
    flops = 0
    for _, K, N in LAYER_MATMULS:
        flops += matmul_flops(1, K, N)
    flops = model.n_layers * flops
    _, K, N = LM_HEAD
    flops += matmul_flops(1, K, N)
    return flops


# ═══════════════════════════════════════════════════════════════════════
# 5. Prefill Crossover Analysis
# ═══════════════════════════════════════════════════════════════════════

def prefill_crossover_seq_len(K: int, N: int, bw: float, ba: float, ridge: float) -> Optional[float]:
    """Sequence length S where matmul [S,K]x[K,N] transitions from memory-bound to compute-bound.

    ridge = peak_compute / bandwidth (ops/byte)
    Returns None if the operation never becomes compute-bound.
    """
    denom = 2 * K * N - ridge * (K + N) * ba
    if denom <= 0:
        return None  # never compute-bound
    numer = ridge * K * N * bw
    return numer / denom


# ═══════════════════════════════════════════════════════════════════════
# 6. Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_roofline(model: ModelSpec, gpu: GPUSpec, precisions: List[PrecisionSpec]):
    """Generate roofline plot with operation markers."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    ai_range = np.logspace(-1.5, 3, 500)  # 0.03 to 1000 ops/byte
    bw = gpu.memory_bandwidth_gbps  # GB/s → GFLOPS when multiplied by ops/byte

    for prec in precisions:
        # Use effective peak (accounts for mixed-precision accumulation cap)
        peak = prec.effective_gflops
        ridge = peak / bw

        # Memory-bound ceiling: GFLOPS = BW * AI
        mem_ceiling = bw * ai_range
        # Compute-bound ceiling: GFLOPS = peak
        comp_ceiling = np.full_like(ai_range, peak)
        # Achievable = min of both
        achievable = np.minimum(mem_ceiling, comp_ceiling)

        # Draw roofline
        ax.loglog(ai_range, achievable, '-', color=prec.color, linewidth=2.5,
                  label=f'{prec.label} (ridge={ridge:.0f})')

        # Ridge point marker
        ax.plot(ridge, peak, 'o', color=prec.color, markersize=10, zorder=5)

        # ── Decode matmul markers (M=1) ──
        for name, K, N in LAYER_MATMULS:
            ai = matmul_ai(1, K, N, prec.bytes_per_weight, prec.bytes_per_activation)
            gflops = min(bw * ai, peak)
            ax.plot(ai, gflops, 's', color=prec.color, markersize=5, alpha=0.6)

        # lm_head
        _, K, N = LM_HEAD
        ai = matmul_ai(1, K, N, prec.bytes_per_weight, prec.bytes_per_activation)
        gflops = min(bw * ai, peak)
        ax.plot(ai, gflops, 'D', color=prec.color, markersize=7, alpha=0.8)

        # ── Prefill matmul markers (M=128) ──
        for name, K, N in LAYER_MATMULS:
            ai = matmul_ai(128, K, N, prec.bytes_per_weight, prec.bytes_per_activation)
            gflops = min(bw * ai, peak)
            ax.plot(ai, gflops, 'o', color=prec.color, markersize=5, alpha=0.4)

        # ── SDPA decode marker ──
        ai_sdpa = sdpa_decode_ai(model.n_heads, model.n_kv_heads, prec.bytes_per_kv)
        gflops_sdpa = min(bw * ai_sdpa, peak)
        ax.plot(ai_sdpa, gflops_sdpa, '*', color=prec.color, markersize=12, alpha=0.8)

    # ── Element-wise op markers (use BF16 activations for all quantized precisions) ──
    for elem_name, elements, ops, reads, has_w in ELEMWISE_OPS:
        # Use BF16 activation size (2 bytes) — element-wise ops don't benefit from weight quantization
        ai = elemwise_ai(elements, ops, 2.0, reads, has_w)
        gflops = min(bw * ai, 7200)  # cap at BF16 peak
        ax.plot(ai, gflops, '^', color='#666666', markersize=5, alpha=0.5)

    # ── Annotate operation regions ──
    ax.axvspan(0.03, 2, alpha=0.03, color='red', zorder=0)
    ax.axvspan(2, 50, alpha=0.03, color='yellow', zorder=0)
    ax.axvspan(50, 1000, alpha=0.03, color='green', zorder=0)

    ax.text(0.15, 1.5, "Memory\nBound", fontsize=9, color='#999', ha='center', style='italic')
    ax.text(8, 1.5, "Transition", fontsize=9, color='#999', ha='center', style='italic')
    ax.text(200, 1.5, "Compute\nBound", fontsize=9, color='#999', ha='center', style='italic')

    # ── Legend annotations ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888', markersize=7, label='Decode matmul (M=1)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#888', markersize=7, label='lm_head (M=1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888', markersize=7, alpha=0.5, label='Prefill matmul (M=128)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#888', markersize=10, label='SDPA decode'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#888', markersize=6, label='Element-wise (BF16 act)'),
    ]

    ax.set_xlabel("Arithmetic Intensity (ops/byte)", fontsize=12)
    ax.set_ylabel("Achievable Performance (GFLOPS)", fontsize=12)
    ax.set_title(f"Roofline Analysis: {model.name} on {gpu.name} GPU", fontsize=14)
    ax.set_xlim(0.03, 800)
    ax.set_ylim(1, 15000)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    # Add precision roofline labels in legend
    for prec in precisions:
        ax.plot([], [], '-', color=prec.color, linewidth=2.5,
                label=f'{prec.label} roofline')

    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig('scripts/roofline_tinyllama_m2.png', dpi=150)
    print("Saved: scripts/roofline_tinyllama_m2.png")


def plot_decode_throughput(model: ModelSpec, gpu: GPUSpec, precisions: List[PrecisionSpec]):
    """Plot decode tok/s vs sequence position."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    positions = [1, 32, 64, 128, 256, 512, 1024, 2048]

    for prec in precisions:
        tps = []
        for P in positions:
            total_gb = total_decode_bytes_gb(model, prec, P)
            tps.append(decode_peak_tps(gpu, total_gb))
        ax.plot(positions, tps, '-o', color=prec.color, linewidth=2, markersize=6,
                label=prec.label)

    ax.set_xlabel("Sequence Position", fontsize=12)
    ax.set_ylabel("Decode Tokens/sec (theoretical peak)", fontsize=12)
    ax.set_title(f"{model.name} Decode Throughput vs Position ({gpu.name})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('scripts/roofline_decode_throughput.png', dpi=150)
    print("Saved: scripts/roofline_decode_throughput.png")


def plot_prefill_ai(model: ModelSpec, precisions: List[PrecisionSpec]):
    """Plot arithmetic intensity vs sequence length for key matmuls."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    seq_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    # Left: wq (representative large matmul)
    ax = axes[0]
    K, N = 2048, 2048
    for prec in precisions:
        ais = [matmul_ai(S, K, N, prec.bytes_per_weight, prec.bytes_per_activation) for S in seq_lengths]
        ax.plot(seq_lengths, ais, '-o', color=prec.color, linewidth=2, markersize=4, label=prec.label)
        # Ridge point
        ridge = prec.effective_gflops / 100.0
        ax.axhline(y=ridge, color=prec.color, linestyle='--', alpha=0.3)
    ax.set_xlabel("Prefill Sequence Length S")
    ax.set_ylabel("Arithmetic Intensity (ops/byte)")
    ax.set_title(f"Matmul [{K},{N}] (wq/wo) AI vs S")
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: lm_head (large N)
    ax = axes[1]
    K, N = 2048, 32000
    for prec in precisions:
        ais = [matmul_ai(S, K, N, prec.bytes_per_weight, prec.bytes_per_activation) for S in seq_lengths]
        ax.plot(seq_lengths, ais, '-o', color=prec.color, linewidth=2, markersize=4, label=prec.label)
        ridge = prec.effective_gflops / 100.0
        ax.axhline(y=ridge, color=prec.color, linestyle='--', alpha=0.3)
    ax.set_xlabel("Prefill Sequence Length S")
    ax.set_ylabel("Arithmetic Intensity (ops/byte)")
    ax.set_title(f"Matmul [{K},{N}] (lm_head) AI vs S")
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model.name} Prefill AI vs Sequence Length", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('scripts/roofline_prefill_ai.png', dpi=150)
    print("Saved: scripts/roofline_prefill_ai.png")


# ═══════════════════════════════════════════════════════════════════════
# 7. Summary Tables
# ═══════════════════════════════════════════════════════════════════════

def print_summary(model: ModelSpec, gpu: GPUSpec, precisions: List[PrecisionSpec]):
    M = model
    G = gpu

    print("=" * 72)
    print(f"  ROOFLINE ANALYSIS: {M.name} on {G.name} GPU")
    print("=" * 72)

    # ── Hardware ──
    print(f"\n{'─' * 72}")
    print("  HARDWARE")
    print(f"{'─' * 72}")
    print(f"  GPU Cores:          {G.gpu_cores}")
    print(f"  Max Clock:          {G.max_clock_ghz} GHz")
    print(f"  ALU/core:           {G.alus_per_core}")
    print(f"  Memory Bandwidth:   {G.memory_bandwidth_gbps:.0f} GB/s")
    print(f"  Peak F32:           {G.peak_f32_gflops:.0f} GFLOPS")

    # ── Model ──
    print(f"\n{'─' * 72}")
    print("  MODEL")
    print(f"{'─' * 72}")
    print(f"  Hidden size:        {M.hidden_size}")
    print(f"  Intermediate:       {M.intermediate_size}")
    print(f"  Layers:             {M.n_layers}")
    print(f"  Heads (Q/KV):       {M.n_heads}/{M.n_kv_heads}")
    print(f"  Head dim:           {M.head_dim}")
    print(f"  Vocab:              {M.vocab_size}")
    print(f"  Total params:       {M.total_params/1e9:.2f}B")
    print(f"  Weight params:      {M.total_weight_params/1e9:.2f}B")
    print(f"  Decode FLOPs/token: {decode_flops_per_token(M)/1e6:.1f}M")

    # ── Ridge points ──
    print(f"\n{'─' * 72}")
    print("  RIDGE POINTS (ops/byte where compute-bound = memory-bound)")
    print(f"{'─' * 72}")
    print(f"  {'Precision':<22} {'Peak (GFLOPS)':>14} {'Effective':>14} {'Ridge (theo)':>14} {'Ridge (eff)':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for p in precisions:
        ridge_theo = p.peak_gflops / G.memory_bandwidth_gbps
        ridge_eff = p.effective_gflops / G.memory_bandwidth_gbps
        print(f"  {p.label:<22} {p.peak_gflops:>14,.0f} {p.effective_gflops:>14,.0f} {ridge_theo:>14.1f} {ridge_eff:>14.1f}")

    # ── Decode matmul AI (M=1) ──
    print(f"\n{'─' * 72}")
    print("  DECODE MATMUL ARITHMETIC INTENSITY (M=1)")
    print(f"{'─' * 72}")
    header = f"  {'Operation':<12} {'FLOPs':>10}"
    for p in precisions:
        header += f" {p.label:>12}"
    print(header)
    print(f"  {'-'*12} {'-'*10}" + " " + " ".join(["-"*12]*len(precisions)))

    all_matmuls = [(n, K, N) for n, K, N in LAYER_MATMULS] + [LM_HEAD]
    for name, K, N in all_matmuls:
        flops = matmul_flops(1, K, N)
        row = f"  {name:<12} {flops/1e6:>9.1f}M"
        for p in precisions:
            ai = matmul_ai(1, K, N, p.bytes_per_weight, p.bytes_per_activation)
            row += f" {ai:>12.2f}"
        print(row)

    # ── Element-wise AI ──
    print(f"\n{'─' * 72}")
    print("  ELEMENT-WISE OP ARITHMETIC INTENSITY (BF16 activations)")
    print(f"{'─' * 72}")
    print(f"  {'Operation':<16} {'Elements':>10} {'ops/elem':>10} {'AI (ops/B)':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*12}")
    for name, elems, ops, reads, has_w in ELEMWISE_OPS:
        ai = elemwise_ai(elems, ops, 2.0, reads, has_w)
        print(f"  {name:<16} {elems:>10} {ops:>10} {ai:>12.2f}")

    # ── SDPA decode AI ──
    print(f"\n{'─' * 72}")
    print("  SDPA DECODE ARITHMETIC INTENSITY (constant w.r.t. position)")
    print(f"{'─' * 72}")
    for p in precisions:
        ai = sdpa_decode_ai(M.n_heads, M.n_kv_heads, p.bytes_per_kv)
        ridge = p.effective_gflops / G.memory_bandwidth_gbps
        bound = "compute" if ai > ridge else "memory"
        print(f"  {p.label:<22} AI={ai:.2f} ops/byte  ridge={ridge:.0f}  → {bound}-bound")

    # ── Decode peak throughput ──
    print(f"\n{'─' * 72}")
    print("  PEAK DECODE THROUGHPUT (tok/s)")
    print(f"{'─' * 72}")
    print(f"  {'Precision':<22} {'Weight GB':>10} {'@P=1':>8} {'@P=128':>8} {'@P=512':>8} {'@P=2048':>9}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for p in precisions:
        wgb = total_weight_bytes_gb(M, p)
        row = f"  {p.label:<22} {wgb:>10.3f}"
        for P in [1, 128, 512, 2048]:
            tgb = total_decode_bytes_gb(M, p, P)
            tps = decode_peak_tps(G, tgb)
            row += f" {tps:>8.1f}"
        print(row)

    # ── KV cache overhead ──
    print(f"\n{'─' * 72}")
    print("  KV CACHE MEMORY IMPACT")
    print(f"{'─' * 72}")
    for p in precisions:
        kv_per_pos = kv_bytes_per_position(M, p)
        wgb = total_weight_bytes_gb(M, p)
        print(f"  {p.label:<22} {kv_per_pos/1024:.1f} KB/pos", end="")
        for P in [128, 512, 2048]:
            kv_total = kv_per_pos * P / 1e9
            pct = kv_total / wgb * 100
            print(f"  P={P}: {kv_total*1024:.1f}MB ({pct:.1f}%)", end="")
        print()

    # ── Prefill crossover ──
    print(f"\n{'─' * 72}")
    print("  PREFILL COMPUTE-BOUND CROSSOVER (sequence length S)")
    print(f"{'─' * 72}")
    header = f"  {'Operation':<12} {'K':>6} {'N':>6}"
    for p in precisions:
        header += f" {p.label + ' (theo)':>14} {p.label + ' (eff)':>14}"
    print(header)
    print(f"  {'-'*12} {'-'*6} {'-'*6}" + " ".join([" " + "-"*14]*len(precisions)))

    for name, K, N in all_matmuls:
        row = f"  {name:<12} {K:>6} {N:>6}"
        for p in precisions:
            ridge_theo = p.peak_gflops / G.memory_bandwidth_gbps
            ridge_eff = p.effective_gflops / G.memory_bandwidth_gbps
            s_theo = prefill_crossover_seq_len(K, N, p.bytes_per_weight, p.bytes_per_activation, ridge_theo)
            s_eff = prefill_crossover_seq_len(K, N, p.bytes_per_weight, p.bytes_per_activation, ridge_eff)
            s_theo_str = f"{s_theo:.1f}" if s_theo else "never"
            s_eff_str = f"{s_eff:.1f}" if s_eff else "never"
            row += f" {s_theo_str:>14} {s_eff_str:>14}"
        print(row)

    # ── Bottleneck summary ──
    print(f"\n{'─' * 72}")
    print("  BOTTLENECK SUMMARY")
    print(f"{'─' * 72}")
    print("  Decode (M=1):    ALL precisions → MEMORY-BOUND")
    print("                    AI < 2 ops/byte for F32, < 5 ops/byte for INT4")
    print("                    Throughput = bandwidth / weight_bytes")
    print()
    print("  Prefill (M=S):   Transition to compute-bound depends on S and precision:")
    for p in precisions:
        ridge = p.effective_gflops / G.memory_bandwidth_gbps
        # Use wq as representative
        s_c = prefill_crossover_seq_len(2048, 2048, p.bytes_per_weight, p.bytes_per_activation, ridge)
        s_str = f"S ≈ {s_c:.0f}" if s_c else "never"
        print(f"    {p.label:<22} ridge={ridge:.0f} ops/B, crossover at {s_str}")
    print()
    print("  SDPA Decode:     MEMORY-BOUND for all precisions")
    print("                    AI = n_heads / (n_kv_heads × bytes_per_kv)")
    print(f"                    F32: AI={M.n_heads/(M.n_kv_heads*4):.1f}, BF16/INT8/INT4: AI={M.n_heads/(M.n_kv_heads*2):.1f}")
    print()
    print("  Element-wise:    MEMORY-BOUND for all precisions")
    print("                    AI ≈ 0.3–1.3 ops/byte")
    print()
    print("  Key insight:     Decode is BANDWIDTH-LIMITED. Batching kernel launches")
    print("                    (single command buffer) reduces latency overhead but")
    print("                    does NOT change the fundamental bandwidth ceiling.")
    print("                    To increase tok/s: reduce weight bytes (quantize)")
    print("                    or increase bandwidth (not possible on M2).")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    model = TINYLLAMA
    gpu = M2
    precisions = PRECISIONS

    print_summary(model, gpu, precisions)
    print("\nGenerating plots...")
    plot_roofline(model, gpu, precisions)
    plot_decode_throughput(model, gpu, precisions)
    plot_prefill_ai(model, gpu, precisions)
    print("\nDone.")


if __name__ == "__main__":
    main()
