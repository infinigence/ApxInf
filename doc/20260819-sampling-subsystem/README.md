# Sampling Subsystem

Date: 2026-08-19
Status: Implemented and validated on NVIDIA Thor

This folder is the canonical documentation for ApxInf's model-neutral sampling
and random-number subsystem. Sampling is a backend concern shared by model
families; it is deliberately separate from the LLM/VLM modality interface.

The documents are split by purpose:

- [Design](design.md) defines the public contracts, ownership, semantics, and
  LLM/VLM/VLA boundaries.
- [Implementation](implementation.md) describes the CPU and CUDA paths, source
  layout, memory behavior, and integration points.
- [Evaluation](evaluation.md) records the current correctness and real-model
  evidence, structural performance improvements, and remaining measurements.
- [Tests](test.md) contains the test matrix, tolerances, commands, and
  regression checklist.

## Scope

The subsystem currently provides:

1. categorical next-token selection for autoregressive LLMs and VLMs;
2. greedy, temperature, top-k, top-p, repetition, frequency, and presence
   policies;
3. deterministic counter-based random streams;
4. optional selected-token log-probabilities;
5. standard-normal generation into a stable backend tensor for continuous VLA
   initial latents; and
6. CPU and CUDA implementations behind the same model-neutral traits.

PI0.5 remains a continuous flow model and therefore uses only the normal
generator. It is not represented as a categorical action head. If a future VLA
actually emits action tokens, it can use the existing categorical sampler
without adding speculative action-head variants to today's API.
