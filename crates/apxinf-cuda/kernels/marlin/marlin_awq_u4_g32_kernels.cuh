// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vLLM v0.27.1 Marlin generate_kernels.py.
// clang-format off
#pragma once


namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);

}  // namespace MARLIN_NAMESPACE_NAME
