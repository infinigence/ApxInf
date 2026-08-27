// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Minimal standalone subset of vLLM v0.27.1 core/scalar_type.hpp required by
// the vendored Marlin BF16/U4/BF16/BF16 specialization. No torch dependency.
#pragma once

#include <cstdint>
#include <type_traits>

namespace vllm {

class ScalarType {
 public:
  using Id = int64_t;

  constexpr ScalarType(Id id, int bits) : id_(id), bits_(bits) {}
  constexpr Id id() const { return id_; }
  constexpr int size_bits() const { return bits_; }
  static constexpr ScalarType from_id(Id id) {
    return id == 1  ? ScalarType(1, 16)
         : id == 2  ? ScalarType(2, 4)
         : id == 3  ? ScalarType(3, 16)
         : id == 5  ? ScalarType(5, 8)
         : id == 6  ? ScalarType(6, 8)
         : id == 7  ? ScalarType(7, 4)
         : id == 8  ? ScalarType(8, 8)
         : id == 9  ? ScalarType(9, 8)
         : id == 10 ? ScalarType(10, 4)
         : id == 11 ? ScalarType(11, 8)
         : id == 12 ? ScalarType(12, 4)
                    : ScalarType(0, 0);
  }
  constexpr bool operator==(const ScalarType& other) const {
    return id_ == other.id_;
  }
  constexpr bool operator!=(const ScalarType& other) const {
    return !(*this == other);
  }

 private:
  Id id_;
  int bits_;
};

using ScalarTypeId = ScalarType::Id;

// IDs are internal compile-time template identifiers. Only kBFloat16 and kU4
// are instantiated by ApxInf; the remaining names keep discarded upstream
// `if constexpr` branches well-formed without adding their implementations.
inline constexpr ScalarType kBFloat16{1, 16};
inline constexpr ScalarType kU4{2, 4};
inline constexpr ScalarType kFloat16{3, 16};
inline constexpr ScalarType kFE4M3fn{5, 8};
inline constexpr ScalarType kS8{6, 8};
inline constexpr ScalarType kU4B8{7, 4};
inline constexpr ScalarType kU8{8, 8};
inline constexpr ScalarType kU8B128{9, 8};
inline constexpr ScalarType kFE2M1f{10, 4};
inline constexpr ScalarType kFE8M0fnu{11, 8};
inline constexpr ScalarType kS4{12, 4};

}  // namespace vllm
