#pragma once

// Minimal compile-time ScalarType compatibility surface for the vendored
// Marlin templates. ApxInf instantiates only the BF16 x asymmetric-U4 -> BF16
// path, but the template parser references the neighboring type constants in
// discarded constexpr branches.

#include <cstdint>

namespace vllm {

class ScalarType {
 public:
  using Id = int64_t;

  constexpr ScalarType(Id id, int bits) : id_(id), bits_(bits) {}
  constexpr Id id() const { return id_; }
  constexpr int64_t size_bits() const { return bits_; }
  constexpr bool operator==(ScalarType other) const { return id_ == other.id_; }
  constexpr bool operator!=(ScalarType other) const { return id_ != other.id_; }

  static constexpr ScalarType from_id(Id id) {
    return ScalarType(id, id == 1 || id == 2 || id == 3
                              ? 16
                              : id == 4 || id == 5 || id == 6 || id == 7 ||
                                        id == 11
                                    ? 8
                                    : 4);
  }

 private:
  Id id_;
  int bits_;
};

using ScalarTypeId = ScalarType::Id;
inline constexpr ScalarType kFloat16(1, 16);
inline constexpr ScalarType kBFloat16(2, 16);
inline constexpr ScalarType kFE8M7(3, 16);
inline constexpr ScalarType kFE4M3fn(4, 8);
inline constexpr ScalarType kFE8M0fnu(5, 8);
inline constexpr ScalarType kS8(6, 8);
inline constexpr ScalarType kU8(7, 8);
inline constexpr ScalarType kU4(8, 4);
inline constexpr ScalarType kU4B8(9, 4);
inline constexpr ScalarType kFE2M1f(10, 4);
inline constexpr ScalarType kU8B128(11, 8);
inline constexpr ScalarType kS4(12, 4);

}  // namespace vllm
