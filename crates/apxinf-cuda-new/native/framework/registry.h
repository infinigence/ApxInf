#pragma once

#include <initializer_list>
#include <vector>

namespace apxinf::framework {

// The registry owns only immutable candidate descriptors. Spec filtering and
// provider state remain operator-specific.
template <class Candidate>
class Registry {
 public:
  Registry(std::initializer_list<Candidate> candidates)
      : candidates_(candidates) {}

  const Candidate* find(uint32_t provider_id, uint32_t implementation_id,
                        uint32_t implementation_version) const {
    for (const auto& candidate : candidates_) {
      if (candidate.provider_id == provider_id &&
          candidate.implementation_id == implementation_id &&
          candidate.implementation_version == implementation_version) {
        return &candidate;
      }
    }
    return nullptr;
  }

  auto begin() const { return candidates_.begin(); }
  auto end() const { return candidates_.end(); }

 private:
  std::vector<Candidate> candidates_;
};

}  // namespace apxinf::framework
