// Immutable candidate storage shared by operator adapters.
//
// Candidate must expose provider_id, implementation_id, and
// implementation_version. find() restores a candidate only when that complete
// identity matches; begin()/end() expose candidates to filtering or tuning.
// The operator adapter remains responsible for selecting the semantic-specific
// registry and for support checks, configurations, provider state, and fallback
// policy.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace apxinf::framework {

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
