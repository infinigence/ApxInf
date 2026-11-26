//! Bounded cache of compiled whole-request profiles.

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use apxinf_core::{Error, Result};

use crate::LoadOptions;

/// Retained profiles per request kind: an entry count, and a budget on the sum
/// of the retained profiles' prompt lengths. The defaults bound a long run over
/// many distinct prompt lengths; a caller with few lengths can raise them to
/// avoid retracing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileLimits {
    pub profiles: NonZeroUsize,
    pub tokens: NonZeroUsize,
}

impl Default for ProfileLimits {
    fn default() -> Self {
        Self {
            profiles: NonZeroUsize::new(DEFAULT_PROFILE_LIMIT).unwrap(),
            tokens: NonZeroUsize::new(DEFAULT_PROFILE_TOKEN_BUDGET).unwrap(),
        }
    }
}

const DEFAULT_PROFILE_LIMIT: usize = 12;
const DEFAULT_PROFILE_TOKEN_BUDGET: usize = 30_000;

impl ProfileLimits {
    /// The text-profile limits from the shared d3 load options.
    pub fn text(options: &LoadOptions) -> Result<Self> {
        let defaults = Self::default();
        let positive = |value: Option<usize>, default: NonZeroUsize, name: &str| match value {
            None => Ok(default),
            Some(value) => NonZeroUsize::new(value)
                .ok_or_else(|| Error::Other(format!("{name} must be positive"))),
        };
        Ok(Self {
            profiles: positive(
                options.d3_text_profile_limit,
                defaults.profiles,
                "d3 text profile cache limit",
            )?,
            tokens: positive(
                options.d3_text_profile_token_budget,
                defaults.tokens,
                "d3 text profile token budget",
            )?,
        })
    }
}

struct Entry<K, V> {
    key: K,
    tokens: usize,
    value: V,
}

/// Least recently used first. Dropping an entry drops its compiled callable,
/// which releases that callable's traced graph. One request longer than the
/// token budget is still admitted, alone.
pub struct ProfileCache<K, V> {
    limits: ProfileLimits,
    entries: VecDeque<Entry<K, V>>,
}

impl<K: PartialEq + Copy, V> ProfileCache<K, V> {
    pub fn new(limits: ProfileLimits) -> Self {
        Self {
            limits,
            entries: VecDeque::new(),
        }
    }

    /// The profile for `key`, created by `make` on a miss after evicting
    /// enough older profiles for it to fit.
    pub fn get_or_insert_with(
        &mut self,
        key: K,
        tokens: usize,
        make: impl FnOnce() -> Result<V>,
    ) -> Result<&V> {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            let entry = self.entries.remove(index).unwrap();
            self.entries.push_back(entry);
        } else {
            while !self.entries.is_empty()
                && (self.entries.len() >= self.limits.profiles.get()
                    || self.retained_tokens().saturating_add(tokens) > self.limits.tokens.get())
            {
                self.entries.pop_front();
            }
            let value = make()?;
            self.entries.push_back(Entry { key, tokens, value });
        }
        Ok(&self.entries.back().unwrap().value)
    }

    /// Retained keys, least recently used first.
    pub fn keys(&self) -> Vec<K> {
        self.entries.iter().map(|entry| entry.key).collect()
    }

    fn retained_tokens(&self) -> usize {
        self.entries
            .iter()
            .fold(0, |sum, entry| sum.saturating_add(entry.tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn limits(profiles: usize, tokens: usize) -> ProfileLimits {
        ProfileLimits {
            profiles: NonZeroUsize::new(profiles).unwrap(),
            tokens: NonZeroUsize::new(tokens).unwrap(),
        }
    }

    /// Records its key when dropped, so a test can see which profiles were
    /// released rather than only which keys are listed.
    struct Profile {
        key: usize,
        dropped: Rc<RefCell<Vec<usize>>>,
    }

    impl Drop for Profile {
        fn drop(&mut self) {
            self.dropped.borrow_mut().push(self.key);
        }
    }

    struct Harness {
        cache: ProfileCache<usize, Profile>,
        dropped: Rc<RefCell<Vec<usize>>>,
        made: Vec<usize>,
    }

    impl Harness {
        fn new(limits: ProfileLimits) -> Self {
            Self {
                cache: ProfileCache::new(limits),
                dropped: Rc::default(),
                made: Vec::new(),
            }
        }

        fn request(&mut self, length: usize) {
            let dropped = self.dropped.clone();
            let made = &mut self.made;
            let profile = self
                .cache
                .get_or_insert_with(length, length, || {
                    made.push(length);
                    Ok(Profile {
                        key: length,
                        dropped,
                    })
                })
                .unwrap();
            assert_eq!(profile.key, length);
        }
    }

    #[test]
    fn the_profile_limit_releases_the_least_recently_used_profile() {
        let mut harness = Harness::new(limits(2, 1_000));
        for length in [10, 20, 10, 30] {
            harness.request(length);
        }
        assert_eq!(harness.cache.keys(), [10, 30]);
        assert_eq!(*harness.dropped.borrow(), [20]);
        assert_eq!(harness.made, [10, 20, 30]);

        harness.request(20);
        assert_eq!(
            harness.made,
            [10, 20, 30, 20],
            "an evicted profile is rebuilt"
        );
        assert_eq!(harness.cache.keys(), [30, 20]);
    }

    #[test]
    fn the_token_budget_bounds_the_sum_of_retained_lengths() {
        let mut harness = Harness::new(limits(100, 100));
        for length in [40, 50, 30] {
            harness.request(length);
        }
        assert_eq!(harness.cache.keys(), [50, 30]);
        assert_eq!(*harness.dropped.borrow(), [40]);
    }

    #[test]
    fn a_request_over_the_token_budget_runs_alone() {
        let mut harness = Harness::new(limits(100, 100));
        for length in [40, 50, 150] {
            harness.request(length);
        }
        assert_eq!(harness.cache.keys(), [150]);
        assert_eq!(*harness.dropped.borrow(), [40, 50]);
    }

    #[test]
    fn a_failed_build_retains_no_profile_for_its_key() {
        let mut cache = ProfileCache::<usize, ()>::new(limits(2, 1_000));
        assert!(cache
            .get_or_insert_with(7, 7, || Err(Error::Other("trace failed".into())))
            .is_err());
        assert!(cache.keys().is_empty());
        let mut built = false;
        cache
            .get_or_insert_with(7, 7, || {
                built = true;
                Ok(())
            })
            .unwrap();
        assert!(built, "a failed key is built again on the next request");
    }

    #[test]
    fn load_options_set_the_text_limits_and_reject_zero() {
        assert_eq!(
            ProfileLimits::text(&LoadOptions::default()).unwrap(),
            ProfileLimits::default()
        );
        let options = LoadOptions {
            d3_text_profile_limit: Some(3),
            d3_text_profile_token_budget: Some(4_096),
            ..LoadOptions::default()
        };
        assert_eq!(ProfileLimits::text(&options).unwrap(), limits(3, 4_096));
        for options in [
            LoadOptions {
                d3_text_profile_limit: Some(0),
                ..LoadOptions::default()
            },
            LoadOptions {
                d3_text_profile_token_budget: Some(0),
                ..LoadOptions::default()
            },
        ] {
            assert!(ProfileLimits::text(&options).is_err(), "{options:?}");
        }
    }
}
