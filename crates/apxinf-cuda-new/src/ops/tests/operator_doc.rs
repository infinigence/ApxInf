//! Keeps the model-facing CUDA operator catalog complete.
//!
//! The test intentionally checks presence rather than prose correctness. A
//! reviewer remains responsible for verifying contracts, limitations, and
//! status whenever an operator changes.

use std::collections::BTreeSet;

use crate::ops::contracts::Semantic;

const CATALOG: &str = include_str!("../../../cuda-operator.md");
const MARKER_PREFIX: &str = "<!-- l3-operator:";
const MARKER_SUFFIX: &str = " -->";

fn documented_operator_ids() -> Vec<&'static str> {
    CATALOG
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix(MARKER_PREFIX)
                .and_then(|value| value.strip_suffix(MARKER_SUFFIX))
        })
        .collect()
}

#[test]
fn every_l3_operator_appears_once_in_cuda_operator_doc() {
    let documented = documented_operator_ids();
    let documented_set: BTreeSet<_> = documented.iter().copied().collect();
    let expected: BTreeSet<_> = Semantic::ALL
        .iter()
        .copied()
        .map(Semantic::doc_id)
        .collect();

    assert_eq!(
        documented.len(),
        documented_set.len(),
        "cuda-operator.md contains duplicate L3 operator markers"
    );
    assert_eq!(
        documented_set, expected,
        "cuda-operator.md and the public L3 semantic list differ"
    );
}
