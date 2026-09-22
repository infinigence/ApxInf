//! Keeps the model-facing CUDA operator catalog complete.
//!
//! The test intentionally checks presence rather than prose correctness. A
//! reviewer remains responsible for verifying contracts, limitations, and
//! status whenever an operator changes.

use std::collections::BTreeSet;

use crate::ops::{attention_contracts, contracts as gemm_contracts};

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
    // This is the catalog of operator families, not a closed GEMM/Attention
    // assumption. A new family must expose semantic doc IDs and join this
    // iterator so its public L3 surface is covered by the same check.
    let expected_ids: Vec<_> = gemm_contracts::Semantic::ALL
        .iter()
        .copied()
        .map(gemm_contracts::Semantic::doc_id)
        .chain(
            attention_contracts::Semantic::ALL
                .iter()
                .copied()
                .map(attention_contracts::Semantic::doc_id),
        )
        .collect();
    let expected: BTreeSet<_> = expected_ids.iter().copied().collect();

    assert_eq!(
        documented.len(),
        documented_set.len(),
        "cuda-operator.md contains duplicate L3 operator markers"
    );
    assert_eq!(
        expected_ids.len(),
        expected.len(),
        "public L3 semantics contain duplicate documentation identifiers"
    );
    assert_eq!(
        documented_set, expected,
        "cuda-operator.md and the public L3 semantic list differ"
    );
}
