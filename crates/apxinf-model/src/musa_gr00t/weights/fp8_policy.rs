use apxinf_musa::kernels::gemm::Fp8Layout;

#[derive(Clone, Copy, Debug)]
pub(super) struct Rule {
    pub min_rows: usize,
    pub max_rows: usize,
    pub cols: usize,
    pub inner: usize,
    pub layout: Fp8Layout,
}

pub(super) const RULES: &[Rule] = &[
    rule(40, 40, 1536, 3072, Fp8Layout::Nt),
    rule(41, 41, 1536, 6144, Fp8Layout::Nt),
    rule(41, 41, 4608, 1536, Fp8Layout::Nn),
    rule(41, 41, 6144, 1536, Fp8Layout::Nn),
    rule(128, 128, 1536, 2048, Fp8Layout::Nn),
    rule(128, 128, 2048, 4096, Fp8Layout::Nn),
    rule(128, 128, 4096, 4096, Fp8Layout::Nt),
    rule(129, 256, 1024, 2048, Fp8Layout::Nt),
    rule(129, 256, 2048, 2048, Fp8Layout::Nt),
    rule(129, 256, 2048, 6144, Fp8Layout::Nt),
    rule(129, 256, 2048, 8192, Fp8Layout::Nt),
    rule(129, 256, 6144, 2048, Fp8Layout::Nn),
    rule(129, 256, 8192, 2048, Fp8Layout::Nt),
    rule(512, 512, 1024, 1536, Fp8Layout::Nt),
    rule(512, 512, 1024, 4096, Fp8Layout::Nt),
    rule(512, 512, 3072, 1024, Fp8Layout::Nt),
    rule(512, 512, 4096, 1024, Fp8Layout::Nt),
];

const fn rule(
    min_rows: usize,
    max_rows: usize,
    cols: usize,
    inner: usize,
    layout: Fp8Layout,
) -> Rule {
    Rule {
        min_rows,
        max_rows,
        cols,
        inner,
        layout,
    }
}

pub(super) fn candidate(cols: usize, inner: usize) -> Option<Rule> {
    RULES
        .iter()
        .find(|rule| rule.cols == cols && rule.inner == inner)
        .copied()
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Policy {
    standalone: Option<Rule>,
    fused: Rule,
}

impl Policy {
    pub fn layout(self) -> Fp8Layout {
        self.fused.layout
    }

    pub fn quantizes_input(self, rows: usize, inner: usize) -> bool {
        self.standalone
            .is_some_and(|rule| rule.inner == inner && rule.matches_rows(rows))
    }

    pub fn accepts_fp8(self, rows: usize, inner: usize) -> bool {
        self.quantizes_input(rows, inner)
            || (self.fused.inner == inner && self.fused.matches_rows(rows))
    }
}

pub(super) fn for_consumer(name: &str, cols: usize, inner: usize) -> Option<Policy> {
    let standalone = candidate(cols, inner);
    let cross_query = name
        .strip_prefix("action_head.dit_blocks.")
        .and_then(|suffix| suffix.strip_suffix(".attention.query"))
        .and_then(|index| index.parse::<usize>().ok())
        .is_some_and(|index| index % 2 == 0);
    let fused = match (name, cols, inner) {
        (_, 1536, 1536) if cross_query => Some(rule(41, 41, cols, inner, Fp8Layout::Nt)),
        ("action_head.output_projection", 1024, 1536) => {
            Some(rule(41, 41, cols, inner, Fp8Layout::Nt))
        }
        _ => None,
    };
    Some(Policy {
        standalone,
        fused: fused.or(standalone)?,
    })
}

impl Rule {
    pub fn matches_rows(self, rows: usize) -> bool {
        (self.min_rows..=self.max_rows).contains(&rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_rows_cover_dynamic_prompt_without_matching_boundaries() {
        for rule in RULES.iter().filter(|rule| rule.min_rows == 129) {
            assert!(!rule.matches_rows(128));
            for rows in 129..=256 {
                assert!(rule.matches_rows(rows));
            }
            assert!(!rule.matches_rows(257));
        }
    }

    #[test]
    fn rules_are_unique_and_exclude_low_gain_shapes() {
        assert_eq!(RULES.len(), 17);
        for (index, rule) in RULES.iter().enumerate() {
            assert!(RULES[index + 1..]
                .iter()
                .all(|other| (rule.cols, rule.inner) != (other.cols, other.inner)));
        }
        for (rows, cols, inner) in [(41, 1536, 1536), (17, 1536, 2048), (512, 1024, 1024)] {
            assert!(!candidate(cols, inner).is_some_and(|rule| rule.matches_rows(rows)));
        }
    }

    #[test]
    fn adaln_query_requires_the_named_cross_consumer_and_fp8_input() {
        for index in 0..32 {
            let name = format!("action_head.dit_blocks.{index}.attention.query");
            let policy = for_consumer(&name, 1536, 1536);
            assert_eq!(policy.is_some(), index % 2 == 0);
            if let Some(policy) = policy {
                assert_eq!(policy.layout(), Fp8Layout::Nt);
                assert!(policy.accepts_fp8(41, 1536));
                assert!(!policy.quantizes_input(41, 1536));
                assert!(!policy.accepts_fp8(40, 1536));
                assert!(!policy.accepts_fp8(42, 1536));
                assert!(!policy.accepts_fp8(41, 1024));
            }
            for projection in ["output", "key", "value"] {
                let name = format!("action_head.dit_blocks.{index}.attention.{projection}");
                assert!(for_consumer(&name, 1536, 1536).is_none());
            }
        }
        for name in [
            "attention.query",
            "action_head.dit_blocks.bad.attention.query",
            "backbone.query",
        ] {
            assert!(for_consumer(name, 1536, 1536).is_none());
        }
    }

    #[test]
    fn final_adaln_preserves_existing_shape_rules_and_bf16_fallback() {
        let policy = for_consumer("action_head.output_projection", 1024, 1536).unwrap();
        assert!(policy.accepts_fp8(41, 1536));
        assert!(!policy.quantizes_input(41, 1536));
        assert!(policy.accepts_fp8(512, 1536));
        assert!(policy.quantizes_input(512, 1536));
        for rows in [40, 42, 145, 511, 513] {
            assert!(!policy.accepts_fp8(rows, 1536));
        }
        let other = for_consumer("backbone.vision.patch_embed", 1024, 1536).unwrap();
        assert!(!other.accepts_fp8(41, 1536));
        assert!(other.quantizes_input(512, 1536));
        for rule in RULES {
            let policy = for_consumer("unrelated.consumer", rule.cols, rule.inner).unwrap();
            assert_eq!(policy.layout(), rule.layout);
            assert!(policy.quantizes_input(rule.min_rows, rule.inner));
            assert!(policy.accepts_fp8(rule.max_rows, rule.inner));
        }
    }
}
