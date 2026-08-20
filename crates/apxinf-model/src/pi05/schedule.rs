//! Static shape schedule shared by graph allocation and GEMM tuning.

use apxinf_core::{Error, Result};

use super::Pi05Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pi05Stage {
    Vision,
    Language,
    Action,
    Conditioning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemmShape {
    pub name: String,
    pub stage: Pi05Stage,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub repetitions: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pi05ExecutionSchedule {
    pub vision_tokens: usize,
    pub prefix_tokens: usize,
    pub action_tokens: usize,
    pub attention_kv_tokens: usize,
    pub gemms: Vec<GemmShape>,
}

impl Pi05ExecutionSchedule {
    pub fn new(config: &Pi05Config) -> Result<Self> {
        Self::for_token_count(config, config.max_token_len)
    }

    pub fn for_token_count(config: &Pi05Config, token_count: usize) -> Result<Self> {
        config.validate()?;
        if token_count == 0 || token_count > config.max_token_len {
            return Err(Error::Other(format!(
                "π0.5 schedule token count must be in 1..={}, got {token_count}",
                config.max_token_len
            )));
        }
        let vision_tokens = config.num_views * config.patches_per_view();
        let prefix_tokens = vision_tokens + token_count;
        let action_tokens = config.action_horizon;
        let attention_kv_tokens = prefix_tokens + action_tokens;
        let mut gemms = Vec::new();
        let mut push = |name: &str, stage, m, n, k, repetitions| {
            gemms.push(GemmShape {
                name: name.into(),
                stage,
                m,
                n,
                k,
                repetitions,
            });
        };

        push(
            "vision.patch",
            Pi05Stage::Vision,
            vision_tokens,
            config.vision_width,
            3 * config.patch_size * config.patch_size,
            1,
        );
        push(
            "vision.qkv",
            Pi05Stage::Vision,
            vision_tokens,
            3 * config.vision_width,
            config.vision_width,
            config.vision_depth,
        );
        push(
            "vision.attention_out",
            Pi05Stage::Vision,
            vision_tokens,
            config.vision_width,
            config.vision_width,
            config.vision_depth,
        );
        push(
            "vision.fc1",
            Pi05Stage::Vision,
            vision_tokens,
            config.vision_mlp_dim,
            config.vision_width,
            config.vision_depth,
        );
        push(
            "vision.fc2",
            Pi05Stage::Vision,
            vision_tokens,
            config.vision_width,
            config.vision_mlp_dim,
            config.vision_depth,
        );
        push(
            "vision.projector",
            Pi05Stage::Vision,
            vision_tokens,
            config.language.width,
            config.vision_width,
            1,
        );

        let l = config.language;
        // The final language layer only produces normalized QKV and the K/V
        // cache consumed by the paired action layer. Its prefix attention and
        // MLP tail are intentionally skipped because no later operation reads
        // the final prefix hidden state.
        let language_tail_repetitions = l.depth.saturating_sub(1);
        push(
            "language.qkv",
            Pi05Stage::Language,
            prefix_tokens,
            l.num_heads * l.head_dim + 2 * l.num_kv_heads * l.head_dim,
            l.width,
            l.depth,
        );
        push(
            "language.attention_out",
            Pi05Stage::Language,
            prefix_tokens,
            l.width,
            l.num_heads * l.head_dim,
            language_tail_repetitions,
        );
        push(
            "language.gate_up",
            Pi05Stage::Language,
            prefix_tokens,
            2 * l.mlp_dim,
            l.width,
            language_tail_repetitions,
        );
        push(
            "language.down",
            Pi05Stage::Language,
            prefix_tokens,
            l.width,
            l.mlp_dim,
            language_tail_repetitions,
        );

        let a = config.action_expert;
        let denoise_repetitions = a.depth * config.num_flow_steps;
        push(
            "action.input",
            Pi05Stage::Action,
            action_tokens,
            a.width,
            config.action_dim,
            config.num_flow_steps,
        );
        push(
            "action.qkv",
            Pi05Stage::Action,
            action_tokens,
            a.num_heads * a.head_dim + 2 * a.num_kv_heads * a.head_dim,
            a.width,
            denoise_repetitions,
        );
        push(
            "action.attention_out",
            Pi05Stage::Action,
            action_tokens,
            a.width,
            a.num_heads * a.head_dim,
            denoise_repetitions,
        );
        push(
            "action.gate_up",
            Pi05Stage::Action,
            action_tokens,
            2 * a.mlp_dim,
            a.width,
            denoise_repetitions,
        );
        push(
            "action.down",
            Pi05Stage::Action,
            action_tokens,
            a.width,
            a.mlp_dim,
            denoise_repetitions,
        );
        push(
            "action.output",
            Pi05Stage::Action,
            action_tokens,
            config.action_dim,
            a.width,
            config.num_flow_steps,
        );

        push(
            "time.mlp_in",
            Pi05Stage::Conditioning,
            1,
            a.width,
            a.width,
            config.num_flow_steps,
        );
        push(
            "time.mlp_out",
            Pi05Stage::Conditioning,
            1,
            a.width,
            a.width,
            config.num_flow_steps,
        );
        push(
            "action.ada_style",
            Pi05Stage::Conditioning,
            1,
            3 * a.width,
            a.width,
            config.num_flow_steps * (2 * a.depth + 1),
        );

        if gemms
            .iter()
            .any(|shape| shape.m == 0 || shape.n == 0 || shape.k == 0)
        {
            return Err(Error::Other("π0.5 schedule contains an empty GEMM".into()));
        }
        Ok(Self {
            vision_tokens,
            prefix_tokens,
            action_tokens,
            attention_kv_tokens,
            gemms,
        })
    }

    pub fn exact_tactic_key(shape: &GemmShape) -> String {
        format!("fp8_f16_m{}_n{}_k{}", shape.m, shape.n, shape.k)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn thor_two_view_schedule_matches_kernel_contract() {
        let schedule = Pi05ExecutionSchedule::new(&Pi05Config::thor_two_view()).unwrap();
        assert_eq!(schedule.vision_tokens, 512);
        assert_eq!(schedule.prefix_tokens, 712);
        assert_eq!(schedule.action_tokens, 10);
        assert_eq!(schedule.attention_kv_tokens, 722);
        let language_gate_up = schedule
            .gemms
            .iter()
            .find(|x| x.name == "language.gate_up")
            .unwrap();
        assert_eq!(
            (language_gate_up.m, language_gate_up.n, language_gate_up.k),
            (712, 32_768, 2048)
        );
        assert_eq!(language_gate_up.repetitions, 17);
        let language_qkv = schedule
            .gemms
            .iter()
            .find(|x| x.name == "language.qkv")
            .unwrap();
        assert_eq!(language_qkv.repetitions, 18);
        let action_qkv = schedule
            .gemms
            .iter()
            .find(|x| x.name == "action.qkv")
            .unwrap();
        assert_eq!((action_qkv.m, action_qkv.n, action_qkv.k), (10, 2560, 1024));
        assert_eq!(action_qkv.repetitions, 180);
        let styles = schedule
            .gemms
            .iter()
            .find(|x| x.name == "action.ada_style")
            .unwrap();
        assert_eq!((styles.m, styles.n, styles.k), (1, 3072, 1024));
        assert_eq!(styles.repetitions, 370);
    }

    #[test]
    fn schedule_can_target_actual_short_prompt_shape() {
        let schedule =
            Pi05ExecutionSchedule::for_token_count(&Pi05Config::thor_two_view(), 10).unwrap();
        assert_eq!(schedule.prefix_tokens, 522);
        let language_gate_up = schedule
            .gemms
            .iter()
            .find(|x| x.name == "language.gate_up")
            .unwrap();
        assert_eq!(language_gate_up.m, 522);
    }

    #[test]
    fn thor_three_view_t10_has_complete_exact_gemm_set() {
        let schedule =
            Pi05ExecutionSchedule::for_token_count(&Pi05Config::thor_three_view(), 10).unwrap();
        assert_eq!(schedule.vision_tokens, 768);
        assert_eq!(schedule.prefix_tokens, 778);
        assert_eq!(schedule.action_tokens, 10);
        assert_eq!(schedule.attention_kv_tokens, 788);
        let keys = schedule
            .gemms
            .iter()
            .map(|shape| (shape.m, shape.n, shape.k))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), 18);
        assert!(keys.contains(&(768, 1152, 588)));
        assert!(keys.contains(&(778, 32_768, 2048)));
        assert!(keys.contains(&(10, 32, 1024)));
    }
}
