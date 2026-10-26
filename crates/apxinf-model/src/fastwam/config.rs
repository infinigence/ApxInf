//! The first supported Fast-WAM checkpoint profile. Reject architectural variants
//! explicitly rather than silently interpreting their tensors as this profile.
use apxinf_core::{Error, Result};
use std::path::Path;
#[derive(Clone)]
pub(super) struct Config {
    pub steps: usize,
}
impl Config {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| Error::Other(e.to_string()))?,
        )
        .map_err(|e| Error::Other(e.to_string()))?;
        Self::from_value(&v)
    }

    fn from_value(v: &serde_json::Value) -> Result<Self> {
        // FastWAMConfig defaults this flag to true and makes it authoritative
        // over both expert configs. This profile implements FP32 attention only.
        match v.get("fp32_attention") {
            None | Some(serde_json::Value::Bool(true)) => {}
            _ => {
                return Err(Error::Other(
                    "unsupported Fast-WAM fp32_attention: expected true".into(),
                ));
            }
        }
        for (key, expected) in [
            ("type", serde_json::json!("fastwam")),
            ("action_dim", serde_json::json!(7)),
            ("proprio_dim", serde_json::json!(8)),
            ("action_horizon", serde_json::json!(32)),
            ("context_len", serde_json::json!(128)),
            ("image_size", serde_json::json!([224, 448])),
        ] {
            if v[key] != expected {
                return Err(Error::Other(format!(
                    "unsupported Fast-WAM {key}: {}",
                    v[key]
                )));
            }
        }
        for (expert, width, ffn) in [("video", 3072, 14336), ("action", 1024, 4096)] {
            let config = &v[format!("{expert}_dit_config")];
            for (key, expected) in [
                ("hidden_dim", width),
                ("ffn_dim", ffn),
                ("num_heads", 24),
                ("attn_head_dim", 128),
                ("num_layers", 30),
                ("text_dim", 4096),
                ("freq_dim", 256),
            ] {
                if config[key].as_u64() != Some(expected) {
                    return Err(Error::Other(format!("unsupported Fast-WAM {expert}.{key}")));
                }
            }
            if config["eps"].as_f64() != Some(1e-6) {
                return Err(Error::Other(
                    "unsupported Fast-WAM normalization epsilon".into(),
                ));
            }
        }
        let video = &v["video_dit_config"];
        if video["patch_size"] != serde_json::json!([1, 2, 2])
            || video["in_dim"] != 48
            || video["seperated_timestep"] != true
            || video["fuse_vae_embedding_in_latents"] != true
            || video["video_attention_mask_mode"] != "first_frame_causal"
            || video["action_conditioned"] != false
        {
            return Err(Error::Other(
                "unsupported Fast-WAM video conditioning".into(),
            ));
        }
        if v["action_scheduler"]["infer_shift"].as_f64() != Some(5.)
            || v["action_scheduler"]["num_train_timesteps"] != 1000
        {
            return Err(Error::Other("unsupported Fast-WAM action schedule".into()));
        }
        // These top-level inference options override the scheduler defaults in
        // the reference. Reject incompatible overrides before loading weights.
        if v.get("num_inference_steps")
            .is_some_and(|value| value.as_u64() != Some(10))
        {
            return Err(Error::Other(
                "unsupported Fast-WAM num_inference_steps: expected 10".into(),
            ));
        }
        if v.get("sigma_shift")
            .is_some_and(|value| !value.is_null() && value.as_f64() != Some(5.))
        {
            return Err(Error::Other(
                "unsupported Fast-WAM sigma_shift: expected null or 5".into(),
            ));
        }
        Ok(Self { steps: 10 })
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use serde_json::{json, Value};

    fn supported_profile() -> Value {
        let mut profile = json!({
            "type": "fastwam",
            "action_dim": 7,
            "proprio_dim": 8,
            "action_horizon": 32,
            "context_len": 128,
            "image_size": [224, 448],
            "fp32_attention": true,
            "action_scheduler": {"infer_shift": 5.0, "num_train_timesteps": 1000}
        });
        for (name, width, ffn) in [("video", 3072, 14336), ("action", 1024, 4096)] {
            profile[format!("{name}_dit_config")] = json!({
                "hidden_dim": width,
                "ffn_dim": ffn,
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1e-6
            });
        }
        let video = profile["video_dit_config"].as_object_mut().unwrap();
        for (key, value) in [
            ("patch_size", json!([1, 2, 2])),
            ("in_dim", json!(48)),
            ("seperated_timestep", json!(true)),
            ("fuse_vae_embedding_in_latents", json!(true)),
            ("video_attention_mask_mode", json!("first_frame_causal")),
            ("action_conditioned", json!(false)),
        ] {
            video.insert(key.into(), value);
        }
        profile
    }

    #[test]
    fn fastwam_config_accepts_explicit_and_default_fp32_attention() {
        let mut profile = supported_profile();
        assert_eq!(Config::from_value(&profile).unwrap().steps, 10);
        profile.as_object_mut().unwrap().remove("fp32_attention");
        assert_eq!(Config::from_value(&profile).unwrap().steps, 10);
    }

    #[test]
    fn fastwam_config_rejects_other_attention_precision_and_invalid_types() {
        for value in [
            json!(false),
            json!(null),
            json!("true"),
            json!(1),
            json!([]),
        ] {
            let mut profile = supported_profile();
            profile["fp32_attention"] = value;
            let result = Config::from_value(&profile);
            assert!(result.is_err(), "accepted {}", profile["fp32_attention"]);
            assert!(result.err().unwrap().to_string().contains("fp32_attention"));
        }
    }

    #[test]
    fn fastwam_config_validates_effective_inference_schedule() {
        let mut profile = supported_profile();
        assert!(Config::from_value(&profile).is_ok());
        profile["num_inference_steps"] = json!(10);
        for shift in [json!(null), json!(5), json!(5.0)] {
            profile["sigma_shift"] = shift;
            assert!(Config::from_value(&profile).is_ok());
        }
        for (field, values) in [
            (
                "num_inference_steps",
                vec![
                    json!(20),
                    json!(0),
                    json!(null),
                    json!("10"),
                    json!(true),
                    json!(10.5),
                ],
            ),
            (
                "sigma_shift",
                vec![json!(3), json!(0), json!("5"), json!(true)],
            ),
        ] {
            for value in values {
                let mut profile = supported_profile();
                profile[field] = value;
                let error = Config::from_value(&profile)
                    .err()
                    .expect("unsupported schedule accepted");
                assert!(error.to_string().contains(field));
            }
        }
    }

    #[test]
    fn fastwam_config_rejects_unsupported_geometry_conditioning_and_schedule() {
        for (path, value) in [
            ("/action_horizon", json!(16)),
            ("/context_len", json!(256)),
            ("/image_size", json!([224, 224])),
            ("/action_dit_config/num_heads", json!(16)),
            ("/video_dit_config/hidden_dim", json!(2048)),
            ("/video_dit_config/eps", json!(1e-5)),
            ("/video_dit_config/patch_size", json!([1, 4, 4])),
            ("/video_dit_config/action_conditioned", json!(true)),
            (
                "/video_dit_config/video_attention_mask_mode",
                json!("causal"),
            ),
            ("/action_scheduler/infer_shift", json!(3.0)),
            ("/action_scheduler/num_train_timesteps", json!(500)),
        ] {
            let mut profile = supported_profile();
            *profile.pointer_mut(path).unwrap() = value;
            assert!(Config::from_value(&profile).is_err(), "accepted {path}");
        }
    }
}
