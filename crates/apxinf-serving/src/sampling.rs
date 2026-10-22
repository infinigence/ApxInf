use crate::supervisor::ApiError;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Selection {
    #[default]
    Greedy,
    Random {
        temperature: f32,
        top_p: f32,
    },
}

impl Selection {
    pub(crate) fn validate_vision(self) -> Result<(), ApiError> {
        self.validate(true)?;
        if matches!(self, Self::Random { temperature, .. } if temperature < 0.01_f32) {
            return Err(ApiError::invalid(
                "Vision temperature must be zero or at least 0.01.",
            ));
        }
        Ok(())
    }

    pub(crate) fn from_vision_request(value: &Value) -> Result<Self, ApiError> {
        if value["temperature"]
            .as_f64()
            .is_some_and(|temperature| temperature > 0.0 && temperature < 0.01)
        {
            return Err(ApiError::invalid(
                "Vision temperature must be zero or at least 0.01.",
            ));
        }
        let selection = Self::from_request(value, true)?;
        selection.validate_vision()?;
        Ok(selection)
    }

    pub(crate) fn vision_submission(self, request_id: &str) -> Value {
        let mut submission = self.submission(request_id);
        if let Self::Random { temperature, .. } = self {
            submission["temperature"] = json!((temperature as f64).max(0.01));
        }
        submission
    }

    pub(crate) fn validate(self, sampled: bool) -> Result<(), ApiError> {
        if let Self::Random { temperature, top_p } = self {
            if !sampled {
                return Err(ApiError::new(
                    400,
                    "unsupported_feature",
                    "This profile requires greedy selection.",
                ));
            }
            if !temperature.is_finite()
                || temperature <= 0.0
                || temperature > 2.0
                || !top_p.is_finite()
                || top_p <= 0.0
                || top_p > 1.0
            {
                return Err(ApiError::invalid(
                    "The sampling parameters are outside the supported range.",
                ));
            }
        }
        Ok(())
    }

    pub fn from_request(value: &Value, sampled: bool) -> Result<Self, ApiError> {
        let unsupported = |message| ApiError::new(400, "unsupported_feature", message);
        if value.get("top_k").is_some() {
            return Err(unsupported("This profile does not support top_k."));
        }
        if !sampled {
            for (key, expected) in [("temperature", 0.0), ("top_p", 1.0)] {
                if value.get(key).is_some_and(|v| v.as_f64() != Some(expected)) {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        format!("This profile requires {key}={expected}."),
                    ));
                }
            }
            return Ok(Self::Greedy);
        }
        let number = |key: &str, default: f64, maximum: f64| -> Result<f32, ApiError> {
            let raw = match value.get(key) {
                None => default,
                Some(v) => v
                    .as_f64()
                    .ok_or_else(|| ApiError::invalid(format!("{key} must be a number.")))?,
            };
            let rounded = raw as f32;
            if !raw.is_finite() || raw < 0.0 || raw > maximum || (raw > 0.0 && rounded == 0.0) {
                return Err(ApiError::invalid(format!(
                    "{key} is outside the supported range."
                )));
            }
            Ok(rounded)
        };
        let temperature = number("temperature", 1.0, 2.0)?;
        let top_p = number("top_p", if temperature == 0.0 { 1.0 } else { 0.95 }, 1.0)?;
        if temperature == 0.0 {
            if value.get("top_p").is_some_and(|v| v.as_f64() != Some(1.0)) {
                return Err(unsupported("Greedy selection requires top_p=1."));
            }
            Ok(Self::Greedy)
        } else if top_p == 0.0 {
            Err(ApiError::invalid("top_p must be greater than zero."))
        } else {
            Ok(Self::Random { temperature, top_p })
        }
    }

    pub fn submission(self, request_id: &str) -> Value {
        match self {
            Self::Greedy => json!({"mode":"greedy"}),
            Self::Random { temperature, top_p } => {
                let mut hash = Sha256::new();
                hash.update(b"apxinf-sampling-seed-v1\0");
                hash.update(request_id.as_bytes());
                let digest = hash.finalize();
                let seed = u64::from_be_bytes(digest[..8].try_into().unwrap()) & ((1u64 << 53) - 1);
                json!({"mode":"random","temperature":temperature,"top_p":top_p,"seed":seed})
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_temperature_checks_raw_input_and_serializes_the_inclusive_minimum() {
        for temperature in [0.009, 0.0099999999, 1e-100] {
            assert!(Selection::from_vision_request(&json!({"temperature":temperature})).is_err());
        }
        let selection = Selection::from_vision_request(&json!({"temperature":0.01})).unwrap();
        assert!(
            selection.submission("request")["temperature"]
                .as_f64()
                .unwrap()
                < 0.01
        );
        let submission = selection.vision_submission("request");
        assert_eq!(submission["temperature"].as_f64(), Some(0.01));
        assert_eq!(submission["seed"], selection.submission("request")["seed"]);
        assert_eq!(
            Selection::from_vision_request(&json!({"temperature":0})).unwrap(),
            Selection::Greedy
        );
        assert!(Selection::Random {
            temperature: 0.009,
            top_p: 0.95
        }
        .validate_vision()
        .is_err());
        assert!(Selection::from_request(&json!({"temperature":0.009}), true).is_ok());
    }

    #[test]
    fn explicit_profile_controls_defaults_and_overrides() {
        assert_eq!(
            Selection::from_request(&json!({}), false).unwrap(),
            Selection::Greedy
        );
        assert_eq!(
            Selection::from_request(&json!({}), true).unwrap(),
            Selection::Random {
                temperature: 1.0,
                top_p: 0.95
            }
        );
        assert_eq!(
            Selection::from_request(&json!({"temperature":0}), true).unwrap(),
            Selection::Greedy
        );
        assert!(Selection::from_request(&json!({"temperature":0.8}), false).is_err());
        assert!(Selection::from_request(&json!({"temperature":0,"top_p":0.95}), true).is_err());
        assert_eq!(
            Selection::from_request(&json!({"temperature":0.7,"top_p":0.8}), true).unwrap(),
            Selection::Random {
                temperature: 0.7,
                top_p: 0.8
            }
        );
        for value in [
            json!({"temperature":true}),
            json!({"temperature":1e-100}),
            json!({"temperature":2.01}),
            json!({"top_p":0}),
            json!({"top_p":false}),
            json!({"top_k":20}),
        ] {
            assert!(Selection::from_request(&value, true).is_err(), "{value}");
        }
    }

    #[test]
    fn request_identity_preserves_seed_across_attempts() {
        let selection = Selection::Random {
            temperature: 1.0,
            top_p: 0.95,
        };
        let first = selection.submission("11111111-1111-4111-8111-111111111111");
        assert_eq!(
            first,
            selection.submission("11111111-1111-4111-8111-111111111111")
        );
        assert_ne!(
            first["seed"],
            selection.submission("22222222-2222-4222-8222-222222222222")["seed"]
        );
        assert!(first["seed"].as_u64().unwrap() <= crate::contracts::MAX_SAFE_INTEGER);
        assert_eq!(
            Selection::Greedy.submission("unused"),
            json!({"mode":"greedy"})
        );
    }
}
