//! Immutable Metal device facts used by kernel hardware dispatch.

/// GPU facts read from MLX once, when a Metal [`crate::Stream`] is made.
///
/// This type deliberately contains hardware facts only. Selected launch
/// geometry or implementations belong beside the kernel that reads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceCaps {
    /// Metal device name, such as `Apple M4`.
    pub device_name: String,
    /// The string MLX's own Metal dispatch reads, such as `applegpu_g16g`.
    /// `MLX_METAL_GPU_ARCH` overrides it for MLX and therefore here.
    pub architecture: String,
    /// The two digits before the final letter (13 on M1, 16 on M4).
    pub generation: Option<u32>,
    /// The final letter. MLX sizes command buffers and GEMM tiles by it; it is
    /// not a product tier (an M1 Pro reports `s`, an M4 `g`).
    pub device_class: Option<char>,
}

impl MetalDeviceCaps {
    /// Parses `applegpu_g<NN><letter>` as MLX does. Any other string leaves
    /// `generation` and `device_class` unset, so policies keep their default.
    pub fn new(device_name: String, architecture: String) -> Self {
        let parsed = match *architecture
            .strip_prefix("applegpu_g")
            .unwrap_or_default()
            .as_bytes()
        {
            [tens @ b'0'..=b'9', ones @ b'0'..=b'9', class @ b'a'..=b'z'] => Some((
                u32::from(tens - b'0') * 10 + u32::from(ones - b'0'),
                char::from(class),
            )),
            _ => None,
        };
        Self {
            device_name,
            architecture,
            generation: parsed.map(|(generation, _)| generation),
            device_class: parsed.map(|(_, class)| class),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(architecture: &str) -> (Option<u32>, Option<char>) {
        let caps = MetalDeviceCaps::new("Apple".into(), architecture.into());
        (caps.generation, caps.device_class)
    }

    #[test]
    fn parses_reported_architectures() {
        // Reported by MLX 0.32 on an M1 Pro and an M4.
        assert_eq!(parse("applegpu_g13s"), (Some(13), Some('s')));
        assert_eq!(parse("applegpu_g16g"), (Some(16), Some('g')));
        assert_eq!(parse("applegpu_g14g"), (Some(14), Some('g')));
    }

    #[test]
    fn unrecognized_architectures_stay_unclassified() {
        for architecture in [
            "",
            "applegpu_g",
            "applegpu_g13",
            "applegpu_g1s",
            "applegpu_gx3s",
            "applegpu_g13S",
            "applegpu_g13sx",
            "applegpu_g013s",
            "nvidia_sm89",
        ] {
            assert_eq!(parse(architecture), (None, None), "{architecture:?}");
        }
    }
}
