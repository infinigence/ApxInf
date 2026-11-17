use std::ffi::OsStr;

pub(crate) fn enabled(feature: &str) -> bool {
    let name = format!("APXINF_GR00T_DISABLE_FP8_{feature}");
    enabled_from_value(&name, std::env::var_os(&name).as_deref())
}

fn enabled_from_value(name: &str, value: Option<&OsStr>) -> bool {
    let Some(value) = value else {
        return true;
    };
    match value
        .to_str()
        .map(|text| text.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("0" | "false") => true,
        Some("1" | "true") => false,
        _ => panic!("{name} must be 0/false (enabled) or 1/true (disabled), got {value:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_keeps_fp8_enabled() {
        assert!(enabled_from_value("test", None));
    }

    #[test]
    fn zero_and_false_keep_fp8_enabled() {
        for value in ["0", "false", "FALSE", " false "] {
            assert!(enabled_from_value("test", Some(OsStr::new(value))));
        }
    }

    #[test]
    fn one_and_true_disable_fp8() {
        for value in ["1", "true", "TRUE", " true "] {
            assert!(!enabled_from_value("test", Some(OsStr::new(value))));
        }
    }

    #[test]
    fn invalid_values_are_not_silently_accepted() {
        for value in ["", "2", "ture", "yes"] {
            assert!(std::panic::catch_unwind(|| {
                enabled_from_value("test", Some(OsStr::new(value)))
            })
            .is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_values_are_rejected() {
        use std::os::unix::ffi::OsStrExt;
        assert!(std::panic::catch_unwind(|| {
            enabled_from_value("test", Some(OsStr::from_bytes(&[0xff])))
        })
        .is_err());
    }
}
