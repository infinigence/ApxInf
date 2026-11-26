#![cfg(all(feature = "signpost", target_os = "macos"))]
//! Reads the emitted intervals back from the unified log, where Points of
//! Interest signposts are persisted, instead of trusting that the FFI ran.

use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use apxinf_mlx::signpost;
use serde_json::Value;

struct Event {
    kind: String,
    id: u64,
    message: String,
    time: u64,
}

fn events(pid: u32) -> Vec<Event> {
    let predicate = format!("subsystem == \"com.apxinf\" AND processID == {pid}");
    let output = Command::new("/usr/bin/log")
        .args(["show", "--signpost", "--last", "2m", "--style", "ndjson"])
        .args(["--predicate", &predicate])
        .output()
        .expect("run /usr/bin/log");
    assert!(
        output.status.success(),
        "log show failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["eventType"] == "signpostEvent")
        .map(|record| Event {
            kind: record["signpostType"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            id: record["signpostID"].as_u64().unwrap_or_default(),
            message: record["eventMessage"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            time: record["machTimestamp"].as_u64().unwrap_or_default(),
        })
        .collect()
}

#[test]
fn nested_ranges_reach_the_unified_log_with_exact_names() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // A name is data, never a format: conversions in it must print verbatim.
    let outer = format!("test-{nonce}/outer %s %n %{{public}}s");
    // A slice of a longer buffer: no NUL ends it, and the tail must not leak.
    let buffer = format!("test-{nonce}/innerTAIL");
    let inner = &buffer[..buffer.len() - "TAIL".len()];

    {
        let _outer = signpost::range(&outer);
        let _inner = signpost::range(inner);
    }

    let pid = std::process::id();
    let deadline = Instant::now() + Duration::from_secs(30);
    let recorded = loop {
        let recorded = events(pid);
        if recorded.len() >= 4 || Instant::now() > deadline {
            break recorded;
        }
        std::thread::sleep(Duration::from_millis(500));
    };

    let begin = |name: &str| {
        let matches: Vec<_> = recorded
            .iter()
            .filter(|e| e.kind == "begin" && e.message == name)
            .collect();
        assert_eq!(matches.len(), 1, "one begin named {name:?}");
        matches[0]
    };
    let end = |id: u64| {
        let matches: Vec<_> = recorded
            .iter()
            .filter(|e| e.kind == "end" && e.id == id)
            .collect();
        assert_eq!(matches.len(), 1, "one end for interval {id}");
        matches[0]
    };
    let (outer_begin, inner_begin) = (begin(&outer), begin(inner));
    assert_ne!(
        outer_begin.id, inner_begin.id,
        "nested intervals share an id"
    );
    let (outer_end, inner_end) = (end(outer_begin.id), end(inner_begin.id));
    assert!(outer_begin.time <= inner_begin.time);
    assert!(inner_begin.time <= inner_end.time);
    assert!(inner_end.time <= outer_end.time);
}
