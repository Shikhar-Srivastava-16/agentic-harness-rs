//! Built-in benchmarking instrumentation for the `llms` crate.
//!
//! Enabled via the `bench` feature. All timers use `std::time::Instant`
//! and results are written as JSONL to a configurable path.
//!
//! Set the output file at runtime via the `LLMS_BENCH_OUTPUT` environment
//! variable (default: `bench_output.jsonl`).
//!
//! Each line of the output file is a JSON object with fields:
//! `ts` (unix millis), `kind` (one of `query`, `send`, `tool_time`,
//! `tool_cycle`), `duration_ms`, `tool` (optional, for tool events),
//! `model`, and `framework`.

use std::cell::RefCell;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

const FRAMEWORK: &str = "llms";

fn output_path() -> String {
    std::env::var("LLMS_BENCH_OUTPUT").unwrap_or_else(|_| "bench_output.jsonl".to_string())
}

#[derive(Serialize)]
struct BenchRecord {
    ts: u128,
    kind: &'static str,
    duration_ms: f64,
    tool: Option<String>,
    model: String,
    framework: &'static str,
}

struct BenchContext {
    prompt_start: Option<Instant>,
    tool_cycle_start: Option<Instant>,
    tool_cycle_name: Option<String>,
}

impl Default for BenchContext {
    fn default() -> Self {
        BenchContext {
            prompt_start: None,
            tool_cycle_start: None,
            tool_cycle_name: None,
        }
    }
}

thread_local! {
    static CTX: RefCell<BenchContext> = RefCell::new(BenchContext::default());
}

static FILE: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn duration_ms(d: Duration) -> f64 {
    d.as_nanos() as f64 / 1_000_000.0
}

fn get_file() -> &'static Mutex<std::fs::File> {
    FILE.get_or_init(|| {
        let path = output_path();

        let _ = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path);

        Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|_| {
                    panic!("bench: failed to open output file at {}", path)
                }),
        )
    })
}

fn emit_record(kind: &'static str, duration: Duration, tool: Option<String>, model: &str) {
    let record = BenchRecord {
        ts: now_ms(),
        kind,
        duration_ms: duration_ms(duration),
        tool,
        model: model.to_string(),
        framework: FRAMEWORK,
    };

    let json = serde_json::to_string(&record).unwrap_or_else(|_| "{}".to_string());
    let mut file = get_file()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let _ = writeln!(file, "{}", json);
}

/// Called at the entry of `prompt()`. Records the start instant used for
/// "send" and "query" measurements. Any leftover tool-cycle state from a
/// previous (possibly errored) prompt is cleared.
pub fn begin_prompt() {
    CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        c.prompt_start = Some(Instant::now());
        c.tool_cycle_start = None;
        c.tool_cycle_name = None;
    });
}

/// Emits a "query" record measuring total `prompt()` duration and clears
/// all thread-local timing state.
pub fn emit_query(duration: Duration, model: &str) {
    emit_record("query", duration, None, model);
    CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        c.prompt_start = None;
        c.tool_cycle_start = None;
        c.tool_cycle_name = None;
    });
}

/// Called right before each HTTP POST `.send()`. If a prompt is in flight
/// this records the "send" metric (time from `prompt()` entry to the HTTP
/// request being made). If a tool cycle is in flight it records the "tool_cycle"
/// metric (time from receiving a tool request to sending the response).
/// Exactly one of these states will be set at any point during a prompt.
pub fn before_http_send(model: &str) {
    CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();

        if let Some(start) = c.prompt_start.take() {
            emit_record("send", start.elapsed(), None, model);
        }

        if let Some(start) = c.tool_cycle_start.take() {
            let tool = c.tool_cycle_name.take();
            emit_record("tool_cycle", start.elapsed(), tool, model);
        }
    });
}

/// Marks the start of a tool-call iteration in the `ToolReady::prompt`
/// loop. Used together with `before_http_send` to measure tool-cycle
/// latency.
pub fn start_tool_cycle(tool_name: &str) {
    CTX.with(|ctx| {
        let mut c = ctx.borrow_mut();
        c.tool_cycle_start = Some(Instant::now());
        c.tool_cycle_name = Some(tool_name.to_string());
    });
}

/// Emits a "tool_time" record for the duration of a single `run_tool` call.
pub fn emit_tool_time(tool_name: &str, duration: Duration, model: &str) {
    emit_record("tool_time", duration, Some(tool_name.to_string()), model);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulates the full lifecycle of `ToolReady::prompt` with a single tool
    /// call and verifies that all four metric types are emitted in the correct
    /// order with the correct tool attribution.
    #[test]
    fn test_bench_emits_all_metrics() {
        // 1. Prompt starts → begin_prompt sets thread-local prompt_start
        begin_prompt();

        // 2. HTTP request fires (initial query) → before_http_send consumes
        //    prompt_start and emits "send"
        std::thread::sleep(Duration::from_millis(5));
        before_http_send("test-model");

        // 3. Tool call received → start_tool_cycle sets cycle start + name
        start_tool_cycle("test_tool");

        // 4. Tool executes → emit_tool_time
        emit_tool_time("test_tool", Duration::from_millis(3), "test-model");

        // 5. tool_respond sends HTTP → before_http_send consumes
        //    tool_cycle_start and emits "tool_cycle"
        std::thread::sleep(Duration::from_millis(7));
        before_http_send("test-model");

        // 6. prompt() returns → emit_query emits "query" and clears state
        emit_query(Duration::from_millis(15), "test-model");

        // Read and parse the output file
        let path = output_path();
        let content = std::fs::read_to_string(&path).expect("failed to read bench output");
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

        assert_eq!(lines.len(), 4, "expected 4 benchmark records");

        let records: Vec<serde_json::Value> = lines
            .iter()
            .map(|l| serde_json::from_str(l).expect("failed to parse JSONL"))
            .collect();

        for record in &records {
            assert!(record["ts"].as_u64().is_some(), "missing ts field");
            assert!(record["kind"].as_str().is_some(), "missing kind field");
            assert!(
                record["duration_ms"].as_f64().is_some(),
                "missing duration_ms field"
            );
            assert!(record["model"].as_str().is_some(), "missing model field");
            assert_eq!(record["framework"].as_str(), Some("llms"));
        }

        assert_eq!(records[0]["kind"], "send");
        assert!(records[0]["tool"].is_null());
        assert!(records[0]["duration_ms"].as_f64().unwrap() > 0.0);

        assert_eq!(records[1]["kind"], "tool_time");
        assert_eq!(records[1]["tool"], "test_tool");
        assert_eq!(records[1]["duration_ms"].as_f64().unwrap(), 3.0);

        assert_eq!(records[2]["kind"], "tool_cycle");
        assert_eq!(records[2]["tool"], "test_tool");
        assert!(records[2]["duration_ms"].as_f64().unwrap() > 0.0);

        assert_eq!(records[3]["kind"], "query");
        assert!(records[3]["tool"].is_null());
        assert_eq!(records[3]["duration_ms"].as_f64().unwrap(), 15.0);
    }
}
