//! data_aggregate — Rust port of `analysis/compare.py`.
//!
//! Reads two JSONL benchmark outputs (Rust `llms` crate and Python LangChain)
//! and emits:
//!   1. A text table with per-metric statistics (count, mean, P50, P95, P99)
//!   2. A PNG bar chart (default `benchmarks/comparison.png`)
//!
//! Usage:
//!     data_aggregate --llms LLMS_JSONL --langchain LANGCHAIN_JSONL --output OUTPUT_PNG

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};
use clap::Parser;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use serde::Deserialize;

const KINDS: [&str; 4] = ["query", "send", "tool_time", "tool_cycle"];
const TITLES: [&str; 4] = [
    "Query (total prompt)",
    "Send (prompt \u{2192} HTTP)",
    "Tool Time",
    "Tool Cycle",
];
const COLORS: [RGBColor; 2] = [RGBColor(0x25, 0x63, 0xeb), RGBColor(0xdc, 0x26, 0x26)];

/// Common paths to try when locating a TrueType font for chart text.
const CANDIDATE_FONTS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
];

/// Locates a TrueType font on disk and registers it with plotters under the
/// `sans-serif` family. Without this, the `ab_glyph` font backend cannot draw
/// any text and chart rendering fails.
fn ensure_font() {
    use std::process::Command;
    use std::sync::Once;

    static REGISTER_FONT: Once = Once::new();
    REGISTER_FONT.call_once(|| {
        let mut bytes = None;
        for path in CANDIDATE_FONTS {
            if let Ok(data) = std::fs::read(path) {
                bytes = Some(data);
                break;
            }
        }
        if bytes.is_none() {
            // Fall back to asking fontconfig for the default sans-serif font.
            if let Ok(out) = Command::new("fc-match")
                .args(["--format", "%{file}", "sans-serif"])
                .output()
            {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !path.is_empty() {
                    bytes = std::fs::read(&path).ok();
                }
            }
        }
        match bytes {
            Some(data) => {
                let static_bytes: &'static [u8] = Box::leak(data.into_boxed_slice());
                let _ =
                    plotters::style::register_font("sans-serif", FontStyle::Normal, static_bytes);
            }
            None => {
                eprintln!("WARNING: no usable font found; the chart may render without text labels")
            }
        }
    });
}

#[derive(Parser, Debug)]
#[command(about = "Compare llms vs LangChain benchmarks")]
struct Args {
    /// Path to llms JSONL output
    #[arg(
        long,
        env = "LLMS_BENCH_OUTPUT",
        default_value = "benchmarks/bench_rust.jsonl"
    )]
    llms: PathBuf,

    /// Path to LangChain JSONL output
    #[arg(
        long,
        env = "LANGCHAIN_BENCH_OUTPUT",
        default_value = "benchmarks/bench_py.jsonl"
    )]
    langchain: PathBuf,

    /// Output PNG path for chart
    #[arg(
        long,
        env = "COMPARE_CHART",
        default_value = "benchmarks/comparison.png"
    )]
    output: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct BenchRecord {
    ts: u128,
    kind: String,
    duration_ms: f64,
    tool: Option<String>,
    model: String,
    framework: String,
}

fn load_jsonl(path: &PathBuf, framework: &str) -> Result<Vec<BenchRecord>> {
    if !path.exists() {
        eprintln!(
            "WARNING: {} not found \u{2014} skipping {}",
            path.display(),
            framework
        );
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut rec: BenchRecord = serde_json::from_str(line)
            .with_context(|| format!("failed to parse JSONL line in {}", path.display()))?;
        rec.framework = framework.to_string();
        records.push(rec);
    }
    Ok(records)
}

#[derive(Default)]
struct Stats {
    count: usize,
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    // pandas-compatible linear interpolation
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (h.ceil() as usize).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

fn stats(durations: &[f64]) -> Stats {
    let n = durations.len();
    if n == 0 {
        return Stats::default();
    }
    let mean = durations.iter().sum::<f64>() / n as f64;
    let mut sorted: Vec<f64> = durations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stats {
        count: n,
        mean,
        p50: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
    }
}

fn print_table(records: &[BenchRecord]) {
    if records.is_empty() {
        println!("No data to compare.");
        return;
    }

    println!();
    println!("{}", "=".repeat(96));
    println!("BENCHMARK COMPARISON: llms crate vs Python LangChain");
    println!("{}", "=".repeat(96));

    for kind in KINDS {
        let sub: Vec<&BenchRecord> = records.iter().filter(|r| r.kind == kind).collect();
        if sub.is_empty() {
            continue;
        }
        let tools: Vec<&str> = {
            let mut seen: Vec<&str> = Vec::new();
            for r in &sub {
                if let Some(tool) = r.tool.as_deref() {
                    if !seen.contains(&tool) {
                        seen.push(tool);
                    }
                }
            }
            seen
        };
        if tools.is_empty() {
            print_metric_table(kind, &sub, "");
        } else {
            for tool_name in tools {
                let tsub: Vec<&BenchRecord> = sub
                    .iter()
                    .filter(|r| r.tool.as_deref() == Some(tool_name))
                    .copied()
                    .collect();
                print_metric_table(kind, &tsub, &format!(" (tool={})", tool_name));
            }
        }
    }

    println!();
    println!("Note: LangChain's AgentExecutor batches multiple tool calls into a");
    println!("single follow-up LLM call. The `llms` crate sends one LLM call per");
    println!("tool. For multi-tool turns, LangChain may produce fewer tool_cycle");
    println!("records. Tool_time durations are comparable directly.");
    println!("{}", "=".repeat(96));
}

fn print_metric_table(kind: &str, sub: &[&BenchRecord], label: &str) {
    println!("\n  {}{}", kind.to_uppercase(), label);
    println!("  {}", "-".repeat(70));
    println!(
        "  {:<16} {:>6} {:>10} {:>10} {:>10} {:>10}  (ms)",
        "Framework", "Count", "Mean", "P50", "P95", "P99"
    );
    println!("  {}", "-".repeat(70));
    for fw in ["llms", "langchain"] {
        let fw_sub: Vec<&BenchRecord> = sub.iter().filter(|r| r.framework == fw).copied().collect();
        if fw_sub.is_empty() {
            println!(
                "  {:<16} {:>6} {:>10} {:>10} {:>10} {:>10}",
                fw, "\u{2014}", "\u{2014}", "\u{2014}", "\u{2014}", "\u{2014}"
            );
            continue;
        }
        let s = stats(&fw_sub.iter().map(|r| r.duration_ms).collect::<Vec<_>>());
        println!(
            "  {:<16} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            fw, s.count, s.mean, s.p50, s.p95, s.p99
        );
    }
}

fn generate_chart(records: &[BenchRecord], output: &PathBuf) -> Result<()> {
    if records.is_empty() {
        println!("No data for chart.");
        return Ok(());
    }

    ensure_font();

    let root = BitMapBackend::new(output, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    root.titled(
        "llms crate vs Python LangChain \u{2014} Benchmark Comparison",
        ("sans-serif", 28).into_font(),
    )?;

    let root_h = root.dim_in_pixel().1;
    let (charts_area, footer_area) = root.split_vertically((root_h as i32) - 50);
    let cells = charts_area.split_evenly((2, 2));

    for (i, (kind, title)) in KINDS.iter().zip(TITLES.iter()).enumerate() {
        chart_one(&cells[i], records, kind, title)?;
    }

    let all_tools: Vec<String> = {
        let s: BTreeSet<String> = records
            .iter()
            .filter(|r| r.kind == "tool_time" || r.kind == "tool_cycle")
            .filter_map(|r| r.tool.clone())
            .collect();
        s.into_iter().collect()
    };
    if !all_tools.is_empty() {
        let txt = format!("Tools: {}", all_tools.join(", "));
        let pos = (
            footer_area.dim_in_pixel().0 as i32 / 2,
            footer_area.dim_in_pixel().1 as i32 / 2,
        );
        footer_area.draw(&Text::new(
            txt,
            pos,
            ("sans-serif", 18)
                .into_font()
                .into_text_style(&footer_area)
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;
    }

    root.present()?;
    println!("\nChart saved to: {}", output.display());
    Ok(())
}

fn chart_one(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    records: &[BenchRecord],
    kind: &str,
    title: &str,
) -> Result<()> {
    let sub: Vec<&BenchRecord> = records.iter().filter(|r| r.kind == kind).collect();
    let mut tools: Vec<&str> = sub.iter().filter_map(|r| r.tool.as_deref()).collect();
    tools.sort_unstable();
    tools.dedup();
    let first = if tools.is_empty() { "(all)" } else { tools[0] };

    let mut means = [0.0f64; 2];
    let mut errors = [0.0f64; 2];
    for (i, fw) in ["llms", "langchain"].iter().enumerate() {
        let fw_sub: Vec<&BenchRecord> = sub
            .iter()
            .filter(|r| r.framework == *fw)
            .filter(|r| first == "(all)" || r.tool.as_deref() == Some(first))
            .copied()
            .collect();
        if fw_sub.is_empty() {
            continue;
        }
        let durations: Vec<f64> = fw_sub.iter().map(|r| r.duration_ms).collect();
        let m = durations.iter().sum::<f64>() / durations.len() as f64;
        let sd = if durations.len() > 1 {
            let var = durations.iter().map(|d| (d - m) * (d - m)).sum::<f64>()
                / (durations.len() as f64 - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        means[i] = m;
        errors[i] = sd;
    }

    let bound = means
        .iter()
        .zip(errors.iter())
        .map(|(m, e)| m + e)
        .fold(0.0, f64::max)
        * 1.15;

    let caption = if first != "(all)" {
        format!("{} \u{2014} tool: {}", title, first)
    } else {
        title.to_string()
    };

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 20).into_font())
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 30)
        .build_cartesian_2d(0.0f64..2.0f64, 0.0f64..bound.max(1.0))?;

    chart
        .configure_mesh()
        .x_labels(3)
        .x_label_formatter(&|v| match v {
            0.0 => "llms".to_string(),
            1.0 => "LangChain".to_string(),
            _ => String::new(),
        })
        .y_desc("Duration (ms)")
        .draw()?;

    for i in 0..2 {
        let x = i as f64;
        let m = means[i];
        let e = errors[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x - 0.35, 0.0), (x + 0.35, m)],
            COLORS[i].mix(0.9).filled(),
        )))?;

        if m > 0.0 {
            chart.draw_series(std::iter::once(Text::new(
                format!("{:.1}ms", m),
                (x, m),
                ("sans-serif", 18).into_font(),
            )))?;
        }
        if m > 0.0 && e > 0.0 {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, m - e), (x, m + e)],
                BLACK.stroke_width(2),
            )))?;
            for y in [m - e, m + e] {
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(x - 0.08, y), (x + 0.08, y)],
                    BLACK.stroke_width(2),
                )))?;
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut records = load_jsonl(&args.llms, "llms")?;
    records.extend(load_jsonl(&args.langchain, "langchain")?);

    if records.is_empty() {
        eprintln!("No benchmark data found. Run both harnesses first.");
        eprintln!("  llms:      {}", args.llms.display());
        eprintln!("  langchain: {}", args.langchain.display());
        process::exit(1);
    }

    print_table(&records);
    generate_chart(&records, &args.output)?;

    Ok(())
}
