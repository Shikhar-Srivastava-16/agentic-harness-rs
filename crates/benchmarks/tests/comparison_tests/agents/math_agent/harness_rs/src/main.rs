use benchmarks::test_config::HarnessConfig;
use llms::LlmLike;
use llms::config::AppConfig;
use llms::config::{self, ErrorMode};
use llms::minimax::Nvidia;
use llms::tooling::ToolReady;
use serde_json::{Value, json};
use std::path::Path;

fn get_i64(v: &Value, key: &str) -> i64 {
    v[key].as_i64().unwrap_or(0)
}

fn fmt_num(f: f64) -> String {
    if f.fract() == 0.0 {
        format!("{}", f as i64)
    } else {
        format!("{:.2}", f)
    }
}

fn add_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    fmt_num(get_i64(&v, "a") as f64 + get_i64(&v, "b") as f64)
}

fn subtract_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    fmt_num(get_i64(&v, "a") as f64 - get_i64(&v, "b") as f64)
}

fn multiply_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    fmt_num(get_i64(&v, "a") as f64 * get_i64(&v, "b") as f64)
}

fn divide_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let b = get_i64(&v, "b");
    if b == 0 {
        return "error: division by zero".to_string();
    }
    fmt_num(get_i64(&v, "a") as f64 / b as f64)
}

fn power_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let base = get_i64(&v, "base");
    let exponent = get_i64(&v, "exponent");
    fmt_num(base.saturating_pow(exponent.max(0) as u32) as f64)
}

fn modulo_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let b = get_i64(&v, "b");
    if b == 0 {
        return "error: modulo by zero".to_string();
    }
    fmt_num((get_i64(&v, "a").rem_euclid(b)) as f64)
}

fn factorial_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let n = get_i64(&v, "n");
    if n < 0 || n > 20 {
        return format!("error: factorial of {} is out of range", n);
    }
    let mut acc: i64 = 1;
    for i in 2..=n {
        acc *= i;
    }
    fmt_num(acc as f64)
}

fn fibonacci_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let n = get_i64(&v, "n").max(0);
    let (mut a, mut b) = (0i64, 1i64);
    for _ in 0..n {
        let t = a.checked_add(b).unwrap_or(i64::MAX);
        a = b;
        b = t;
    }
    fmt_num(a as f64)
}

fn gcd_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let mut a = get_i64(&v, "a").abs();
    let mut b = get_i64(&v, "b").abs();
    while b != 0 {
        let t = a;
        a = b;
        b = t % b;
    }
    fmt_num(a as f64)
}

fn average_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let nums: Vec<f64> = v["nums"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|n| n.as_f64().or_else(|| n.as_i64().map(|i| i as f64)))
                .collect()
        })
        .unwrap_or_default();
    if nums.is_empty() {
        return "error: no numbers provided".to_string();
    }
    let sum: f64 = nums.iter().sum();
    fmt_num(sum / nums.len() as f64)
}

fn init_backend(cfg: &AppConfig) -> Result<Nvidia, llms::LlmError> {
    let api = std::env::var("API_KEY").expect("API key not found");

    eprintln!("{api}");

    let mut o = Nvidia::init(
        Some(cfg.system_prompt.clone()),
        Some(cfg.url.clone()),
        llms::minimax::NvidiaConfig {
            api_key: api,
            model: cfg.model.clone(),
        },
    )?;

    o.register_tool(
        "add_tool".to_string(),
        "Adds two integers a and b and returns the sum. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        Box::new(add_tool),
    )?;

    o.register_tool(
        "subtract_tool".to_string(),
        "Subtracts integer b from integer a and returns the difference. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        Box::new(subtract_tool),
    )?;

    o.register_tool(
        "multiply_tool".to_string(),
        "Multiplies two integers a and b and returns the product. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        Box::new(multiply_tool),
    )?;

    o.register_tool(
        "divide_tool".to_string(),
        "Divides integer a by integer b and returns the result. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "Dividend" },
                "b": { "type": "integer", "description": "Divisor" }
            },
            "required": ["a", "b"]
        }),
        Box::new(divide_tool),
    )?;

    o.register_tool(
        "power_tool".to_string(),
        "Raises integer base to the integer exponent and returns the result. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "base": { "type": "integer", "description": "The base number" },
                "exponent": { "type": "integer", "description": "The power to raise to" }
            },
            "required": ["base", "exponent"]
        }),
        Box::new(power_tool),
    )?;

    o.register_tool(
        "modulo_tool".to_string(),
        "Returns the remainder when integer a is divided by integer b. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        Box::new(modulo_tool),
    )?;

    o.register_tool(
        "factorial_tool".to_string(),
        "Returns the factorial of a non-negative integer n (n <= 20). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "n": { "type": "integer", "description": "Non-negative number to factor" }
            },
            "required": ["n"]
        }),
        Box::new(factorial_tool),
    )?;

    o.register_tool(
        "fibonacci_tool".to_string(),
        "Returns the nth Fibonacci number (0-indexed). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "n": { "type": "integer", "description": "Which Fibonacci number to compute" }
            },
            "required": ["n"]
        }),
        Box::new(fibonacci_tool),
    )?;

    o.register_tool(
        "gcd_tool".to_string(),
        "Returns the greatest common divisor of two integers a and b. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        Box::new(gcd_tool),
    )?;

    o.register_tool(
        "average_tool".to_string(),
        "Returns the arithmetic mean of a list of numbers. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "nums": { "type": "array", "items": { "type": "number" }, "description": "Numbers to average" }
            },
            "required": ["nums"]
        }),
        Box::new(average_tool),
    )?;

    Ok(o)
}

fn main() {
    let cfg_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config.toml");
    let content = std::fs::read_to_string(&cfg_path).expect("failed to read config.toml");
    let cfg: HarnessConfig = toml::from_str(&content).expect("failed to parse config.toml");

    unsafe {
        std::env::set_var(
            "LLMS_BENCH_OUTPUT",
            Path::new(env!("CARGO_MANIFEST_DIR")).join(&cfg.bench_output),
        );
    }

    let app_cfg = AppConfig {
        system_prompt: cfg.system_prompt.clone(),
        model: cfg.model.clone(),
        url: cfg.url.clone(),
        api_key: std::env::var("API_KEY").ok().filter(|s| !s.is_empty()),
        backend: cfg.backend.clone(),
        query: cfg.query.clone(),
        error_mode: ErrorMode::from_str(&cfg.error_mode),
    };
    config::init(app_cfg.clone());

    let mut backend = init_backend(&app_cfg).expect("failed to init backend");

    let prompts = &cfg.prompts;
    let iterations = cfg.iterations as usize;

    println!(
        "Running {} iterations with model '{}'",
        cfg.iterations, cfg.model
    );
    for i in 0..iterations {
        let prompt = prompts[i % prompts.len()].clone();
        print!("  [{}/{}] prompt: \"{}\" ... ", i + 1, iterations, prompt);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match ToolReady::prompt(&mut backend, prompt) {
            Ok(ans) => println!("OK ({})", ans),
            Err(e) => panic!("ERROR: {}", e),
        }
    }

    println!("Done. Results written to bench output file.");
    llms::bench::finalize();
}
