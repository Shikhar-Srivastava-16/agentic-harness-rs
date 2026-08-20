use benchmarks::test_config::HarnessConfig;
use llms::LlmLike;
use llms::config::AppConfig;
use llms::config::{self, ErrorMode};
use llms::minimax::Nvidia;
use llms::tooling::ToolReady;
use serde_json::{Value, json};
use std::path::Path;

fn get_str(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn get_f64(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or(0.0)
}

fn fmt_num(f: f64) -> String {
    if f.fract() == 0.0 {
        format!("{}", f as i64)
    } else {
        format!("{:.2}", f)
    }
}

fn fmt_2(f: f64) -> String {
    format!("{:.2}", f)
}

fn rates() -> Vec<(&'static str, f64)> {
    vec![
        ("USD", 1.0),
        ("EUR", 0.92),
        ("GBP", 0.79),
        ("JPY", 149.5),
        ("INR", 83.1),
    ]
}

fn rate_for(code: &str) -> Option<f64> {
    let c = code.trim().to_uppercase();
    rates().into_iter().find(|(k, _)| *k == c).map(|(_, v)| v)
}

fn stock_price(ticker: &str) -> Option<f64> {
    match ticker.trim().to_uppercase().as_str() {
        "AAPL" => Some(185.25),
        "MSFT" => Some(410.50),
        "GOOG" => Some(175.10),
        "RIVR" => Some(12.45),
        _ => None,
    }
}

fn get_stock_price_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "ticker");
    match stock_price(&t) {
        Some(p) => format!("{}: {}", t.trim().to_uppercase(), fmt_2(p)),
        None => format!("error: unknown ticker '{}'", t),
    }
}

fn convert_currency_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let amount = get_f64(&v, "amount");
    let from = get_str(&v, "from");
    let to = get_str(&v, "to");
    match (rate_for(&from), rate_for(&to)) {
        (Some(r_from), Some(r_to)) => {
            let result = amount / r_from * r_to;
            format!(
                "{} {} = {} {}",
                fmt_2(amount),
                from.trim().to_uppercase(),
                fmt_2(result),
                to.trim().to_uppercase()
            )
        }
        _ => format!("error: unknown currency '{}' or '{}'", from, to),
    }
}

fn compound_growth_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let principal = get_f64(&v, "principal");
    let rate_pct = get_f64(&v, "annual_rate_pct");
    let years = get_f64(&v, "years");
    let result = principal * (1.0 + rate_pct / 100.0).powf(years);
    format!(
        "{} grows to {} after {} years",
        fmt_num(principal),
        fmt_2(result),
        fmt_num(years)
    )
}

fn loan_payment_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let principal = get_f64(&v, "principal");
    let rate_pct = get_f64(&v, "annual_rate_pct");
    let years = get_f64(&v, "years");
    let r = rate_pct / 1200.0;
    let n = years * 12.0;
    let payment = if r == 0.0 {
        principal / n
    } else {
        principal * r / (1.0 - (1.0 + r).powf(-n))
    };
    format!(
        "monthly payment for a {} loan at {}% over {} years: {}",
        fmt_num(principal),
        fmt_num(rate_pct),
        fmt_num(years),
        fmt_2(payment)
    )
}

fn tax_bracket_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let salary = get_f64(&v, "salary");
    let (bracket, lo, hi) = if salary <= 15000.0 {
        ("0%", None, Some(15000.0))
    } else if salary <= 50000.0 {
        ("10%", Some(15000.0), Some(50000.0))
    } else if salary <= 120000.0 {
        ("20%", Some(50000.0), Some(120000.0))
    } else {
        ("30%", Some(120000.0), None)
    };
    let range = match (lo, hi) {
        (Some(l), Some(h)) => format!("income over {} up to {}", fmt_num(l), fmt_num(h)),
        (Some(l), None) => format!("income over {}", fmt_num(l)),
        (None, Some(h)) => format!("income up to {}", fmt_num(h)),
        (None, None) => String::new(),
    };
    format!(
        "the salary {} is in the {} tax bracket ({})",
        fmt_num(salary),
        bracket,
        range
    )
}

fn net_worth_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let assets = get_f64(&v, "assets");
    let liabilities = get_f64(&v, "liabilities");
    fmt_2(assets - liabilities)
}

fn parse_ledger_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let text = get_str(&v, "text");
    let mut sum = 0.0;
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let mut sign = 1.0;
        let mut digits = String::new();
        for (i, c) in tok.chars().enumerate() {
            if i == 0 && c == '-' {
                sign = -1.0;
                continue;
            }
            if i == 0 && c == '+' {
                continue;
            }
            if c.is_ascii_digit() || c == '.' {
                digits.push(c);
            } else {
                break;
            }
        }
        if let Ok(n) = digits.parse::<f64>() {
            sum += sign * n;
        }
    }
    fmt_2(sum)
}

fn init_backend(cfg: &AppConfig) -> Result<Nvidia, llms::LlmError> {
    let mut o = Nvidia::init(
        Some(cfg.system_prompt.clone()),
        Some(cfg.url.clone()),
        llms::minimax::NvidiaConfig {
            api_key: std::env::var("API_KEY").expect("API key not found"),
            model: cfg.model.clone(),
        },
    )?;

    o.register_tool(
        "get_stock_price_tool".to_string(),
        "Returns the current price of a stock ticker (AAPL, MSFT, GOOG, RIVR). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "ticker": { "type": "string", "description": "Stock ticker symbol, e.g. AAPL" }
            },
            "required": ["ticker"]
        }),
        Box::new(get_stock_price_tool),
    )?;

    o.register_tool(
        "convert_currency_tool".to_string(),
        "Converts an amount from one currency to another using fixed rates (USD, EUR, GBP, JPY, INR). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number", "description": "Amount to convert" },
                "from": { "type": "string", "description": "Source currency code, e.g. USD" },
                "to": { "type": "string", "description": "Target currency code, e.g. EUR" }
            },
            "required": ["amount", "from", "to"]
        }),
        Box::new(convert_currency_tool),
    )?;

    o.register_tool(
        "compound_growth_tool".to_string(),
        "Computes the future value of a principal growing at an annual percentage rate over a number of years. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "principal": { "type": "number", "description": "Starting amount" },
                "annual_rate_pct": { "type": "number", "description": "Annual growth rate in percent" },
                "years": { "type": "number", "description": "Number of years" }
            },
            "required": ["principal", "annual_rate_pct", "years"]
        }),
        Box::new(compound_growth_tool),
    )?;

    o.register_tool(
        "loan_payment_tool".to_string(),
        "Computes the fixed monthly payment for a loan given principal, annual interest rate in percent and years. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "principal": { "type": "number", "description": "Loan amount" },
                "annual_rate_pct": { "type": "number", "description": "Annual interest rate in percent" },
                "years": { "type": "number", "description": "Loan term in years" }
            },
            "required": ["principal", "annual_rate_pct", "years"]
        }),
        Box::new(loan_payment_tool),
    )?;

    o.register_tool(
        "tax_bracket_tool".to_string(),
        "Returns the tax bracket (0%, 10%, 20%, 30%) that a given salary falls into. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "salary": { "type": "number", "description": "Annual salary" }
            },
            "required": ["salary"]
        }),
        Box::new(tax_bracket_tool),
    )?;

    o.register_tool(
        "net_worth_tool".to_string(),
        "Computes net worth as assets minus liabilities. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "assets": { "type": "number", "description": "Total assets" },
                "liabilities": { "type": "number", "description": "Total liabilities" }
            },
            "required": ["assets", "liabilities"]
        }),
        Box::new(net_worth_tool),
    )?;

    o.register_tool(
        "parse_ledger_tool".to_string(),
        "Sums all signed monetary amounts appearing in a ledger text (e.g. '+2000, -4.50'). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Ledger text containing signed amounts" }
            },
            "required": ["text"]
        }),
        Box::new(parse_ledger_tool),
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
