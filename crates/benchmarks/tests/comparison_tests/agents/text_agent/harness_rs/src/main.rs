use benchmarks::test_config::HarnessConfig;
use llms::LlmLike;
use llms::config::AppConfig;
use llms::config::{self, ErrorMode};
use llms::ollama::Ollama;
use llms::tooling::ToolReady;
use serde_json::{Value, json};
use std::path::Path;

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn get_str(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_ALPHABET[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn b64_val(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a' + 26) as u32),
        b'0'..=b'9' => Some((c - b'0' + 52) as u32),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let clean: Vec<u32> = s
        .bytes()
        .filter(|&b| b != b'=')
        .map(b64_val)
        .collect::<Option<Vec<u32>>>()?;
    let mut out = Vec::new();
    for chunk in clean.chunks(4) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &v)| acc | (v << (18 - 6 * i)));
        out.push((n >> 16) as u8);
        if chunk.len() >= 3 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() == 4 {
            out.push(n as u8);
        }
    }
    Some(out)
}

fn reverse_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    get_str(&v, "text").chars().rev().collect()
}

fn word_count_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    get_str(&v, "text").split_whitespace().count().to_string()
}

fn char_count_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    get_str(&v, "text")
        .chars()
        .filter(|c| !c.is_whitespace())
        .count()
        .to_string()
}

fn strip_wrapped(tok: &str) -> &str {
    let mut s = tok;
    while let Some(stripped) = s.strip_prefix(|c: char| !c.is_alphanumeric()) {
        s = stripped;
    }
    while let Some(stripped) = s.strip_suffix(|c: char| !c.is_alphanumeric()) {
        s = stripped;
    }
    s
}

fn extract_emails_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let text = get_str(&v, "text");
    let emails: Vec<&str> = text
        .split_whitespace()
        .map(strip_wrapped)
        .filter(|tok| tok.contains('@') && tok.contains('.') && tok.find('@').unwrap_or(0) > 0)
        .collect();
    if emails.is_empty() {
        return "no emails found".to_string();
    }
    emails.join(", ")
}

fn is_palindrome_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let cleaned: String = get_str(&v, "text")
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();
    if cleaned.is_empty() {
        return "true".to_string();
    }
    (cleaned == cleaned.chars().rev().collect::<String>()).to_string()
}

fn is_anagram_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let mut a: Vec<char> = get_str(&v, "text_a")
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect();
    let mut b: Vec<char> = get_str(&v, "text_b")
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect();
    a.sort_unstable();
    b.sort_unstable();
    (a == b).to_string()
}

fn base64_encode_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    b64_encode(get_str(&v, "text").as_bytes())
}

fn base64_decode_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    match b64_decode(&get_str(&v, "encoded")) {
        Some(bytes) => String::from_utf8(bytes).unwrap_or_else(|_| "error: invalid utf8".to_string()),
        None => "error: invalid base64".to_string(),
    }
}

fn slugify_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let mut out = String::new();
    let mut last_was_sep = true;
    for c in get_str(&v, "text").to_lowercase().chars() {
        if c.is_alphanumeric() {
            out.push(c);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('-');
            last_was_sep = true;
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "-".to_string()
    } else {
        out
    }
}

fn camel_to_snake_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let mut out = String::new();
    for (i, c) in get_str(&v, "text").chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn init_backend(cfg: &AppConfig) -> Result<Ollama, llms::LlmError> {
    let mut o = Ollama::init(
        Some(cfg.system_prompt.clone()),
        Some(cfg.url.clone()),
        llms::ollama::OllamaConfig {
            name: cfg.model.clone(),
        },
    )?;

    o.register_tool(
        "reverse_tool".to_string(),
        "Reverses the characters of a string. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "The string to reverse" }
            },
            "required": ["text"]
        }),
        Box::new(reverse_tool),
    )?;

    o.register_tool(
        "word_count_tool".to_string(),
        "Counts the number of words in a string. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "The string to count words in" }
            },
            "required": ["text"]
        }),
        Box::new(word_count_tool),
    )?;

    o.register_tool(
        "char_count_tool".to_string(),
        "Counts the number of characters in a string, excluding whitespace. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "The string to count characters in" }
            },
            "required": ["text"]
        }),
        Box::new(char_count_tool),
    )?;

    o.register_tool(
        "extract_emails_tool".to_string(),
        "Extracts all email addresses from a text and returns them as a comma-separated list. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to extract emails from" }
            },
            "required": ["text"]
        }),
        Box::new(extract_emails_tool),
    )?;

    o.register_tool(
        "is_palindrome_tool".to_string(),
        "Returns true or false indicating whether the given text is a palindrome (ignoring spaces, punctuation and case). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to check" }
            },
            "required": ["text"]
        }),
        Box::new(is_palindrome_tool),
    )?;

    o.register_tool(
        "is_anagram_tool".to_string(),
        "Returns true or false indicating whether two texts are anagrams of each other (ignoring spaces and case). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text_a": { "type": "string", "description": "First text" },
                "text_b": { "type": "string", "description": "Second text" }
            },
            "required": ["text_a", "text_b"]
        }),
        Box::new(is_anagram_tool),
    )?;

    o.register_tool(
        "base64_encode_tool".to_string(),
        "Encodes a string as base64. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to base64 encode" }
            },
            "required": ["text"]
        }),
        Box::new(base64_encode_tool),
    )?;

    o.register_tool(
        "base64_decode_tool".to_string(),
        "Decodes a base64 string back into plain text. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "encoded": { "type": "string", "description": "Base64 string to decode" }
            },
            "required": ["encoded"]
        }),
        Box::new(base64_decode_tool),
    )?;

    o.register_tool(
        "slugify_tool".to_string(),
        "Converts text into a URL-friendly slug (lowercase, non-alphanumerics replaced by hyphens). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to slugify" }
            },
            "required": ["text"]
        }),
        Box::new(slugify_tool),
    )?;

    o.register_tool(
        "camel_to_snake_tool".to_string(),
        "Converts a camelCase string into snake_case. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "camelCase string to convert" }
            },
            "required": ["text"]
        }),
        Box::new(camel_to_snake_tool),
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
        api_key: cfg.api_key.clone(),
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
            Ok(ans) => println!("OK ({} chars)", ans.len()),
            Err(e) => panic!("ERROR: {}", e),
        }
    }

    println!("Done. Results written to bench output file.");
    llms::bench::finalize();
}