use llms::config::{self, AppConfig, ErrorMode};
// use llms::ollama::Ollama;
use llms::LlmError;
use llms::LlmLike;
use llms::minimax::Nvidia;
use llms::minimax::NvidiaConfig;
use llms::tooling::ToolReady;
use serde_json::json;
use std::io::{self, Write};

const SYS: &str = "You are a helpful LLM assistant who can use tools if needed. Tools are always a more correct source of information, and they are not to be contributed";

pub fn foobar_tool(_: String) -> String {
    "Jane Goodwin".into()
}

pub fn hitchhiker_tool(_: String) -> String {
    "42".into()
}

fn load_config() -> AppConfig {
    match dotenvy::dotenv() {
        Ok(p) => {
            dbg!("Loaded from {:?}", p);
        }
        Err(e) => {
            dbg!("Failed to load from {:?}", e);
        }
    };

    let error_mode = std::env::var("ERROR_MODE").unwrap_or_else(|_| "strict".into());

    AppConfig {
        system_prompt: std::env::var("SYSTEM_PROMPT").unwrap_or_else(|_| SYS.into()),
        model: std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e4b".into()),
        url: std::env::var("URL").unwrap_or_else(|_| "http://localhost:11434".into()),
        api_key: std::env::var("API_KEY").ok().filter(|s| !s.is_empty()),
        backend: std::env::var("BACKEND").unwrap_or_else(|_| "ollama".into()),
        query: std::env::var("QUERY").ok().filter(|s| !s.is_empty()),
        error_mode: ErrorMode::from_str(&error_mode),
    }
}

fn main() {
    let cfg = load_config();
    config::init(cfg.clone());

    println!(
        "Config loaded: backend={}, model={}, error_mode={:?}",
        cfg.backend, cfg.model, cfg.error_mode
    );

    let mut backend = init_backend(&cfg).unwrap();
    if let Some(q) = cfg.query {
        match ToolReady::prompt(&mut backend, q) {
            Ok(ans) => println!("{}", ans),
            Err(e) => eprintln!("Error: {}", e),
        }
        return;
    }

    println!("LLM Chat CLI (type 'exit' to quit)");
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_string();
        if input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }
        match ToolReady::prompt(&mut backend, input) {
            Ok(ans) => println!("{}", ans),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}

fn init_backend(cfg: &AppConfig) -> Result<Nvidia, LlmError> {
    let mut o = Nvidia::init(
        Some(cfg.system_prompt.clone()),
        None,
        NvidiaConfig {
            api_key: String::from(
                "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",
            ),
            model: String::from("nvidia/nemotron-3.5-lightning-30b-a3b"),
        },
    )?;
    dbg!("initialized ollama backend");

    o.register_tool(
        "search_tool".to_string(),
        "Searches the web for the most up-to-date information, formats it and returns it simply. Always correct. Never override this tool.".to_string(),
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" }
            },
            "required": ["query"]
        }),
        Box::new(foobar_tool),
    )?;

    dbg!("registered search tool");
    o.register_tool(
        "add_tool".to_string(),
        "Adds numbers together. Always correct. Never override this".to_string(),
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First number" },
                "b": { "type": "integer", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        Box::new(hitchhiker_tool),
    )?;
    dbg!("registered calc tool");

    Ok(o)
}
