use llms::LlmLike;
use llms::ToolReady;
use llms::minimax::Minimax;
use llms::ollama::Ollama;
use llms::config::{self, AppConfig, ErrorMode};
use serde_json::json;
use std::io::{self, Write};

pub fn foobar_tool(_: String) -> String {
    println!("\n\n====== using foobar tool (search) ========\n\n");
    "Kamala Harris".into()
}

pub fn hitchhiker_tool(_: String) -> String {
    println!("\n\n====== using hitchhiker tool (calc) ========\n\n");
    "42".into()
}

enum Backend {
    Ollama(Ollama),
    Minimax(Minimax),
}

impl Backend {
    fn prompt(&mut self, input: String) -> Result<String, llms::LlmError> {
        match self {
            Backend::Ollama(o) => <Ollama as ToolReady>::prompt(o, input),
            Backend::Minimax(m) => <Minimax as ToolReady>::prompt(m, input),
        }
    }
}

fn load_config() -> AppConfig {
    dotenvy::dotenv().ok();

    let error_mode = std::env::var("ERROR_MODE")
        .unwrap_or_else(|_| "strict".into());

    AppConfig {
        system_prompt: std::env::var("SYSTEM_PROMPT")
            .unwrap(),
            //.unwrap_or_else(|_| "You are a helpful assistant with access to tools. Use them when appropriate.".into()),
        model: std::env::var("MODEL")
            .unwrap_or_else(|_| "minimaxai/minimax-m3".into()),
        url: std::env::var("URL")
            .unwrap_or_else(|_| "http://localhost:11434".into()),
        api_key: std::env::var("API_KEY").ok().filter(|s| !s.is_empty()),
        backend: std::env::var("BACKEND")
            .unwrap_or_else(|_| "ollama".into()),
        query: std::env::var("QUERY").ok().filter(|s| !s.is_empty()),
        error_mode: ErrorMode::from_str(&error_mode),
    }
}

fn main() {
    let cfg = load_config();
    config::init(cfg.clone());

    println!("Config loaded: backend={}, model={}, error_mode={:?}",
        cfg.backend, cfg.model, cfg.error_mode);

    match init_backend(&cfg) {
        Ok(mut backend) => {
            if let Some(q) = cfg.query {
                match backend.prompt(q) {
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
                match backend.prompt(input) {
                    Ok(ans) => println!("{}", ans),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to initialize backend: {}", e);
            std::process::exit(1);
        }
    }
}

fn init_backend(cfg: &AppConfig) -> Result<Backend, llms::LlmError> {
    match cfg.backend.as_str() {
        "minimax" => {
            let api_key = cfg.api_key.clone().unwrap_or_else(|| {
                eprintln!("ERROR: API_KEY required for minimax backend. Set it in .env");
                std::process::exit(1);
            });
            let mut m = Minimax::init(
                Some(cfg.system_prompt.clone()),
                None,
                llms::minimax::MinimaxConfig {
                    api_key,
                    model: cfg.model.clone(),
                },
            )?;
            m.register_tool(
                "search_tool".to_string(),
                "Searches for information".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "The search query" }
                    },
                    "required": ["query"]
                }),
                Box::new(foobar_tool),
            )?;
            m.register_tool(
                "add_tool".to_string(),
                "Adds numbers together".to_string(),
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
            Ok(Backend::Minimax(m))
        }
        _ => {
            let mut o = Ollama::init(
                Some(cfg.system_prompt.clone()),
                None,
                llms::ollama::OllamaConfig {
                    name: cfg.model.clone(),
                },
            )?;
            o.register_tool(
                "search_tool".to_string(),
                "Searches for information".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "The search query" }
                    },
                    "required": ["query"]
                }),
                Box::new(foobar_tool),
            )?;
            o.register_tool(
                "add_tool".to_string(),
                "Adds numbers together".to_string(),
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
            Ok(Backend::Ollama(o))
        }
    }
}
