use llms::LlmLike;
use llms::LlmError;
use llms::tooling::ToolReady;
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

fn load_config() -> AppConfig {
    dotenvy::dotenv().ok();

    let error_mode = std::env::var("ERROR_MODE")
        .unwrap_or_else(|_| "strict".into());

    AppConfig {
        system_prompt: std::env::var("SYSTEM_PROMPT")
            .unwrap(),
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
        Err(e) => {
            eprintln!("Failed to initialize backend: {}", e);
            std::process::exit(1);
        }
    }
}

fn init_backend(cfg: &AppConfig) -> Result<Ollama, LlmError> {
    let mut o = Ollama::init(
        Some(cfg.system_prompt.clone()),
        None,
        llms::ollama::OllamaConfig {
            name: cfg.model.clone(),
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
    // <Ollama as ToolReady>::prompt(&mut o, input)
}
