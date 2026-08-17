use llms::LlmLike;
use llms::config::AppConfig;
use llms::config::{self, ErrorMode};
use llms::ollama::Ollama;
use llms::tooling::ToolReady;
use serde_json::json;

fn foobar_tool(_: String) -> String {
    "Jane Goodwin".into()
}

fn hitchhiker_tool(_: String) -> String {
    "42".into()
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

    Ok(o)
}

fn main() {
    let iterations: u32 = std::env::var("BENCH_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let model: String = std::env::var("BENCH_MODEL")
        .unwrap_or_else(|_| "qwen3:8B".to_string());

    let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());

    let cfg = AppConfig {
        system_prompt: "You are a helpful assistant that uses tools when needed.".to_string(),
        model: model.clone(),
        url: ollama_url,
        api_key: None,
        backend: "ollama".to_string(),
        query: None,
        error_mode: ErrorMode::Strict,
    };
    config::init(cfg.clone());

    let mut backend = init_backend(&cfg).expect("failed to init backend");

    let prompts = [
        "Who is Jane Goodwin?",
        "What is the answer to life, the universe, and everything?",
        "Combine search_tool and add_tool to find Jane's lucky number.",
        "Use search_tool to find information about Jane Goodwin.",
        "Use add_tool to compute 42 + 0.",
        "What did search_tool tell you about Jane Goodwin?",
        "Calculate 42 using the add_tool.",
        "Tell me a story about Jane Goodwin found via search_tool.",
        "Using add_tool, what is 40 + 2?",
        "Query search_tool for Jane Goodwin's details.",
    ];

    println!("Running {} iterations with model '{}'", iterations, model);
    for i in 0..iterations as usize {
        let prompt = prompts[i % prompts.len()].to_string();
        print!("  [{}/{}] prompt: \"{}\" ... ", i + 1, iterations, prompt);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match ToolReady::prompt(&mut backend, prompt) {
            Ok(ans) => println!("OK ({} chars)", ans.len()),
            Err(e) => eprintln!("ERROR: {}", e),
        }
    }

    println!("Done. Results written to bench output file.");
    llms::bench::finalize();
}
