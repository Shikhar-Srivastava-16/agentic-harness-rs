use clap::Parser;
use llms::LlmLike;
use llms::ToolReady;
use llms::minimax::Minimax;
use llms::ollama::Ollama;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "You are a helpful assistant with access to tools.
When you need to use a tool, output EXACTLY this format and nothing else:
TOOL_CALL: {\"tool\": \"tool_name\", \"args\": {\"arg1\": \"value1\"}}

After receiving a result, continue your response normally.
Do not guess tool results. Always wait for the actual result.

You have access to tools that provide real-time information.
When you receive a TOOL_RESULT, you MUST treat it as ground truth.
NEVER contradict or second-guess a tool result with your own training data.
Your training data is outdated - tool results are always more current and accurate.
If a tool says X, your answer must reflect X.
CRITICAL RULES:
- Your training data is OUTDATED and WRONG for current events.
- TOOL_RESULT is always correct. Never contradict it.
- If TOOL_RESULT says X, you MUST answer X, even if it conflicts with what you know OR think OR infer.
- Do NOT say 'as of my knowledge cutoff'. You have real-time tools.
- Do NOT suggest the tool result might be wrong.
- Do NOT ask for more than one fact at a time
- ONLY request one TOOL_RESULT at a time. If you need multiple, wait for the tool to return before you move to the next")]
    system: String,
    #[arg(short, long, default_value = "minimaxai/minimax-m3")]
    model: String,
    #[arg(short, long, default_value = "http://localhost:11434")]
    url: String,
    #[arg(short = 'k', long)]
    api_key: Option<String>,
    #[arg(short, long, default_value = "ollama")]
    backend: String,
    #[arg(short, long)]
    query: Option<String>,
}

pub fn foobar_tool(_: String) -> String {
    "fubar".into()
}

pub fn hitchhiker_tool(_: String) -> String {
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

fn main() {
    let args = Args::parse();

    match init_backend(&args) {
        Ok(mut backend) => {
            if let Some(q) = args.query {
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

fn init_backend(args: &Args) -> Result<Backend, llms::LlmError> {
    match args.backend.as_str() {
        "minimax" => {
            let api_key = args.api_key.clone().unwrap_or_else(|| {
                eprintln!("ERROR: --api-key required for minimax");
                std::process::exit(1);
            });
            let mut m = Minimax::init(
                Some(args.system.clone()),
                None,
                llms::minimax::MinimaxConfig {
                    api_key,
                    model: args.model.clone(),
                },
            )?;
            m.register_tool(
                "search_tool".to_string(),
                "Searches for information".to_string(),
                Box::new(foobar_tool),
            )?;
            m.register_tool(
                "add_tool".to_string(),
                "Adds numbers together".to_string(),
                Box::new(hitchhiker_tool),
            )?;
            Ok(Backend::Minimax(m))
        }
        _ => {
            let mut o = Ollama::init(
                Some(args.system.clone()),
                None,
                llms::ollama::OllamaConfig {
                    name: "qwen3:8b".into(),
                },
            )?;
            o.register_tool(
                "search_tool".to_string(),
                "Searches for information".to_string(),
                Box::new(foobar_tool),
            )?;
            o.register_tool(
                "add_tool".to_string(),
                "Adds numbers together".to_string(),
                Box::new(hitchhiker_tool),
            )?;
            Ok(Backend::Ollama(o))
        }
    }
}
