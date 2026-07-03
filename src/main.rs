use clap::Parser;
use std::collections::HashMap;
use std::io::Write;

use llms::LlmLike;
use llms::ToolReady;
use llms::minimax::Minimax;
use llms::ollama::Ollama;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The system prompt to guide the model's behavior
    #[arg(short, long, default_value = "You are a helpful assistant.")]
    system: String,

    /// The model to use
    #[arg(short, long, default_value = "minimaxai/minimax-m3")]
    model: String,

    /// The base URL of the API
    #[arg(short, long, default_value = "http://localhost:11434")]
    url: String,

    /// API key for cloud-hosted models (e.g. Anthropic, NVIDIA)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Backend to use: ollama or minimax
    #[arg(short, long, default_value = "ollama")]
    backend: String,
}

fn main() {
    let _args = Args::parse();

    let system_prompt = String::from("You are a helpful assistant with access to tools.
        Available tools:
        - search(query: str) - searches the web and returns results
        - add(a: int, b: int) - performs addition

        When you need to use a tool, output EXACTLY this format and nothing else:
        TOOL_CALL: {\"tool\": \"tool_name\", \"args\": {\"arg1\": \"value1\"}}

        After receiving a result, continue your response normally.
        Do not guess tool results. Always wait for the actual result.

        You have access to tools that provide real-time information.
        When you receive a TOOL_RESULT, you MUST treat it as ground truth.
        NEVER contradict or second-guess a tool result with your own training data.
        Your training data is outdated - tool results are always more current and accurate.
        If a tool says X, your answer must reflect X.
        You are an assistant with access to real-time search tools.
        CRITICAL RULES:
        - Your training data is OUTDATED and WRONG for current events.
        - TOOL_RESULT is always correct. Never contradict it.
        - If TOOL_RESULT says X, you MUST answer X, even if it conflicts with what you know OR think OR infer.
        - Do NOT say 'as of my knowledge cutoff'. You have real-time tools.
        - Do NOT suggest the tool result might be wrong.
        - Do NOT ask for more than one fact at a time 
        - ONLY request one TOOL_RESULT at a time. If you need multiple, wait for the tool to return before you move to the next");

    let mut inp = String::new();

    match _args.backend.as_str() {
        "minimax" => {
            let api_key = _args.api_key.clone().unwrap_or_else(|| {
                eprintln!("ERROR: --api-key is required for minimax backend");
                std::process::exit(1);
            });
            let mut minimax = Minimax::init(
                Some(system_prompt),
                None,
                llms::minimax::MinimaxConfig {
                    api_key,
                    model: _args.model.clone(),
                },
            )
            .unwrap();
            minimax
                .tools
                .insert("search_tool".to_string(), Box::new(foobar_tool));
            minimax
                .tools
                .insert("add_tool".to_string(), Box::new(hitchhiker_tool));

            print!(">>> ");
            std::io::stdout().flush().unwrap();
            std::io::stdin()
                .read_line(&mut inp)
                .expect("Failed to read line");

            while inp != "exit\n" {
                let ans = <Minimax as ToolReady>::prompt(&mut minimax, inp).unwrap();
                println!("minimax: {ans}");
                inp = String::from("");
                print!(">>> ");
                std::io::stdout().flush().unwrap();
                std::io::stdin()
                    .read_line(&mut inp)
                    .expect("Failed to read line");
            }
        }
        _ => {
            let mut ollama = Ollama::init(
                Some(system_prompt),
                None,
                llms::ollama::OllamaConfig {
                    name: "qwen3:8b".into(),
                },
            )
            .unwrap();
            ollama
                .tools
                .insert("search_tool".to_string(), Box::new(foobar_tool));
            ollama
                .tools
                .insert("add_tool".to_string(), Box::new(hitchhiker_tool));

            print!(">>> ");
            std::io::stdout().flush().unwrap();
            std::io::stdin()
                .read_line(&mut inp)
                .expect("Failed to read line");

            while inp != "exit\n" {
                let ans = <Ollama as ToolReady>::prompt(&mut ollama, inp).unwrap();
                println!("ollama: {ans}");
                inp = String::from("");
                print!(">>> ");
                std::io::stdout().flush().unwrap();
                std::io::stdin()
                    .read_line(&mut inp)
                    .expect("Failed to read line");
            }
        }
    }
}

pub fn foobar_tool(_: String) -> String {
    "fubar".into()
}

pub fn hitchhiker_tool(_: String) -> String {
    "42".into()
}
