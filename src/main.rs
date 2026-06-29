use clap::Parser;
use std::collections::HashMap;
use std::io::Write;

use llms::LlmLike;
use llms::ToolReady;
use llms::ollama::Ollama;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The system prompt to guide the model's behavior
    #[arg(short, long, default_value = "You are a helpful assistant.")]
    system: String,

    /// The model to use
    #[arg(short, long, default_value = "mistral")]
    model: String,

    /// The base URL of the Ollama API
    #[arg(short, long, default_value = "http://localhost:11434")]
    url: String,
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

    let mut mistral = Ollama::init(
        Some(system_prompt),
        None,
        llms::ollama::OllamaConfig {
            name: "qwen3:8b".into(),
        },
    )
    .unwrap();
    mistral
        .tools
        .insert("search_tool".to_string(), Box::new(foobar_tool));
    mistral
        .tools
        .insert("add_tool".to_string(), Box::new(hitchhiker_tool));

    let mut inp = String::new();

    print!(">>> ");
    std::io::stdout().flush().unwrap();
    std::io::stdin()
        .read_line(&mut inp)
        .expect("Failed to read line");

    while inp != "exit\n" {
        let ans = <Ollama as ToolReady>::prompt(&mut mistral, inp).unwrap();

        println!("mistral: {ans}");
        inp = String::from("");
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin()
            .read_line(&mut inp)
            .expect("Failed to read line");
    }
}

pub fn foobar_tool(_: String) -> String {
    "fubar".into()
}

pub fn hitchhiker_tool(_: String) -> String {
    "42".into()
}
