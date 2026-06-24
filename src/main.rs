use clap::Parser;
use std::error::Error;

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
    let args = Args::parse();

    // requesting start here
    let mut mistral = Ollama::default();
    mistral.set_sys_prompt(String::from(
        "You are a helpful assistant with access to tools.
        Available tools:
        - search_web(query: str) — searches the web and returns results

        When you need to use a tool, output EXACTLY this format and nothing else:
        TOOL_CALL: {\"tool\": \"tool_name\", \"args\": {\"arg1\": \"value1\"}}

        After receiving a result, continue your response normally.
        Do not guess tool results. Always wait for the actual result.

        You have access to tools that provide real-time information.
        When you receive a TOOL_RESULT, you MUST treat it as ground truth.
        NEVER contradict or second-guess a tool result with your own training data.
        Your training data is outdated - tool results are always more current and accurate.
        If a tool says X, your answer must reflect X.
    ",
    ));
    let mut inp = String::new();
    while !(inp == "exit") {
        std::io::stdin()
            .read_line(&mut inp)
            .expect("Failed to read line");

        let ans = prompt_wrap(&mut mistral, inp);

        println!("mistral: {ans}");
        inp = String::from("");
    }
}

fn prompt_wrap(mistral: &mut Ollama, a: String) -> String {
    let ans = <Ollama as LlmLike>::prompt(mistral, a).unwrap();
    let ans = ans.trim();
    eprintln!("DEBUG: LLM raw response: {:?}", ans);
    if ans.starts_with("TOOL_CALL") {
        println!("tool: {}", ans);
        let mut mock_out = String::new();
        print!("Tool Ans: ");
        std::io::stdin()
            .read_line(&mut mock_out)
            .expect("Failed to read line");

        mistral.tool_respond(mock_out).unwrap()
    } else {
        ans.to_string()
    }
}
