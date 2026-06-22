use clap::Parser;
use std::error::Error;

use llms::LlmLike;
use llms::ollama::Ollama;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The user query to send to Ollama
    #[arg(short, long)]
    query: String,

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

    mistral.set_sys_prompt(args.system);

    let ans = mistral.prompt(args.query).unwrap();

    println!("mistral: {ans}");
}
