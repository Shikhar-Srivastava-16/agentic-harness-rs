use clap::Parser;
use std::error::Error;

use llms::LlmLike;
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
    mistral.set_sys_prompt(args.system);
    let mut inp = String::new();
    while !(inp == "exit") {
        std::io::stdin()
            .read_line(&mut inp)
            .expect("Failed to read line");

        let ans = mistral.prompt(inp).unwrap();

        println!("mistral: {ans}");
        inp = String::from("");
    }
}
