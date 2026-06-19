use clap::Parser;
use std::error::Error;

use llms;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let client = reqwest::Client::new();
    let api_url = format!("{}/api/chat", args.url.trim_end_matches('/'));

    let messages = vec![
        llms::general::Message {
            role: "system".to_string(),
            content: args.system,
        },
        llms::general::Message {
            role: "user".to_string(),
            content: args.query,
        },
    ];

    let request_body = llms::general::ChatRequest {
        model: args.model,
        messages,
        stream: false,
    };

    println!("Connecting to Ollama at {}...", api_url);

    let response = client.post(&api_url).json(&request_body).send().await?;

    if response.status().is_success() {
        let chat_response: llms::general::ChatResponse = response.json().await?;
        println!("\nAssistant: {}", chat_response.message.content);
    } else {
        let error_text = response.text().await?;
        eprintln!("Error from Ollama API: {}", error_text);
    }

    Ok(())
}
