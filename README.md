# Ollama Rust Client

A simple Rust CLI application to interact with the Ollama REST API.

## Features

- Supports custom **System Prompts**.
- Supports custom **User Queries**.
- Configurable **Model** and **API URL**.
- Built with `tokio`, `reqwest`, `serde`, and `clap`.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) installed.
- [Ollama](https://ollama.com/) running locally or accessible via network.

## Installation

1. Clone or copy the source code.
2. Build the project:
   ```bash
   cargo build --release
   ```

## Usage

Run the application using `cargo run`:

```bash
cargo run -- --query "Explain quantum entanglement" --system "You are a theoretical physicist."
```

### Options

- `-q, --query <QUERY>`: The user query to send.
- `-s, --system <SYSTEM>`: The system prompt (default: "You are a helpful assistant.").
- `-m, --model <MODEL>`: The model to use (default: "llama3").
- `-u, --url <URL>`: The base URL of Ollama (default: "http://localhost:11434").

## Project Structure

- `src/main.rs`: Contains the CLI logic and API interaction.
- `Cargo.toml`: Project dependencies.
