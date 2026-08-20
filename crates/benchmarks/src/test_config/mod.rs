use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct HarnessConfig {
    pub model: String,
    pub url: String,
    pub iterations: u32,
    pub system_prompt: String,
    pub error_mode: String,
    pub backend: String,
    #[serde(default)]
    pub query: Option<String>,
    pub bench_output: String,
    pub langchain_output: String,
    pub prompts: Vec<String>,
}
