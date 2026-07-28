use crate::tooling::ToolDef;
use serde::{Deserialize, Serialize};

// Request Body
#[derive(Default, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_name: Option<String>,
}

#[derive(Default, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ToolDef>>,
    // format is in schemas but we only support JSON, so not needed
    pub seed: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub stop: Option<String>,
    pub max_ctx_chars: Option<usize>,
    pub max_prediction_tokens: Option<usize>,

    pub stream: Option<bool>,
    pub think: Option<String>,
    pub keep_alive: Option<String>,
    // TODO: Add logging token probabilities
    // pub lobprobs: boolean,
    // pub top_logprobs: usize
}

// responses

#[derive(Debug)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String, // date-time
    pub message: Message,
    pub done: bool,
    pub done_reason: bool,
    pub total_duration: usize, // in nanoseconds
    pub load_duration: usize,
    pub prompt_eval_count: usize,
    pub prompt_eval_duration: usize,
    pub eval_count: usize,

    // TODO: Add logging token probabilities
    // logprobs: Vec<TokProb>
    pub tool_calls: Option<Vec<ToolCall>>, // TODO: check correct type?
}

// for ollama only
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}
