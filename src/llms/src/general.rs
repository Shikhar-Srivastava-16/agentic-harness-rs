use crate::tooling;
use serde::{Deserialize, Serialize};

// Request Body
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    // format is in schemas but we only support JSON, so not needed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_ctx_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_prediction_tokens: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,

    // TODO: Add logging token probabilities
    // pub lobprobs: boolean,
    // pub top_logprobs: usize

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<tooling::ToolDef>>,

}

#[derive(Deserialize, Debug)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub spec: ChatToolSpec
}

#[derive(Deserialize, Debug)]
pub struct ChatToolSpec {
    pub tool_type: String,
    pub parameters: std::collections::HashMap<String, ChatToolParamSpec>
}

#[derive(Deserialize, Debug)]
pub struct ChatToolParamSpec {
    pub param_type: String,
    pub param_description: String
}

// response
#[derive(Deserialize, Debug)]
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

    #[serde(default)]
    pub tool_calls: Option<Vec<ChatTool>>,
}

// for ollama only
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}
