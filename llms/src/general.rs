use crate::tooling::ToolDef;
use serde::{Deserialize, Serialize};

/// A single turn in a conversation — role, content, and optional tool calls.
#[derive(Default, Debug, Clone)]
pub struct Message {
    /// The message role (e.g. `"user"`, `"assistant"`, `"system"`, `"tool"`), to indicate what this message is meant to do
    pub role: String,
    /// The message content (`None` for tool-only messages).
    pub content: Option<String>,
    /// Tool calls associated with this message (present only on assistant messages).
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The name of the tool called (present only on tool messages).
    pub tool_name: Option<String>,
}

/// Unified request body used across all LLM backends.
#[derive(Default, Debug)]
pub struct ChatRequest {
    /// The model identifier.
    pub model: String,
    /// The conversation history.
    pub messages: Vec<Message>,
    /// Optional tool/function definitions the model can use.
    pub tools: Option<Vec<ToolDef>>,
    /// Fixed seed for deterministic outputs.
    // format is in schemas but we only support JSON, so not needed
    pub seed: Option<usize>,
    /// Sampling temperature (higher = more random, lower = more focused).
    pub temperature: Option<f32>,
    /// Nucleus sampling probability threshold.
    pub top_p: Option<f32>,
    /// Limits sampling to the top k tokens.
    pub top_k: Option<usize>,
    /// Minimum probability threshold for sampling.
    pub min_p: Option<f32>,
    /// Stop sequences to end generation.
    pub stop: Option<String>,
    /// Maximum context window size in characters.
    pub max_ctx_chars: Option<usize>,
    /// Maximum number of tokens to predict.
    pub max_prediction_tokens: Option<usize>,

    /// Whether to stream the response.
    pub stream: Option<bool>,
    /// Boolean-ish (`"true"` / `"false"`) or a thinking level (`"high"`, `"medium"`, `"low"`, `"max"`).
    pub think: Option<String>,
    /// Keep-alive duration, e.g. `"5m"` or `"0"` to unload immediately.
    pub keep_alive: Option<String>,
    // TODO: Add logging token probabilities
    // pub lobprobs: boolean,
    // pub top_logprobs: usize
}

/// Unified response body returned by all LLM backends.
#[derive(Debug)]
pub struct ChatResponse {
    /// The model that generated the response.
    pub model: String,
    /// Timestamp of the response generation.
    pub created_at: String,
    /// The assistant's response message.
    pub message: Message,
    /// Whether the response is complete.
    pub done: bool,
    /// Whether the response finished with a stop reason (`true` = `"stop"`).
    pub done_reason: bool,
    /// Total duration in nanoseconds.
    pub total_duration: usize,
    /// Time spent loading the model in nanoseconds.
    pub load_duration: usize,
    /// Number of tokens evaluated in the prompt.
    pub prompt_eval_count: usize,
    /// Duration of prompt evaluation in nanoseconds.
    pub prompt_eval_duration: usize,
    /// Number of tokens evaluated in generation.
    pub eval_count: usize,

    /// Tool calls made by the model (if any).
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Represents a tool call with an optional type and function payload.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// The tool type (e.g. `"function"`); `None` for unified format.
    pub tool_type: Option<String>,
    /// The function name and arguments.
    pub function: FunctionCall,
}

/// The function name and arguments pair.
#[derive(Debug, Clone)]
pub struct FunctionCall {
    /// The function name.
    pub name: String,
    /// JSON object of arguments to pass to the function.
    pub arguments: serde_json::Value,
}
