//! Rust structs for the Ollama Chat API request body
//! (`POST /api/chat`).
//!
//! Design notes:
//!   - All `Option<T>` fields use `skip_serializing_if = "Option::is_none"`
//!     so unset fields aren't sent — Ollama ignores unknown fields but
//!     keeping payloads clean is still good practice.
//!   - `Format`, `Think`, and `KeepAlive` are plain `String` / `serde_json::Value`
//!     because Ollama accepts multiple shapes (e.g. `"json"` or a schema object
//!     for `format`) and modelling that as enums adds complexity for little gain.
//!   - `parameters` and `arguments` are `serde_json::Value` — they carry
//!     arbitrary JSON Schema / JSON objects that don't warrant dedicated types.

use crate::general;
use crate::tooling;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

// ---------------------------------------------------------------------
// Top-level request
// ---------------------------------------------------------------------

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatRequest {
    // --- required ---
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,

    // --- optional fields ---
    #[serde(default)]
    pub tools: Option<Vec<OllamaToolDefinition>>,
    /// `"json"` or a JSON Schema object.
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub options: Option<OllamaModelOptions>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Boolean-ish (`"true"` / `"false"`) or a thinking level
    /// (`"high"`, `"medium"`, `"low"`, `"max"`).
    #[serde(default)]
    pub think: Option<String>,
    /// Keep-alive duration, e.g. `"5m"` or `"0"` to unload immediately.
    #[serde(default)]
    pub keep_alive: Option<String>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
}

// ---------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    /// One of `"system"`, `"user"`, `"assistant"`, `"tool"`.
    pub role: String,
    pub content: String,
    /// Base64-encoded images for multimodal models.
    #[serde(default)]
    pub images: Option<Vec<String>>,
    /// Tool call requests produced by the model (only on assistant messages).
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

// ---------------------------------------------------------------------
// Tools / function calling
// ---------------------------------------------------------------------

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolDefinition {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OllamaFunctionDef,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the function parameters.
    pub parameters: serde_json::Value,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub function: OllamaFunctionCall,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON object of arguments to pass to the function.
    pub arguments: serde_json::Value,
}

// ---------------------------------------------------------------------
// Model options
// ---------------------------------------------------------------------

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelOptions {
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<String>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub num_predict: Option<i32>,
}

// ---------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------

#[skip_serializing_none]
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaResponseMessage,
    pub done: bool,
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

#[skip_serializing_none]
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    /// Deliberate thinking trace when `think` is enabled.
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

// ---------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------

impl From<general::ChatRequest> for OllamaChatRequest {
    fn from(g: general::ChatRequest) -> Self {
        OllamaChatRequest {
            model: g.model,
            messages: g.messages.into_iter().map(OllamaChatMessage::from).collect(),
            tools: g.tools.map(|ts| ts.into_iter().map(OllamaToolDefinition::from).collect()),
            format: None,
            options: Some(OllamaModelOptions {
                seed: g.seed.map(|s| s as i64),
                temperature: g.temperature,
                top_k: g.top_k.map(|v| v as u32),
                top_p: g.top_p,
                min_p: g.min_p,
                stop: g.stop,
                num_ctx: g.max_ctx_chars.map(|v| v as u32),
                num_predict: g.max_prediction_tokens.map(|v| v as i32),
            }),
            stream: g.stream,
            think: g.think,
            keep_alive: g.keep_alive,
            logprobs: None,
            top_logprobs: None,
        }
    }
}

impl From<general::Message> for OllamaChatMessage {
    fn from(m: general::Message) -> Self {
        OllamaChatMessage {
            role: m.role,
            content: m.content.unwrap_or_default(),
            images: None,
            tool_calls: m.tool_calls.map(|tc| tc.into_iter().map(OllamaToolCall::from).collect()),
        }
    }
}

impl From<tooling::ToolDef> for OllamaToolDefinition {
    fn from(t: tooling::ToolDef) -> Self {
        OllamaToolDefinition {
            tool_type: t.tool_type,
            function: OllamaFunctionDef {
                name: t.function.name,
                description: Some(t.function.description),
                parameters: t.function.parameters,
            },
        }
    }
}

impl From<general::ToolCall> for OllamaToolCall {
    fn from(tc: general::ToolCall) -> Self {
        OllamaToolCall {
            function: OllamaFunctionCall {
                name: tc.function.name,
                description: None,
                arguments: tc.function.arguments,
            },
        }
    }
}

impl From<OllamaChatResponse> for general::ChatResponse {
    fn from(r: OllamaChatResponse) -> Self {
        general::ChatResponse {
            model: r.model,
            created_at: r.created_at,
            message: general::Message {
                role: r.message.role,
                content: r.message.content,
                tool_calls: r.message.tool_calls.map(|tc| tc.into_iter().map(general::ToolCall::from).collect()),
                tool_name: None,
            },
            done: r.done,
            done_reason: r.done_reason.as_deref() == Some("stop"),
            total_duration: r.total_duration.unwrap_or_default() as usize,
            load_duration: r.load_duration.unwrap_or_default() as usize,
            prompt_eval_count: r.prompt_eval_count.unwrap_or_default() as usize,
            prompt_eval_duration: r.prompt_eval_duration.unwrap_or_default() as usize,
            eval_count: r.eval_count.unwrap_or_default() as usize,
            tool_calls: None,
        }
    }
}

impl From<OllamaToolCall> for general::ToolCall {
    fn from(tc: OllamaToolCall) -> Self {
        general::ToolCall {
            tool_type: None,
            function: general::FunctionCall {
                name: tc.function.name,
                arguments: tc.function.arguments,
            },
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_request_with_tools() {
        let req = OllamaChatRequest {
            model: "qwen3".to_string(),
            messages: vec![OllamaChatMessage {
                role: "user".to_string(),
                content: "What is the weather in Paris?".to_string(),
                images: None,
                tool_calls: None,
            }],
            tools: Some(vec![OllamaToolDefinition {
                tool_type: "function".to_string(),
                function: OllamaFunctionDef {
                    name: "get_weather".to_string(),
                    description: Some("Get current weather".to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": { "type": "string" }
                        },
                        "required": ["location"]
                    }),
                },
            }]),
            format: None,
            options: Some(OllamaModelOptions {
                seed: Some(42),
                temperature: Some(0.7),
                top_k: None,
                top_p: None,
                min_p: None,
                stop: None,
                num_ctx: Some(4096),
                num_predict: None,
            }),
            stream: Some(false),
            think: None,
            keep_alive: None,
            logprobs: None,
            top_logprobs: None,
        };

        let json = serde_json::to_string_pretty(&req).unwrap();
        assert!(json.contains("get_weather"));
        assert!(json.contains("\"seed\": 42"));
        assert!(json.contains("\"stream\": false"));
    }

    #[test]
    fn deserialize_response() {
        let json = r#"{
            "model": "gemma4",
            "created_at": "2025-10-17T23:14:07.414671Z",
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "done": true,
            "done_reason": "stop",
            "total_duration": 174560334,
            "load_duration": 101397084,
            "prompt_eval_count": 11,
            "eval_count": 18
        }"#;

        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "gemma4");
        assert!(resp.done);
        assert_eq!(resp.done_reason.as_deref(), Some("stop"));
        assert_eq!(resp.message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.total_duration, Some(174560334));
    }
}
