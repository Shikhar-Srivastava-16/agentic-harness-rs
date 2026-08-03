//! Rust structs for the OpenAI Chat Completions API request body
//! (`POST /v1/chat/completions`).
//!
//! Design notes for anyone wiring this into a provider adapter:
//!   - Every `Option<T>` field is `skip_serializing_if = "Option::is_none"`
//!     so unset fields don't get sent at all — matters for non-OpenAI
//!     backends (NIM/vLLM, Ollama compat, DashScope compat) that may
//!     reject or choke on unrecognized/null fields rather than ignoring them.
//!   - `Content` and `StopSequence` are untagged enums because OpenAI accepts
//!     either a bare string or a structured array in those positions —
//!     serde_json needs the untagged hint to pick the right variant.
//!   - `Message` is a tagged enum on `role` so you get compile-time
//!     guarantees that e.g. a `tool` message always carries `tool_call_id`
//!     and a `system` message can't accidentally carry `tool_calls`.
//!   - Deprecated fields (`functions`, `function_call`, legacy `max_tokens`)
//!     are included but marked, since a framework may still need to detect
//!     and warn on them rather than silently drop support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------
// Top-level request
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Top-level request body for the OpenAI Chat Completions API.
pub struct ChatCompletionRequest {
    /// The model identifier.
    pub model: String,
    /// The conversation messages.
    pub messages: Vec<Message>,

    // --- sampling / generation control ---
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Number of response candidates to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Maximum number of tokens in the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Deprecated — kept for back-compat. Prefer `max_completion_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
    /// Presence penalty for topic repetition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty for token repetition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Adjust the likelihood of specified tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Fixed seed for deterministic outputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Reasoning-model-only (o-series etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,

    // --- output shape control ---
    /// Controls the output format (text, JSON object, or JSON Schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Output modalities (text, audio).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,
    /// Audio output configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,
    /// Structured prediction output configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionConfig>,

    // --- logprobs ---
    /// Whether to return token log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return (0-20).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,

    // --- tools ---
    /// Tool/function definitions the model can call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Controls which tool the model uses (`auto`, `none`, `required`, or specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Whether to allow parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Deprecated — superseded by `tools`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<FunctionDef>>,
    /// Deprecated — superseded by `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallChoice>,

    // --- streaming ---
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Streaming configuration options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    // --- infra / operational ---
    /// OpenAI service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Whether to store the completion for model distillation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Arbitrary metadata attached to the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    /// End-user identifier for abuse monitoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

// ---------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
#[serde(rename_all = "snake_case")]
/// A single message in a chat conversation. Tagged on `role` so each variant
/// carries only the fields relevant to that role.
pub enum Message {
    System {
        content: Content,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: Content,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Content>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
    },
    Tool {
        content: Content,
        tool_call_id: String,
    },
}

/// Message content can be a bare string or an array of typed parts
/// (text/image/audio blocks) for multimodal input. `untagged` lets serde
/// pick whichever shape matches at (de)serialize time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A single part of a multimodal message (text, image, or audio).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
}

/// An image URL for multimodal input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    /// Level of detail for the image ("auto", "low", or "high").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto" | "low" | "high"
}

/// Audio input for multimodal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data.
    pub data: String,
    /// Audio format (e.g. `"wav"` or `"mp3"`).
    pub format: String,
}

// ---------------------------------------------------------------------
// Tools / function calling
// ---------------------------------------------------------------------

/// A tool definition the model can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function { function: FunctionDef },
}

/// A function definition for use as a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// The function name.
    pub name: String,
    /// Optional description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Raw JSON Schema — left as `serde_json::Value` since schema shape
    /// is arbitrary and you don't want to model JSON Schema itself in Rust.
    pub parameters: serde_json::Value,
    /// Whether to enforce strict schema adherence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A tool call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The tool type (always `"function"`).
    #[serde(rename = "type")]
    pub kind: String,
    /// The function call details.
    pub function: FunctionCall,
}

/// The `function` object inside a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The function name.
    pub name: String,
    /// Arguments arrive as a JSON *string* per the OpenAI spec —
    /// caller is responsible for parsing it, don't assume it's valid JSON.
    pub arguments: String,
}

/// Controls which tool the model uses. Can be a mode string
/// (`"auto"`, `"none"`, `"required"`) or a specific named function.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Named {
        r#type: String,
        function: NamedFunctionChoice,
    },
}

/// The mode for `tool_choice`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
}

/// A named function choice for `tool_choice`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedFunctionChoice {
    /// The function name to call.
    pub name: String,
}

/// Deprecated legacy pre-`tool_choice` shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallChoice {
    Mode(String), // "auto" | "none"
    Named { name: String },
}

// ---------------------------------------------------------------------
// Response format / structured outputs
// ---------------------------------------------------------------------

/// Controls the output format for the model's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaSpec },
}

/// JSON Schema specification for structured outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
    /// The name of the schema.
    pub name: String,
    /// Optional description of the schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema object.
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema adherence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ---------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------

/// Configuration for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Whether to include token usage information in stream events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

// ---------------------------------------------------------------------
// Audio / multimodal output
// ---------------------------------------------------------------------

/// Output modality for the model's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Audio,
}

/// Audio output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// The voice to use for audio output.
    pub voice: String,
    /// The audio format (e.g. `"wav"`, `"mp3"`, `"opus"`).
    pub format: String,
}

/// Structured prediction output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// The prediction type (e.g. `"content"`).
    #[serde(rename = "type")]
    pub kind: String,
    /// The content to predict.
    pub content: Content,
}

// ---------------------------------------------------------------------
// Small enums / helpers
// ---------------------------------------------------------------------

/// Stop sequence(s) that halt model generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    One(String),
    Many(Vec<String>), // max 4
}

/// OpenAI service tier for request routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Priority,
}

/// Reasoning effort level for reasoning models (o-series).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

// ---------------------------------------------------------------------
// Example construction
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_serializes_a_tool_call_request() {
        let req = ChatCompletionRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![
                Message::System {
                    content: Content::Text("You are a helpful assistant.".into()),
                    name: None,
                },
                Message::User {
                    content: Content::Text("What's the weather in Boston?".into()),
                    name: None,
                },
            ],
            temperature: Some(0.3),
            top_p: None,
            n: None,
            max_completion_tokens: Some(512),
            max_tokens: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            seed: None,
            reasoning_effort: None,
            response_format: None,
            modalities: None,
            audio: None,
            prediction: None,
            logprobs: None,
            top_logprobs: None,
            tools: Some(vec![Tool::Function {
                function: FunctionDef {
                    name: "get_current_weather".into(),
                    description: Some("Get the current weather for a location".into()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "format": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location", "format"]
                    }),
                    strict: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Mode(ToolChoiceMode::Auto)),
            parallel_tool_calls: Some(true),
            functions: None,
            function_call: None,
            stream: Some(false),
            stream_options: None,
            service_tier: None,
            store: None,
            metadata: None,
            user: None,
        };

        let json = serde_json::to_string_pretty(&req).unwrap();
        // println!("{json}");
        assert!(json.contains("get_current_weather"));
    }
}
