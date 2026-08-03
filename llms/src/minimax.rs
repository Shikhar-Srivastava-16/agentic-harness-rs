use crate::LlmError;
use crate::LlmLike;
use crate::LlmResult;
use crate::general;
use crate::tooling::FunctionDef;
use crate::tooling::ToolDef;
use crate::tooling::ToolFn;
use crate::tooling::ToolMap;
use crate::tooling::ToolReady;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

/// Minimax/NVIDIA backend implementation of `LlmLike` and `ToolReady`.
pub struct Minimax {
    sys_pr: Option<String>,
    model: String,
    api_url: String,
    pub api_key: String,
    hist: Vec<general::Message>,
    timeout: u64,
    pub tools: ToolMap,
    pub registered_tools: Vec<ToolDef>,
}

impl Default for Minimax {
    /// Sensible defaults: model `minimaxai/minimax-m3`, NVIDIA API URL, 300s timeout.
    fn default() -> Self {
        Minimax {
            sys_pr: None,
            model: String::from("minimaxai/minimax-m3"),
            api_url: String::from("https://integrate.api.nvidia.com/v1/chat/completions"),
            api_key: String::new(),
            hist: Vec::new(),
            timeout: 300,
            tools: HashMap::new(),
            registered_tools: Vec::new(),
        }
    }
}

/// Internal deserialization helper for the NVIDIA response shape.
#[derive(Deserialize)]
struct NvidiaChoice {
    message: NvidiaMessage,
}

/// Internal deserialization helper for a message in the NVIDIA response.
#[derive(Deserialize)]
struct NvidiaMessage {
    content: String,
}

/// Internal deserialization helper for the top-level NVIDIA response.
#[derive(Deserialize)]
struct NvidiaResponse {
    choices: Vec<NvidiaChoice>,
}

/// Minimax-specific configuration.
pub struct MinimaxConfig {
    /// The API key for the NVIDIA/Minimax endpoint.
    pub api_key: String,
    /// The model identifier.
    pub model: String,
}

impl LlmLike for Minimax {
    type Conf = MinimaxConfig;

    /// Returns `"user"`.
    fn user_cue(&self) -> String {
        "user".to_string()
    }

    /// Returns `"assistant"`.
    fn response_cue(&self) -> String {
        "assistant".to_string()
    }

    /// Returns the request timeout in seconds.
    fn timeout(&self) -> &u64 {
        &self.timeout
    }

    /// Sends the request to the NVIDIA `/v1/chat/completions` endpoint with Minimax-specific
    /// overrides (`max_tokens=8192`, `temperature=1.0`, `top_p=0.95`).
    fn query(&mut self, request_body: general::ChatRequest) -> LlmResult<general::ChatResponse> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
            .build()
            .unwrap();

        let mut body = serde_json::to_value(&request_body).unwrap();
        body["max_tokens"] = json!(8192);
        body["temperature"] = json!(1.00);
        body["top_p"] = json!(0.95);

        let response = client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .json(&body)
            .send()?;

        if response.status().is_success() {
            let nv: NvidiaResponse = response.json()?;
            let content = nv
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_default();
            Ok(general::ChatResponse {
                message: general::Message {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_name: None,
                },
                tool_calls: None,
            })
        } else {
            let error_text = response.text()?;
            #[cfg(feature = "log")]
            dbg!("Minimax: Error: {}", error_text);
            Err(LlmError::Other)
        }
    }

    /// Constructs a `Minimax` instance from config. Applies optional system prompt and URL overrides.
    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self> {
        let mut st = Minimax::default();
        if let Some(spr) = sys_prompt {
            let _ = st.set_sys_prompt(spr);
        }
        if let Some(u) = url {
            st.api_url = u;
        }
        st.api_key = conf.api_key;
        st.model = conf.model;
        Ok(st)
    }

    /// Immutable reference to the conversation history.
    fn history(&self) -> &Vec<general::Message> {
        &self.hist
    }

    /// Mutable reference to the conversation history.
    fn history_mut(&mut self) -> &mut Vec<general::Message> {
        &mut self.hist
    }

    /// The model identifier currently in use.
    fn model(&self) -> &String {
        &self.model
    }

    /// The system prompt, if one has been set.
    fn sys_prompt(&self) -> &Option<String> {
        &self.sys_pr
    }

    /// Internal setter for `sys_prompt`, gated by `Filter`.
    fn private_set_sys_pr(&mut self, a: String, _: crate::private::Filter) {
        self.sys_pr = Some(a);
    }
}

impl ToolReady for Minimax {
    /// Immutable reference to the tool function map.
    fn tools(&self) -> &ToolMap {
        &self.tools
    }

    /// Immutable reference to the list of registered tool declarations.
    fn registered_tools(&self) -> &Vec<ToolDef> {
        &self.registered_tools
    }

    /// Registers a new tool — adds both the `ToolDef` and the callable function.
    fn register_tool(
        &mut self,
        name: String,
        description: String,
        parameters: serde_json::Value,
        func: Box<ToolFn>,
    ) -> LlmResult<()> {
        self.registered_tools.push(ToolDef {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: name.clone(),
                description,
                parameters,
            },
        });
        self.tools.insert(name, func);
        Ok(())
    }

    /// Returns `"tool"`.
    fn tool_cue(&self) -> String {
        "tool".to_string()
    }
}
