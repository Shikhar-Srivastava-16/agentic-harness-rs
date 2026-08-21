use crate::LlmError;
use crate::LlmLike;
use crate::LlmResult;
use crate::format;
use crate::general;
use crate::tooling::FunctionDef;
use crate::tooling::ToolDef;
use crate::tooling::ToolFn;
use crate::tooling::ToolMap;
use crate::tooling::ToolReady;
use std::collections::HashMap;

use reqwest::blocking::Client;

/// NVIDIA OpenAI-compatible backend implementation of [`LlmLike`] and [`ToolReady`].
///
/// Talks to any OpenAI Chat Completions-compatible endpoint (NVIDIA NIM, vLLM, ...)
/// using the shared [`format::openai`] types, mirroring the `ollama` backend pattern.
pub struct Nvidia {
    sys_pr: Option<String>,
    model: String,
    /// Full endpoint URL, e.g. `https://integrate.api.nvidia.com/v1/chat/completions`.
    rest_url: String,
    api_key: String,
    hist: Vec<general::Message>,
    timeout: u64,
    client: Client,
    pub tools: ToolMap,
    pub registered_tools: Vec<ToolDef>,
}

impl Default for Nvidia {
    /// Sensible defaults: Nemotron on the NVIDIA API, 300s timeout.
    fn default() -> Self {
        let tout = 300;
        eprintln!("building client");
        let cl = Client::builder()
            .timeout(std::time::Duration::from_secs(tout))
            .build()
            .unwrap();

        Nvidia {
            sys_pr: None,
            model: String::from("nvidia/nemotron-3.5-lightning-30b-a3b"),
            rest_url: String::from("https://integrate.api.nvidia.com/v1/chat/completions"),
            api_key: String::new(),
            hist: Vec::new(),
            timeout: tout,
            client: cl,
            tools: HashMap::new(),
            registered_tools: Vec::new(),
        }
    }
}

/// NVIDIA-specific configuration.
pub struct NvidiaConfig {
    /// The API key for the NVIDIA endpoint.
    pub api_key: String,
    /// The model identifier.
    pub model: String,
}

impl LlmLike for Nvidia {
    type Conf = NvidiaConfig;
    type ApiRequest = format::openai::ChatCompletionRequest;
    type ApiResponse = format::openai::ChatCompletionResponse;

    fn user_cue(&self) -> String {
        "user".to_string()
    }

    fn response_cue(&self) -> String {
        "assistant".to_string()
    }

    fn timeout(&self) -> &u64 {
        &self.timeout
    }

    fn client(&self) -> &Client {
        &self.client
    }

    fn raw_query(
        &mut self,
        request_body: format::openai::ChatCompletionRequest,
    ) -> LlmResult<format::openai::ChatCompletionResponse> {
        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        crate::bench::before_http_send(&self.model);

        let response = self
            .client()
            .post(&self.rest_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .json(&request_body)
            .send()?;

        if response.status().is_success() {
            let chat_response: format::openai::ChatCompletionResponse = response.json()?;
            Ok(chat_response)
        } else {
            let error_text = response.text()?;
            eprintln!("Nvidia: Error: {}", error_text);
            Err(LlmError::Other)
        }
    }

    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self> {
        let mut st = Nvidia::default();

        if let Some(spr) = sys_prompt {
            let _ = st.set_sys_prompt(spr);
        }

        if let Some(u) = url {
            st.rest_url = u;
        }

        st.api_key = conf.api_key;
        st.model = conf.model;

        Ok(st)
    }

    fn history(&self) -> &Vec<general::Message> {
        &self.hist
    }

    fn history_mut(&mut self) -> &mut Vec<general::Message> {
        &mut self.hist
    }

    fn model(&self) -> &String {
        &self.model
    }

    fn sys_prompt(&self) -> &Option<String> {
        &self.sys_pr
    }

    fn private_set_sys_pr(&mut self, a: String, _: crate::private::Filter) {
        self.sys_pr = Some(a);
    }
}

impl ToolReady for Nvidia {
    fn tools(&self) -> &ToolMap {
        &self.tools
    }

    fn registered_tools(&self) -> &Vec<ToolDef> {
        &self.registered_tools
    }

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

    fn tool_cue(&self) -> String {
        "tool".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_nemotron() {
        let n = Nvidia::default();
        assert_eq!(n.model, "nvidia/nemotron-3.5-lightning-30b-a3b");
        assert_eq!(
            n.rest_url,
            "https://integrate.api.nvidia.com/v1/chat/completions"
        );
    }

    #[test]
    fn init_applies_config() {
        let mut n = Nvidia::init(
            Some("You are Nemotron".to_string()),
            None,
            NvidiaConfig {
                api_key: "sk-test".to_string(),
                model: "nvidia/nemotron-3.5-lightning-30b-a3b".to_string(),
            },
        )
        .unwrap();

        assert_eq!(n.api_key, "sk-test");
        assert_eq!(*n.sys_prompt(), Some("You are Nemotron".to_string()));
        assert_eq!(n.history().len(), 1);
        assert_eq!(n.history()[0].role, "system");

        let g = general::ChatRequest {
            model: n.model().clone(),
            messages: n.history().clone(),
            ..Default::default()
        };
        let req: format::openai::ChatCompletionRequest = g.into();
        assert_eq!(req.model, "nvidia/nemotron-3.5-lightning-30b-a3b");
        assert_eq!(req.messages.len(), 1);
    }
}
