use crate::HistConfig;
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

pub struct Ollama {
    // None system prompt is possible
    sys_pr: Option<String>,
    model: String,
    rest_url: String,
    hist: Vec<general::Message>,
    hist_config: HistConfig,
    timeout: u64,
    pub tools: ToolMap,
    pub registered_tools: Vec<ToolDef>,
}

impl Default for Ollama {
    fn default() -> Self {
        Ollama {
            sys_pr: None,
            rest_url: String::from("http://localhost:11434"),
            model: String::from("qwen3:8B"),
            hist: Vec::new(),
            hist_config: HistConfig::ConversationBuffer(0),
            timeout: 300,
            tools: HashMap::new(),
            registered_tools: Vec::new(),
        }
    }
}

// FIXME: stub
pub struct OllamaConfig {
    pub name: String,
}

impl LlmLike for Ollama {
    type Conf = OllamaConfig;
    type ApiRequest = format::ollama::ChatRequest;
    type ApiResponse = format::ollama::ChatResponse;

    fn user_cue(&self) -> String {
        "user".to_string()
    }

    fn response_cue(&self) -> String {
        "assistant".to_string()
    }

    fn timeout(&self) -> &u64 {
        &self.timeout
    }

    fn raw_query(
        &mut self,
        request_body: general::ChatRequest,
    ) -> LlmResult<general::ChatResponse> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout)) // 5 minutes
            .build()
            .unwrap();
        let api_url = format!("{}/api/chat", self.rest_url.trim_end_matches('/'));
        let response = client.post(&api_url).json(&request_body).send()?;

        if response.status().is_success() {
            let mut chat_response: general::ChatResponse = response.json()?;
            if chat_response.tool_calls.is_some() && chat_response.message.tool_calls.is_none() {
                chat_response.message.tool_calls = chat_response.tool_calls.clone();
            }
            Ok(chat_response)
        } else {
            // FIXME: This needs to be changed to properly write errors text
            let error_text = response.text()?;
            println!("Ollama: Error: {}", error_text);
            Err(LlmError::Other)
        }
    }

    /// Here, the config type is 'OllamaConfig', which will contain Ollama-Specific things like the
    /// name of the model and so on
    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self> {
        // FIXME: The init should actually work properly
        let mut st = Ollama::default();

        // if sys prompt exists, set it
        if let Some(spr) = sys_prompt {
            st.set_sys_prompt(spr);
        };

        // if the url is specially provided, reset it
        if let Some(u) = url {
            st.rest_url = u;
        };

        st.model = conf.name;

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

impl ToolReady for Ollama {
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
