use crate::HistConfig;
use crate::LlmError;
use crate::LlmLike;
use crate::LlmResult;
use crate::ToolReady;
use crate::general;
use std::collections::HashMap;

pub struct Ollama {
    // None system prompt is possible
    sys_pr: Option<String>,
    model: String,
    rest_url: String,
    hist: Vec<general::Message>,
    hist_config: HistConfig,
    timeout: u64,
    pub tools: HashMap<String, Box<dyn Fn(String) -> String>>,
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
        }
    }
}

// FIXME: stub
pub struct OllamaConfig {
    pub name: String,
}

// user cue: "user"
// response cue: "assistant"
// tool cue: "tool"

impl LlmLike for Ollama {
    type Conf = OllamaConfig;

    fn user_cue(&self) -> String {
        "user".to_string()
    }

    fn response_cue(&self) -> String {
        "assistant".to_string()
    }

    fn timeout(&self) -> &u64 {
        &self.timeout
    }

    fn query(&mut self, request_body: general::ChatRequest) -> LlmResult<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout)) // 5 minutes
            .build()
            .unwrap();
        let api_url = format!("{}/api/chat", self.rest_url.trim_end_matches('/'));
        let response = client.post(&api_url).json(&request_body).send()?;

        if response.status().is_success() {
            let chat_response: general::ChatResponse = response.json()?;
            Ok(chat_response.message.content)
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

    fn set_sys_prompt(&mut self, sys_prompt: String) -> LlmResult<()> {
        if self.sys_pr.is_none() {
            self.sys_pr = Some(sys_prompt);
            let _ = self.add_to_history(general::Message {
                role: "system".to_string(),
                content: self.sys_pr.clone().unwrap(),
            });
            Ok(())
        } else {
            // NOTE: This might not be correct, check conventions
            Err(LlmError::OpNotSupported(String::from(
                "Cannot reset system prompt once set",
            )))
        }
    }

    fn model(&self) -> &String {
        &self.model
    }
}

impl ToolReady for Ollama {
    fn tools(&self) -> &HashMap<String, Box<dyn Fn(String) -> String>> {
        &self.tools
    }
    fn register_tool() -> LlmResult<()> {
        Err(LlmError::OpNotImplemented)
    }

    fn tool_cue(&self) -> String {
        "tool".to_string()
    }
}
