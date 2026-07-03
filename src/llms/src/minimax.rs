use crate::LlmError;
use crate::LlmLike;
use crate::LlmResult;
use crate::ToolReady;
use crate::general;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

pub struct Minimax {
    sys_pr: Option<String>,
    model: String,
    api_url: String,
    pub api_key: String,
    hist: Vec<general::Message>,
    timeout: u64,
    pub tools: HashMap<String, Box<dyn Fn(String) -> String>>,
}

impl Default for Minimax {
    fn default() -> Self {
        Minimax {
            sys_pr: None,
            model: String::from("minimaxai/minimax-m3"),
            api_url: String::from("https://integrate.api.nvidia.com/v1/chat/completions"),
            api_key: String::new(),
            hist: Vec::new(),
            timeout: 300,
            tools: HashMap::new(),
        }
    }
}

#[derive(Deserialize)]
struct NvidiaChoice {
    message: NvidiaMessage,
}

#[derive(Deserialize)]
struct NvidiaMessage {
    content: String,
}

#[derive(Deserialize)]
struct NvidiaResponse {
    choices: Vec<NvidiaChoice>,
}

pub struct MinimaxConfig {
    pub api_key: String,
    pub model: String,
}

impl LlmLike for Minimax {
    type Conf = MinimaxConfig;

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
            Ok(nv
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_default())
        } else {
            let error_text = response.text()?;
            println!("Minimax: Error: {}", error_text);
            Err(LlmError::Other)
        }
    }

    fn init(
        sys_prompt: Option<String>,
        url: Option<String>,
        conf: Self::Conf,
    ) -> LlmResult<Self> {
        let mut st = Minimax::default();
        if let Some(spr) = sys_prompt {
            st.set_sys_prompt(spr);
        }
        if let Some(u) = url {
            st.api_url = u;
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

impl ToolReady for Minimax {
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
