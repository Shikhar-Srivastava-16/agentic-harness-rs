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
