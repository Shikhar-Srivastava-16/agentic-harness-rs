use crate::LlmError;
use crate::LlmLike;
use crate::LlmResult;
use crate::general;

pub struct Ollama {
    // None system prompt is possible
    sys_pr: Option<String>,
    model: String,
    rest_url: String,
}

impl Default for Ollama {
    fn default() -> Self {
        Ollama {
            sys_pr: None,
            rest_url: String::from("http://localhost:11434"),
            model: String::from("mistral"),
        }
    }
}

// stub
struct OllamaConfig {}

/// The LlmLike trait defines shared behaviour for all object that are LLMs or act like LLMs.
/// Simple examples include LLMs like Claude or GPT, or tools like Ollama which are not LLMs
/// themselves but behave similarly enough that for our purposes, they may as well be the same
impl LlmLike<OllamaConfig> for Ollama {
    /// This method must be synchronous, meaning your implementation is expected to _only_ return
    /// when the prompt has be 'sent' and the LLM or Llm-like object to which it is sent has
    /// finished responding or an error has occurred
    fn prompt(&self, prompt: String) -> LlmResult<String> {
        let messages = match &self.sys_pr {
            Some(pr) => vec![
                general::Message {
                    role: "system".to_string(),
                    content: pr.clone(),
                },
                general::Message {
                    role: "user".to_string(),
                    content: prompt,
                },
            ],

            None => vec![general::Message {
                role: "user".to_string(),
                content: prompt,
            }],
        };

        let request_body = general::ChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
        };

        let client = reqwest::blocking::Client::new();
        let api_url = format!("{}/api/chat", self.rest_url.trim_end_matches('/'));
        let response = client.post(&api_url).json(&request_body).send()?;

        if response.status().is_success() {
            let chat_response: general::ChatResponse = response.json()?;
            Ok(chat_response.message.content)
        } else {
            // FIXME: This needs to be changed to properly write text
            let error_text = response.text()?;
            Err(LlmError::Other)
        }
    }
    /// This method allows you to set a system prompt after the LLM has been created. This is
    /// because the system prompt is the only `dynamic' property that is currently anticipated to
    /// exist.
    /// All other members of the implementor must either be one-time-set (or `final'), or you must
    /// make your own setters
    fn set_sys_prompt(&self, sys_prompt: Option<&str>) -> LlmResult<()> {
        Err(LlmError::OpNotImplemented)
    }
    /// Here, the config type is 'OllamaConfig', which will contain Ollama-Specific things like the
    /// name of the model and so on
    fn init(
        sys_prompt: Option<String>,
        url: Option<String>,
        conf: OllamaConfig,
    ) -> LlmResult<Self> {
        Ok(match (sys_prompt, url) {
            // both need to be set
            // (Some(prompt), Some(url_str)) => Ollama {
            //     sys_pr: Some(prompt),
            //     rest_url: url_str,
            // },
            // default prompt
            // (None, Some(url_str)) => Ollama {
            //     rest_url: url_str,
            //     ..Default::default()
            // },
            // default url
            // (Some(prompt), None) => Ollama {
            //     sys_pr: Some(prompt),
            //     ..Default::default()
            // },
            // default
            // FIXME: The init should actually work properly
            _ => Ollama::default(),
        })
    }
}
