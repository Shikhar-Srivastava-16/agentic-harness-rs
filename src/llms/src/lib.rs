use std::fmt::Debug;
use thiserror::Error;

pub mod config;
pub mod format;
pub mod general;
// pub mod minimax;
pub mod ollama;
pub mod tooling;

use tooling::ToolReady;

type LlmResult<T> = std::result::Result<T, LlmError>;

// NOTE: For potentially supporting other forms of output, which are parrsed into structs
enum LlmOutput {
    String(String),
}

pub mod private {
    pub struct Filter(());
    impl Filter {
        // this function can be used anywhere inside the crate but _only_ within the crate, due to
        // the pub(crate) classification
        pub(crate) fn new() -> Filter {
            Filter(())
        }
    }
}

fn msg(
    role: &str,
    content: Option<String>,
    tool_calls: Option<Vec<general::ToolCall>>,
    tool_name: Option<String>,
) -> general::Message {
    general::Message {
        role: role.to_string(),
        content,
        tool_calls,
        tool_name,
    }
}

pub trait LlmLike {
    type Conf;
    type ApiRequest: From<general::ChatRequest> + Debug;
    type ApiResponse: Into<general::ChatResponse> + Debug;

    /// The low-level query each backend must implement — works with API types directly.
    fn raw_query(&mut self, req: Self::ApiRequest) -> LlmResult<Self::ApiResponse>;

    /// Higher-level query that does the conversion automatically.
    fn query(&mut self, req: general::ChatRequest) -> LlmResult<general::ChatResponse> {
        let api_req: Self::ApiRequest = req.into();

        eprintln!("Request: {:#?}", api_req);

        let api_resp = self.raw_query(api_req)?;
        Ok(api_resp.into())
    }

    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self>
    where
        Self: Sized;

    // Getters for default implementations
    fn history(&self) -> &Vec<general::Message>;
    fn history_mut(&mut self) -> &mut Vec<general::Message>;
    fn model(&self) -> &String;
    fn timeout(&self) -> &u64;

    /// This method allows you to set a system prompt after the LLM has been created. This is
    /// because the system prompt is the only `dynamic' property that is currently anticipated to
    /// exist.
    /// All other members of the implementor must either be one-time-set (or `final'), or you must
    /// make your own setters
    fn sys_prompt(&self) -> &Option<String>;

    fn private_set_sys_pr(&mut self, pr: String, _: private::Filter) -> ();

    fn user_cue(&self) -> String;

    fn response_cue(&self) -> String;
    /// The LlmLike trait defines shared behaviour for all object that are LLMs or act like LLMs.
    /// Simple examples include LLMs like Claude or GPT, or tools like Ollama which are not LLMs
    /// themselves but behave similarly enough that for our purposes, they may as well be the same
    /// This method must be synchronous, meaning your implementation is expected to _only_ return
    /// when the prompt has be 'sent' and the LLM or Llm-like object to which it is sent has
    /// finished responding or an error has occurred
    fn prompt(&mut self, prompt: String) -> LlmResult<String> {
        let user_cue = self.user_cue();
        let response_cue = self.response_cue();

        self.history_mut()
            .push(msg(&user_cue, Some(prompt), None, None));

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: None,
            tools: None,
            ..Default::default()
        };

        // query
        let resp = self.query(request_body.into())?;
        let content = resp.message.content.unwrap_or_default();
        self.history_mut().push(msg(
            &response_cue,
            Some(content.clone()),
            resp.message.tool_calls,
            None,
        ));
        Ok(content)
        // query
    }
    /// You define how history is maintained for your particular model. In some cases, history
    /// might not need to be maintained by this layer at all, so this method would be a stub.
    /// Otherwise, the implementation of this method gives you complete control over the
    /// maintainance of history in your struct
    fn add_to_history(&mut self, msg_obj: general::Message) -> LlmResult<()> {
        self.history_mut().push(msg_obj);

        Ok(())
    }
    /// This function would be used in order to manage/engineer the context and history
    fn summarise(&mut self, from: usize, to: usize) -> LlmResult<()> {
        assert!(from > to);

        let mut msg_text: String = String::from("");
        for i in &self.history_mut()[from..=to] {
            msg_text += i.content.as_deref().unwrap_or("");
        }

        // NOTE: this syntax is notable. The two `LlmLike` and `ToolReady` are both used to
        // represent LLMs and `ToolReady` is dependent on `LlmLike`
        // This means that we must specify which `prompt` this is using
        let summary = self.prompt(String::from(msg_text.clone()));
        self.history_mut().reverse();
        self.history_mut()
            .push(msg("user", Some(summary.unwrap()), None, None));
        self.history_mut().reverse();

        Ok(())
    }

    fn set_sys_prompt(&mut self, sys_prompt: String) -> LlmResult<()> {
        if self.sys_prompt().is_none() {
            self.private_set_sys_pr(sys_prompt, private::Filter::new());
            let _ = self.add_to_history(msg("system", self.sys_prompt().clone(), None, None));
            Ok(())
        } else {
            // NOTE: This might not be correct, check conventions
            Err(LlmError::OpNotSupported(String::from(
                "Cannot reset system prompt once set",
            )))
        }
    }
    // might benefit from VecDeque + pop. Otherwise, most common case is also worst case for
    // complexity
    fn remove_from_history(&mut self, idx: usize) -> LlmResult<()> {
        self.history_mut().remove(idx);
        Ok(())
    }
}

// can have non-volatile memory, using a VectorDB
// From vectorDB_rs
pub trait Memory {}

// AgentReady Models are ToolReady and have non-volatile Memory
pub trait AgentReady: ToolReady + Memory {}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Unexpected Format found!")]
    UnexpectedFormat,
    #[error("Operation Not Implemented!")]
    OpNotImplemented,
    #[error("Operation Not Supported by This Implementor, because {0}!")]
    OpNotSupported(String),

    #[error("Async Operation Timed Out!")]
    TimedOut,
    // FIXME: Placeholder
    #[error("Other!")]
    Other,
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
}

// FIXME: Use HistConfig
enum HistConfig {
    /// Maintain all of the messages, upto a certain size. If size = 0, then never stop recording
    ConversationBuffer(i32),
    /// Maintain the `k` most recent messages, discard anything older
    LastKConversationBuffer(i32),
    /// Incrementally summarise all messages
    ConversationSummary,
    // NOTE: Consider adding a token-limited ConveresationSummaryBuffer
    /// Maintain the `k` most recent messages, summarize the rest
    ConversationSummaryBuffer,
}
