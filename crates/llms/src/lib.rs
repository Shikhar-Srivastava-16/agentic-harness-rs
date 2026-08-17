use std::fmt::Debug;
use thiserror::Error;

#[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
pub mod bench;

pub mod config;
pub mod format;
pub mod general;
pub mod ollama;
pub mod tooling;

pub use tooling::ToolReady;

pub type LlmResult<T> = std::result::Result<T, LlmError>;

/// Placeholder enum for future output types (currently only `String` is supported).
enum LlmOutput {
    String(String),
}

pub mod private {
    /// Crate-internal access gate; `Filter::new()` is `pub(crate)` and only callable within `llms`.
    pub struct Filter(());
    impl Filter {
        // this function can be used anywhere inside the crate but _only_ within the crate, due to
        // the pub(crate) classification
        pub(crate) fn new() -> Filter {
            Filter(())
        }
    }
}

/// Helper to construct a [`general::Message`] without needing to specify all fields.
pub(crate) fn msg(
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

/// Shared behaviour for all LLM backends or Llm-like objects (e.g. Ollama, which is not an LLM but behaves similarly).
pub trait LlmLike {
    type Conf;
    type ApiRequest: From<general::ChatRequest> + Debug;
    type ApiResponse: Into<general::ChatResponse> + Debug;

    /// Low-level query — works with API request/response types directly. Each backend must implement this.
    fn raw_query(&mut self, req: Self::ApiRequest) -> LlmResult<Self::ApiResponse>;

    /// Higher-level query that does the conversion automatically.
    fn query(&mut self, req: general::ChatRequest) -> LlmResult<general::ChatResponse> {
        let api_req: Self::ApiRequest = req.into();

        #[cfg(feature = "log")]
        dbg!("Request: {:#?}", &api_req);

        let api_resp = self.raw_query(api_req)?;
        Ok(api_resp.into())
    }

    /// Constructs a new backend instance. `sys_prompt` and `url` are optional overrides; `conf` is the backend-specific configuration type.
    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self>
    where
        Self: Sized;

    /// Immutable reference to the conversation history.
    fn history(&self) -> &Vec<general::Message>;
    /// Mutable reference to the conversation history.
    fn history_mut(&mut self) -> &mut Vec<general::Message>;
    /// The model identifier currently in use.
    fn model(&self) -> &String;
    /// Request timeout in seconds.
    fn timeout(&self) -> &u64;

    /// This method allows you to set a system prompt after the LLM has been created. This is
    /// because the system prompt is the only `dynamic` property that is currently anticipated to
    /// exist.
    /// All other members of the implementor must either be one-time-set (or `final`), or you must
    /// make your own setters.
    fn sys_prompt(&self) -> &Option<String>;

    /// Internal setter for `sys_prompt`, gated by `Filter` to prevent external callers from resetting it.
    fn private_set_sys_pr(&mut self, pr: String, _: private::Filter) -> ();

    /// Returns the role string used for user messages (e.g. `"user"`).
    fn user_cue(&self) -> String;

    /// Returns the role string used for assistant responses (e.g. `"assistant"`).
    fn response_cue(&self) -> String;
    /// Shared behaviour for all LLM backends or Llm-like objects (e.g. Ollama, which is not an LLM but behaves similarly).
    /// Simple examples include LLMs like Claude or GPT, or tools like Ollama which are not LLMs
    /// themselves but behave similarly enough that for our purposes, they may as well be the same.
    /// This method must be synchronous, meaning your implementation is expected to _only_ return
    /// when the prompt has been 'sent' and the LLM or Llm-like object to which it is sent has
    /// finished responding or an error has occurred.
    fn prompt(&mut self, prompt: String) -> LlmResult<String> {
        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        let query_start = std::time::Instant::now();

        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        crate::bench::begin_prompt();

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

        let resp = self.query(request_body)?;
        let content = resp.message.content.unwrap_or_default();
        self.history_mut().push(msg(
            &response_cue,
            Some(content.clone()),
            resp.message.tool_calls,
            None,
        ));

        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        crate::bench::emit_query(query_start.elapsed(), self.model());

        Ok(content)
    }
    /// You define how history is maintained for your particular model. In some cases, history
    /// might not need to be maintained by this layer at all, so this method would be a stub.
    /// Otherwise, the implementation of this method gives you complete control over the
    /// maintenance of history in your struct.
    fn add_to_history(&mut self, msg_obj: general::Message) -> LlmResult<()> {
        self.history_mut().push(msg_obj);

        Ok(())
    }
    /// Concatenates messages from `from` to `to`, sends the concatenated text as a prompt, and replaces that range with the summary. Note: `from` must be greater than `to`.
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

    /// Sets the system prompt once; subsequent calls return `OpNotSupported`. Also prepends a system message to history.
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
// From vectorDB_rs?
pub trait Memory {}

// AgentReady Models are ToolReady and have non-volatile Memory
pub trait AgentReady: ToolReady + Memory {}

/// Errors that can occur during LLM operations.
#[derive(Error, Debug)]
pub enum LlmError {
    /// An unexpected format was encountered.
    #[error("Unexpected Format found!")]
    UnexpectedFormat,
    /// The requested operation is not implemented yet.
    #[error("Operation Not Implemented!")]
    OpNotImplemented,
    /// The operation is not supported by this implementor.
    #[error("Operation Not Supported by This Implementor, because {0}!")]
    OpNotSupported(String),
    /// An async operation timed out.
    #[error("Async Operation Timed Out!")]
    TimedOut,
    /// An unspecified or other error occurred.
    #[error("Other!")]
    Other,
    /// An HTTP request failed.
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
}

/// Controls how conversation history is maintained.
pub(crate) enum HistConfig {
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
