use thiserror::Error;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
pub mod general;
pub mod ollama;
pub mod tooling;

type LlmResult<T> = std::result::Result<T, LlmError>;

// NOTE: For potentially supporting other forms of output, such are parrsed into structs
enum LlmOutput {
    String(String),
}

pub trait LlmLike {
    type Conf;

    /// Set up the logic needed to prompt the LLM, accesing only the associated object and the string value of a
    /// user prompt. Returns the output of the LLM
    fn prompt(&mut self, prompt: String) -> LlmResult<String>;
    /// Set the system prompt and change the
    fn set_sys_prompt(&mut self, sys_prompt: String) -> LlmResult<()>;
    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self>
    where
        Self: Sized;

    // can keep track of the chat history, but history is volatile memory (like RAM)
    fn add_to_history(&mut self, msg: general::Message) -> LlmResult<()>;

    fn summarise(&mut self, from: usize, to: usize) -> LlmResult<()>;

    fn remove_from_history(&mut self, idx: usize) -> LlmResult<()>;
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
