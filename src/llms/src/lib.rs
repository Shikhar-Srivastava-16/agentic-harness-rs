use thiserror::Error;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
pub mod general;
pub mod ollama;

type LlmResult<T> = std::result::Result<T, LlmError>;

enum LlmOutput {
    String(String),
}

pub trait LlmLike<T> {
    /// Set up the logic needed to prompt the LLM, accesing only the associated object and the string value of a
    /// user prompt. Returns the output of the LLM
    fn prompt(&self, prompt: String) -> LlmResult<String>;
    /// Set the system prompt and change the
    fn set_sys_prompt(&self, sys_prompt: Option<&str>) -> LlmResult<()>;

    /// The only option all LlmLikes are expected to have in common are a system prompt and an API
    /// URL. All other details are expected to be included using the config, which is of a generic
    /// type T
    fn init(sys_prompt: Option<String>, url: Option<String>, conf: T) -> LlmResult<Self>
    where
        Self: Sized;
}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Unexpected Format found!")]
    UnexpectedFormat,
    #[error("Operation Not Implemented!")]
    OpNotImplemented,
    #[error("Operation Not Supported by This Implementor!")]
    OpNotSupported,

    #[error("Async Operation Timed Out!")]
    TimedOut,
    // FIXME: Placeholder
    #[error("Other!")]
    Other,
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
}
