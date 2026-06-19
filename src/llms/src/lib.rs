use thiserror::Error;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
pub mod general;
pub mod ollama;

type LlmResult<T> = std::result::Result<T, LlmError>;

trait LlmLike {
    fn prompt(&self, prompt: &str) -> LlmResult<&str>;
    fn sys_prompt(&self, sys_prompt: Option<&str>) -> LlmResult<()>;
    fn init(sys_prompt: Option<&str>) -> LlmResult<&Self>;
}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Unexpected Format found!")]
    UnexpectedFormat,
}
