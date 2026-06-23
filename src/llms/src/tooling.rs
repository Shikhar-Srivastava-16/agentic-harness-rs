use crate::LlmLike;
use crate::LlmResult;

pub type ToolType<ToolOutType> = dyn Fn(&str) -> ToolOutType;

// ToolReady Models can track their history and are LlmLike
pub trait ToolReady: LlmLike {
    fn prompt(&mut self, prompt: String) -> LlmResult<String>;
}

// FIXME: stub
fn run_tool<ToolOutType>(_llm: impl ToolReady, _tool: ToolType<ToolOutType>) -> LlmResult<String> {
    Err(crate::LlmError::OpNotImplemented)
}
