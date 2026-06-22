use crate::LlmResult;
use crate::ToolReady;

pub fn run_tool<ToolOutType>(
    _tool: &dyn Fn(&str) -> ToolOutType,
    _llm: impl ToolReady,
) -> LlmResult<&str> {
    Ok("42")
}

pub fn hitchiker_tool() -> String {
    String::from("42")
}
