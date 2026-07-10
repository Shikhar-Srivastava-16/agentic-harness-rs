use crate::LlmLike;
use crate::LlmResult;

pub type Tool = dyn Fn(String) -> String;
pub type ToolMap = std::collections::HashMap<String, Box<Tool>>;

pub fn run_tool(tool: &Tool, input: String) -> LlmResult<String> {
    Ok(tool(input))
}
