use crate::LlmLike;
use crate::LlmResult;

pub fn run_tool(tool: &dyn Fn(String) -> String, input: String) -> LlmResult<String> {
    Ok(tool(input))
}
