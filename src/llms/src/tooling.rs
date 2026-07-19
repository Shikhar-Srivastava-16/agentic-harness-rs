use crate::LlmResult;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
}

pub type ToolFn = dyn Fn(String) -> String;
pub type ToolMap = std::collections::HashMap<String, Box<ToolFn>>;

pub fn run_tool(tool: &ToolFn, input: String) -> LlmResult<String> {
    Ok(tool(input))
}
