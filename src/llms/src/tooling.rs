use crate::LlmResult;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Serialize, Clone, Debug)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

pub type ToolFn = dyn Fn(String) -> String;
pub type ToolMap = std::collections::HashMap<String, Box<ToolFn>>;

pub fn run_tool(tool: &ToolFn, input: String) -> LlmResult<String> {
    Ok(tool(input))
}
