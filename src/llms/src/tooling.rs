use crate::LlmResult;
use crate::ToolReady;

pub fn run_tool(_tool: &dyn Fn(&str) -> String) -> LlmResult<&str> {
    Ok("42")
}

pub fn hitchiker_tool(_: &str) -> String {
    String::from("42")
}
