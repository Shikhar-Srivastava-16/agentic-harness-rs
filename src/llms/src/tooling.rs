use crate::LlmResult;
use crate::ToolReady;

pub fn run_tool(tool: &dyn Fn(String) -> String, input: String) -> LlmResult<String> {
    Ok(tool(input))
}

pub fn foobar_tool(_: String) -> String {
    "fubar".into()
}

pub fn hitchhiker_tool(_: String) -> String {
    "42".into()
}
