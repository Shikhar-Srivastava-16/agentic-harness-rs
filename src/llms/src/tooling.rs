use crate::LlmResult;
use crate::ToolReady;

pub fn run_tool(tool: &dyn Fn(String) -> String, input: String) -> LlmResult<String> {
    Ok(tool(input))
}

pub fn hitchiker_tool(a: String) -> String {
    if a.to_lowercase().contains("president") {
        eprintln!("resp: john doe");
        String::from("John Doe")
    } else {
        eprintln!("resp: 42");
        String::from("42")
    }
}
