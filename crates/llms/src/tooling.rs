use crate::LlmLike;
use crate::LlmResult;
use crate::general;
use crate::msg;
use serde::Serialize;

/// A tool/function specification which can be sent to the LLM, so that it can request the harness
/// for a call.
#[derive(Serialize, Clone, Debug)]
pub struct ToolDef {
    #[serde(rename = "type")]
    /// Always `"function"`; retained as a string for forward compatibility.
    pub tool_type: String,
    /// The function definition (name, description, JSON Schema parameters).
    pub function: FunctionDef,
}

/// The inner definition of a tool — name, description, and parameters.
#[derive(Serialize, Clone, Debug)]
pub struct FunctionDef {
    /// The function name.
    pub name: String,
    /// A description of what the function does.
    pub description: String,
    /// JSON Schema for the function parameters.
    pub parameters: serde_json::Value,
}

/// Function pointer type for tool implementations — takes a `String` input and returns a `String` output.
pub type ToolFn = dyn Fn(String) -> String;
/// Map from tool name to its callable function.
pub type ToolMap = std::collections::HashMap<String, Box<ToolFn>>;

/// Invokes a tool function with the given input, returning its output or an error.
pub fn run_tool(tool: &ToolFn, input: String) -> LlmResult<String> {
    Ok(tool(input))
}

/// Trait for LLMs that support tool/function calling. Extends `LlmLike` with tool registration and execution.
pub trait ToolReady: LlmLike {
    /// Returns the role string used for tool messages (e.g. `"tool"`).
    fn tool_cue(&self) -> String;
    /// Immutable reference to the tool function map.
    fn tools(&self) -> &ToolMap;
    /// Immutable reference to the list of tool declarations sent to the API.
    fn registered_tools(&self) -> &Vec<ToolDef>;
    /// Registers a new tool with the LLM instance — adds both the `ToolDef` and the callable function.
    fn register_tool(
        &mut self,
        name: String,
        description: String,
        parameters: serde_json::Value,
        func: Box<ToolFn>,
    ) -> LlmResult<()>;

    /// Formats tool output as a message and queries the LLM again, returning the updated response.
    fn tool_respond(
        &mut self,
        tool_name: String,
        tool_output: String,
    ) -> LlmResult<general::ChatResponse> {
        self.history_mut()
            .push(msg("tool", Some(tool_output), None, Some(tool_name)));

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: Some(false),
            tools: Some(self.registered_tools().clone()),
            ..Default::default()
        };

        #[cfg(feature = "log")]
        dbg!("[tool_respond]: TOOL RESPONSE: {:#?}", &request_body);

        let resp = self.query(request_body)?;
        let content = resp.message.content.clone().unwrap_or_default();
        self.history_mut().push(msg(
            "assistant",
            Some(content),
            resp.message.tool_calls.clone(),
            None,
        ));
        Ok(resp)
    }

    /// High-level prompt that auto-detects tool calls in the LLM response and executes them in a loop.
    fn prompt(&mut self, prompt: String) -> LlmResult<String> {
        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        let query_start = std::time::Instant::now();

        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        crate::bench::begin_prompt();

        let user_cue = self.user_cue();

        self.history_mut()
            .push(msg(&user_cue, Some(prompt), None, None));

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: Some(false),
            tools: Some(self.registered_tools().clone()),
            ..Default::default()
        };

        let mut r = self.query(request_body);

        #[cfg(feature = "log")]
        dbg!("response done: {:#?}", &r);

        let mut resp = r?;
        let mut content = resp.message.content.clone().unwrap();

        self.history_mut().push(msg(
            "assistant",
            Some(content.clone()),
            resp.message.tool_calls.clone(),
            None,
        ));

        #[cfg(feature = "log")]
        dbg!("LLM raw response: {:?}\n\n", &content);

        // Structured tool call loop
        while let Some(tool_calls) = resp.message.tool_calls.clone() {
            if tool_calls.is_empty() {
                break;
            }

            for tc in &tool_calls {
                let func_name = &tc.function.name;
                let args = &tc.function.arguments;
                #[cfg(any(feature = "tinylog", feature = "log"))]
                dbg!("[ToolReady::prompt] tool call: {}({})", func_name, args);

                #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
                crate::bench::start_tool_cycle(func_name);

                let tool_fn: &ToolFn = self
                    .tools()
                    .get(func_name)
                    .unwrap_or_else(|| panic!("Unknown tool: {}", func_name));

                let args_str = args.to_string();

                #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
                let tool_time_start = std::time::Instant::now();

                let tool_out = crate::tooling::run_tool(tool_fn, args_str)?;

                #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
                crate::bench::emit_tool_time(func_name, tool_time_start.elapsed(), self.model());

                #[cfg(feature = "log")]
                dbg!("[ToolReady::prompt] tool output: {}", &tool_out);

                resp = self.tool_respond(func_name.clone(), tool_out)?;
            }

            content = resp.message.content.clone().unwrap_or_default();
        }

        #[cfg(any(feature = "bench", feature = "bench-threadsafe"))]
        crate::bench::emit_query(query_start.elapsed(), self.model());

        Ok(content)
    }
}
