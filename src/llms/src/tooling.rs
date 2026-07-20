use crate::LlmResult;
use serde::Serialize;
use crate::msg;
use crate::general;
use crate::LlmLike;

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

pub trait ToolReady: LlmLike {
    fn tool_cue(&self) -> String;
    fn tools(&self) -> &ToolMap;
    fn registered_tools(&self) -> &Vec<ToolDef>;
    fn register_tool(
        &mut self,
        name: String,
        description: String,
        parameters: serde_json::Value,
        func: Box<ToolFn>,
    ) -> LlmResult<()>;

    fn tool_respond(&mut self, tool_name: String, tool_output: String) -> LlmResult<general::ChatResponse> {
        self.history_mut().push(msg("tool", Some(tool_output), None, Some(tool_name)));

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: false,
            tools: Some(self.registered_tools().clone()),
        };

        eprintln!("[tool_respond]: TOOL RESPONSE: {:#?}", request_body);

        let resp = self.query(request_body)?;
        let content = resp.message.content.clone().unwrap_or_default();
        self.history_mut().push(msg("assistant", Some(content), resp.message.tool_calls.clone(), None));
        Ok(resp)
    }

    fn prompt(&mut self, prompt: String) -> LlmResult<String> {
        let user_cue = self.user_cue();

        self.history_mut().push(msg(&user_cue, Some(prompt), None, None));

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: false,
            tools: Some(self.registered_tools().clone()),
        };

        eprintln!(
            "[ToolReady::prompt] request body:\n{}",
            serde_json::to_string_pretty(&request_body).unwrap()
        );

        let mut resp = self.query(request_body)?;
        let mut content = resp.message.content.clone().unwrap_or_default();
        self.history_mut().push(msg("assistant", Some(content.clone()), resp.message.tool_calls.clone(), None));

        eprintln!("\n\nDEBUG: LLM raw response: {:?}\n\n", content);

        // Structured tool call loop
        while let Some(tool_calls) = resp.message.tool_calls.clone() {
            if tool_calls.is_empty() {
                break;
            }

            for tc in &tool_calls {
                let func_name = &tc.function.name;
                let args = &tc.function.arguments;
                eprintln!(
                    "[ToolReady::prompt] tool call: {}({})",
                    func_name, args
                );

                let tool_fn: &ToolFn = self
                    .tools()
                    .get(func_name)
                    .unwrap_or_else(|| panic!("Unknown tool: {}", func_name));

                let args_str = args.to_string();
                let tool_out = crate::tooling::run_tool(tool_fn, args_str)?;
                eprintln!("[ToolReady::prompt] tool output: {}", tool_out);

                resp = self.tool_respond(func_name.clone(), tool_out)?;
            }

            content = resp.message.content.clone().unwrap_or_default();
        }

        Ok(content)
    }
}
