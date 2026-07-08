use std::collections::HashMap;
use thiserror::Error;

pub mod format;
pub mod general;
pub mod minimax;
pub mod ollama;
pub mod tooling;

type LlmResult<T> = std::result::Result<T, LlmError>;

// NOTE: For potentially supporting other forms of output, which are parrsed into structs
enum LlmOutput {
    String(String),
}

pub mod private {
    pub struct Filter(());
    impl Filter {
        // this function can be used anywhere inside the crate but _only_ within the crate, due to
        // the pub(crate) classification
        pub(crate) fn new() -> Filter {
            Filter(())
        }
    }
}

pub trait LlmLike {
    type Conf;

    /// Set up the logic needed to prompt the LLM, accesing only the associated object and the string value of a
    /// user prompt. Returns the output of the LLM
    /// Set the system prompt and change the
    fn query(&mut self, req: general::ChatRequest) -> LlmResult<String>;
    fn init(sys_prompt: Option<String>, url: Option<String>, conf: Self::Conf) -> LlmResult<Self>
    where
        Self: Sized;

    // Getters for default implementations
    fn history(&self) -> &Vec<general::Message>;
    fn history_mut(&mut self) -> &mut Vec<general::Message>;
    fn model(&self) -> &String;
    fn timeout(&self) -> &u64;

    /// This method allows you to set a system prompt after the LLM has been created. This is
    /// because the system prompt is the only `dynamic' property that is currently anticipated to
    /// exist.
    /// All other members of the implementor must either be one-time-set (or `final'), or you must
    /// make your own setters
    fn sys_prompt(&self) -> &Option<String>;

    fn private_set_sys_pr(&mut self, pr: String, _: private::Filter) -> ();

    fn user_cue(&self) -> String;

    fn response_cue(&self) -> String;
    /// The LlmLike trait defines shared behaviour for all object that are LLMs or act like LLMs.
    /// Simple examples include LLMs like Claude or GPT, or tools like Ollama which are not LLMs
    /// themselves but behave similarly enough that for our purposes, they may as well be the same
    /// This method must be synchronous, meaning your implementation is expected to _only_ return
    /// when the prompt has be 'sent' and the LLM or Llm-like object to which it is sent has
    /// finished responding or an error has occurred
    fn prompt(&mut self, prompt: String) -> LlmResult<String> {
        let user_cue = self.user_cue();
        let response_cue = self.response_cue();

        self.history_mut().push(general::Message {
            role: user_cue.to_string(),
            content: prompt,
        });

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: false,
        };

        // query
        let resp = self.query(request_body)?;
        self.history_mut().push(general::Message {
            role: response_cue.to_string(),
            content: resp.clone(),
        });
        Ok(resp)
        // query
    }
    /// You define how history is maintained for your particular model. In some cases, history
    /// might not need to be maintained by this layer at all, so this method would be a stub.
    /// Otherwise, the implementation of this method gives you complete control over the
    /// maintainance of history in your struct
    fn add_to_history(&mut self, msg: general::Message) -> LlmResult<()> {
        self.history_mut().push(msg);

        Ok(())
    }
    /// This function would be used in order to manage/engineer the context and history
    fn summarise(&mut self, from: usize, to: usize) -> LlmResult<()> {
        assert!(from > to);

        let mut msg: String = String::from("");
        for i in &self.history_mut()[from..=to] {
            msg += i.content.as_str();
        }

        // NOTE: this syntax is notable. The two `LlmLike` and `ToolReady` are both used to
        // represent LLMs and `ToolReady` is dependent on `LlmLike`
        // This means that we must specify which `prompt` this is using
        let summary = self.prompt(String::from(msg.clone()));
        self.history_mut().reverse();
        self.history_mut().push(general::Message {
            role: String::from("user"),
            content: summary.unwrap(),
        });
        self.history_mut().reverse();

        Ok(())
    }

    fn set_sys_prompt(&mut self, sys_prompt: String) -> LlmResult<()> {
        if self.sys_prompt().is_none() {
            self.private_set_sys_pr(sys_prompt, private::Filter::new());
            let _ = self.add_to_history(general::Message {
                role: "system".to_string(),
                content: self.sys_prompt().clone().unwrap(),
            });
            Ok(())
        } else {
            // NOTE: This might not be correct, check conventions
            Err(LlmError::OpNotSupported(String::from(
                "Cannot reset system prompt once set",
            )))
        }
    }
    // might benefit from VecDeque + pop. Otherwise, most common case is also worst case for
    // complexity
    fn remove_from_history(&mut self, idx: usize) -> LlmResult<()> {
        self.history_mut().remove(idx);
        Ok(())
    }
}

// can have non-volatile memory, using a VectorDB
// From vectorDB_rs
pub trait Memory {}

pub trait ToolReady: LlmLike {
    fn tool_cue(&self) -> String;
    fn tools(&self) -> &HashMap<String, Box<dyn Fn(String) -> String>>;
    fn register_tool() -> LlmResult<()>;

    fn tool_respond(&mut self, prompt: String) -> LlmResult<String> {
        let tool = self.tool_cue();
        let resp_c = self.response_cue();

        self.history_mut().push(general::Message {
            role: "assistant".to_string(),
            // role: resp_c,
            content: prompt,
        });

        let request_body = general::ChatRequest {
            model: self.model().clone(),
            messages: self.history().clone(),
            stream: false,
        };

        eprintln!("[tool_respond]: TOOL RESPONSE: {:#?}", request_body);

        let resp = self.query(request_body)?;
        self.history_mut().push(general::Message {
            role: "tool".to_string(),
            // role: tool,
            content: resp.clone(),
        });
        Ok(resp)
    }
    fn prompt(&mut self, prompt: String) -> LlmResult<String> {
        let mut ans = <Self as LlmLike>::prompt(self, prompt).unwrap();
        ans = String::from(ans.trim());
        eprintln!("DEBUG: LLM raw response: {:?}", ans);

        while ans.starts_with("TOOL_CALL") {
            println!("tool: {}", ans);

            let tool: &dyn Fn(String) -> String = if ans.contains("search") {
                self.tools().get("search_tool").unwrap()
            } else if ans.contains("add") {
                self.tools().get("add_tool").unwrap()
            } else {
                panic!("Unknown tool call: {}", ans)
            };
            let tool_out = crate::tooling::run_tool(tool, String::from(ans)).unwrap();

            // FIXME: More Error Handling
            ans = String::from(self.tool_respond(String::from(tool_out)).unwrap().trim());
        }
        Ok(String::from(ans))
    }
}

// AgentReady Models are ToolReady and have non-volatile Memory
pub trait AgentReady: ToolReady + Memory {}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Unexpected Format found!")]
    UnexpectedFormat,
    #[error("Operation Not Implemented!")]
    OpNotImplemented,
    #[error("Operation Not Supported by This Implementor, because {0}!")]
    OpNotSupported(String),

    #[error("Async Operation Timed Out!")]
    TimedOut,
    // FIXME: Placeholder
    #[error("Other!")]
    Other,
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
}

// FIXME: Use HistConfig
enum HistConfig {
    /// Maintain all of the messages, upto a certain size. If size = 0, then never stop recording
    ConversationBuffer(i32),
    /// Maintain the `k` most recent messages, discard anything older
    LastKConversationBuffer(i32),
    /// Incrementally summarise all messages
    ConversationSummary,
    // NOTE: Consider adding a token-limited ConveresationSummaryBuffer
    /// Maintain the `k` most recent messages, summarize the rest
    ConversationSummaryBuffer,
}
