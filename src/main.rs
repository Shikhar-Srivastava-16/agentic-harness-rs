use clap::Parser;
use eframe::egui::{self, Ui};
use llms::LlmLike;
use llms::ToolReady;
use llms::minimax::Minimax;
use llms::ollama::Ollama;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "You are a helpful assistant with access to tools.
Available tools:
- search(query: str) - searches the web and returns results
- add(a: int, b: int) - performs addition

When you need to use a tool, output EXACTLY this format and nothing else:
TOOL_CALL: {\"tool\": \"tool_name\", \"args\": {\"arg1\": \"value1\"}}

After receiving a result, continue your response normally.
Do not guess tool results. Always wait for the actual result.

You have access to tools that provide real-time information.
When you receive a TOOL_RESULT, you MUST treat it as ground truth.
NEVER contradict or second-guess a tool result with your own training data.
Your training data is outdated - tool results are always more current and accurate.
If a tool says X, your answer must reflect X.
CRITICAL RULES:
- Your training data is OUTDATED and WRONG for current events.
- TOOL_RESULT is always correct. Never contradict it.
- If TOOL_RESULT says X, you MUST answer X, even if it conflicts with what you know OR think OR infer.
- Do NOT say 'as of my knowledge cutoff'. You have real-time tools.
- Do NOT suggest the tool result might be wrong.
- Do NOT ask for more than one fact at a time
- ONLY request one TOOL_RESULT at a time. If you need multiple, wait for the tool to return before you move to the next")]
    system: String,
    #[arg(short, long, default_value = "minimaxai/minimax-m3")]
    model: String,
    #[arg(short, long, default_value = "http://localhost:11434")]
    url: String,
    #[arg(short = 'k', long)]
    api_key: Option<String>,
    #[arg(short, long, default_value = "ollama")]
    backend: String,
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LLM Chat",
        options,
        Box::new(move |_cc| Ok(Box::new(LlmChatApp::new(args)))),
    )
}

struct LlmChatApp {
    args: Args,
    messages: Vec<ChatMessage>,
    input: String,
    loading: bool,
    backend_initialized: bool,
    minimax: Option<Minimax>,
    ollama: Option<Ollama>,
}

struct ChatMessage {
    role: String,
    content: String,
}

impl LlmChatApp {
    fn new(args: Args) -> Self {
        Self {
            args,
            messages: Vec::new(),
            input: String::new(),
            loading: false,
            backend_initialized: false,
            minimax: None,
            ollama: None,
        }
    }

    fn init_backend(&mut self) {
        if self.backend_initialized {
            return;
        }
        match self.args.backend.as_str() {
            "minimax" => {
                let api_key = self.args.api_key.clone().unwrap_or_else(|| {
                    eprintln!("ERROR: --api-key required for minimax");
                    std::process::exit(1);
                });
                let mut m = Minimax::init(
                    Some(self.args.system.clone()),
                    None,
                    llms::minimax::MinimaxConfig {
                        api_key,
                        model: self.args.model.clone(),
                    },
                )
                .unwrap();
                m.tools
                    .insert("search_tool".to_string(), Box::new(foobar_tool));
                m.tools
                    .insert("add_tool".to_string(), Box::new(hitchhiker_tool));
                self.minimax = Some(m);
            }
            _ => {
                let mut o = Ollama::init(
                    Some(self.args.system.clone()),
                    None,
                    llms::ollama::OllamaConfig {
                        name: "qwen3:8b".into(),
                    },
                )
                .unwrap();
                o.tools
                    .insert("search_tool".to_string(), Box::new(foobar_tool));
                o.tools
                    .insert("add_tool".to_string(), Box::new(hitchhiker_tool));
                self.ollama = Some(o);
            }
        }
        self.backend_initialized = true;
    }
}

impl eframe::App for LlmChatApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("LLM Chat");
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(ui.available_height() - 60.0)
                .show(ui, |ui| {
                    for msg in &self.messages {
                        ui.label(format!("[{}] {}", msg.role, msg.content));
                    }
                    if self.loading {
                        ui.label("⏳ Waiting for response...");
                    }
                });

            ui.separator();

            ui.horizontal(|ui| {
                let width = ui.available_width() - 80.0;
                let input = ui.add_sized(
                    [width, 30.0],
                    egui::TextEdit::singleline(&mut self.input)
                        .hint_text("Type your message here..."),
                );

                let send_clicked = ui
                    .add_enabled(
                        !self.loading && !self.input.is_empty(),
                        egui::Button::new("Send"),
                    )
                    .clicked();

                let enter_pressed =
                    input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                if send_clicked || enter_pressed {
                    let prompt = self.input.trim().to_string();
                    if !prompt.is_empty() {
                        self.input.clear();
                        self.messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: prompt.clone(),
                        });
                        self.loading = true;

                        self.init_backend();

                        let result = match self.args.backend.as_str() {
                            "minimax" => {
                                let m = self.minimax.as_mut().unwrap();
                                <Minimax as ToolReady>::prompt(m, prompt)
                            }
                            _ => {
                                let o = self.ollama.as_mut().unwrap();
                                <Ollama as ToolReady>::prompt(o, prompt)
                            }
                        };

                        match result {
                            Ok(ans) => {
                                self.messages.push(ChatMessage {
                                    role: "assistant".to_string(),
                                    content: ans,
                                });
                            }
                            Err(e) => {
                                self.messages.push(ChatMessage {
                                    role: "error".to_string(),
                                    content: format!("{e}"),
                                });
                            }
                        }
                        self.loading = false;
                    }
                }
            });
        });
    }
}

pub fn foobar_tool(_: String) -> String {
    "fubar".into()
}

pub fn hitchhiker_tool(_: String) -> String {
    "42".into()
}
