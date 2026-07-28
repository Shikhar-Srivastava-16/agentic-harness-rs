use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorMode {
    Strict,
    Lenient,
}

impl ErrorMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lenient" => ErrorMode::Lenient,
            _ => ErrorMode::Strict,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub system_prompt: String,
    pub model: String,
    pub url: String,
    pub api_key: Option<String>,
    pub backend: String,
    pub query: Option<String>,
    pub error_mode: ErrorMode,
}

static GLOBAL_CONFIG: OnceLock<AppConfig> = OnceLock::new();

pub fn init(config: AppConfig) {
    GLOBAL_CONFIG.set(config).ok();
}

pub fn get() -> &'static AppConfig {
    GLOBAL_CONFIG.get().expect("Config not initialized. Call config::init() first.")
}
