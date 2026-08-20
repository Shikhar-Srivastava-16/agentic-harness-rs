use benchmarks::test_config::HarnessConfig;
use llms::LlmLike;
use llms::config::AppConfig;
use llms::config::{self, ErrorMode};
use llms::minimax::Nvidia;
use llms::tooling::ToolReady;
use serde_json::{Value, json};
use std::path::Path;

fn get_str(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn profile(person: &str) -> Option<String> {
    let p = person.trim().to_lowercase();
    match p.as_str() {
        "jane goodwin" => Some(
            "Profile for Jane Goodwin: born 1979, occupation novelist, first published book 'The Glass Lantern' (2004), lives at 14 Willow Lane.".to_string(),
        ),
        "jane" => Some(
            "Profile for Jane Goodwin: born 1979, occupation novelist, first published book 'The Glass Lantern' (2004), lives at 14 Willow Lane.".to_string(),
        ),
        "tom ashford" => Some(
            "Profile for Tom Ashford: born 1982, occupation journalist at the Rivertown Gazette.".to_string(),
        ),
        "tom" => Some(
            "Profile for Tom Ashford: born 1982, occupation journalist at the Rivertown Gazette.".to_string(),
        ),
        "mayor ellis" => Some(
            "Profile for Mayor Ellis: born 1961, occupation mayor of Rivertown, first elected in 2016.".to_string(),
        ),
        "ellis" => Some(
            "Profile for Mayor Ellis: born 1961, occupation mayor of Rivertown, first elected in 2016.".to_string(),
        ),
        "rosa lin" => Some(
            "Profile for Rosa Lin: born 1985, occupation head librarian at the Rivertown library.".to_string(),
        ),
        "rosa" => Some(
            "Profile for Rosa Lin: born 1985, occupation head librarian at the Rivertown library.".to_string(),
        ),
        _ => None,
    }
}

fn article(topic: &str) -> Option<String> {
    let t = topic.trim().to_lowercase();
    if t.contains("founding") || t.contains("founder") {
        Some(
            "Rivertown was founded in 1847 by Josiah Crane on the banks of the Willowriver, growing from a sawmill settlement into a market town of 12,400 people.".to_string(),
        )
    } else if t.contains("library") {
        Some(
            "The Rivertown Library opened in 1923 and holds over 40,000 volumes; it is run by head librarian Rosa Lin.".to_string(),
        )
    } else if t.contains("festival") || t.contains("fair") {
        Some(
            "The Rivertown Annual Festival is held every October in the town square, featuring markets, lanterns and a parade.".to_string(),
        )
    } else if t.contains("railway") || t.contains("station") {
        Some(
            "The Rivertown railway line connected the town to the coast until it closed in 1968; the old station is now a museum.".to_string(),
        )
    } else {
        None
    }
}

fn fact_for(query: &str) -> Option<String> {
    let q = query.trim().to_lowercase();
    if q.contains("founding") || q.contains("founded") || q.contains("founder") {
        Some("Search result: Rivertown was founded in 1847 by Josiah Crane.".to_string())
    } else if q.contains("population") {
        Some("Search result: Rivertown has a population of 12,400.".to_string())
    } else if q.contains("festival") {
        Some("Search result: The annual festival is held in October.".to_string())
    } else if q.contains("library") {
        Some("Search result: The Rivertown library opened in 1923.".to_string())
    } else if q.contains("railway") {
        Some("Search result: The railway line closed in 1968.".to_string())
    } else {
        None
    }
}

fn related(person: &str) -> Option<String> {
    match person.trim().to_lowercase().as_str() {
        "jane goodwin" | "jane" => {
            Some("Jane Goodwin is related to: Tom Ashford, Rosa Lin.".to_string())
        }
        "tom ashford" | "tom" => {
            Some("Tom Ashford is related to: Jane Goodwin, Mayor Ellis.".to_string())
        }
        "rosa lin" | "rosa" => Some("Rosa Lin is related to: Jane Goodwin.".to_string()),
        "mayor ellis" | "ellis" => Some("Mayor Ellis is related to: Tom Ashford.".to_string()),
        _ => None,
    }
}

fn profile_parts(person: &str) -> (String, String) {
    match person.trim().to_lowercase().as_str() {
        "jane goodwin" | "jane" => ("1979".to_string(), "novelist".to_string()),
        "tom ashford" | "tom" => ("1982".to_string(), "journalist".to_string()),
        "mayor ellis" | "ellis" => ("1961".to_string(), "mayor".to_string()),
        "rosa lin" | "rosa" => ("1985".to_string(), "librarian".to_string()),
        _ => ("unknown".to_string(), "unknown".to_string()),
    }
}

fn search_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let q = get_str(&v, "query");
    fact_for(&q).unwrap_or_else(|| format!("No results found for query '{}'.", q))
}

fn get_profile_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let p = get_str(&v, "person");
    profile(&p).unwrap_or_else(|| format!("No profile found for person '{}'.", p))
}

fn fetch_article_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "topic");
    article(&t).unwrap_or_else(|| format!("No article found for topic '{}'.", t))
}

fn list_related_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let p = get_str(&v, "person");
    related(&p).unwrap_or_else(|| format!("No known relations for person '{}'.", p))
}

fn compare_people_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let a = get_str(&v, "person_a");
    let b = get_str(&v, "person_b");
    let (ay, ao) = profile_parts(&a);
    let (by, bo) = profile_parts(&b);
    format!("{} ({}, born {}) | {} ({}, born {})", a, ao, ay, b, bo, by)
}

fn init_backend(cfg: &AppConfig) -> Result<Nvidia, llms::LlmError> {
    let mut o = Nvidia::init(
        Some(cfg.system_prompt.clone()),
        Some(cfg.url.clone()),
        llms::minimax::NvidiaConfig {
            api_key: std::env::var("API_KEY").expect("API key not found"),
            model: cfg.model.clone(),
        },
    )?;

    o.register_tool(
        "search_tool".to_string(),
        "Searches the Rivertown knowledge base for facts matching a query. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" }
            },
            "required": ["query"]
        }),
        Box::new(search_tool),
    )?;

    o.register_tool(
        "get_profile_tool".to_string(),
        "Returns the profile of a Rivertown resident (e.g. 'Jane Goodwin', 'Tom Ashford', 'Mayor Ellis', 'Rosa Lin'). Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "person": { "type": "string", "description": "The person to look up" }
            },
            "required": ["person"]
        }),
        Box::new(get_profile_tool),
    )?;

    o.register_tool(
        "fetch_article_tool".to_string(),
        "Fetches a short article about a Rivertown topic such as its founding, library, festival or railway. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "topic": { "type": "string", "description": "The topic to fetch an article for" }
            },
            "required": ["topic"]
        }),
        Box::new(fetch_article_tool),
    )?;

    o.register_tool(
        "list_related_tool".to_string(),
        "Lists the residents related to a given person. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "person": { "type": "string", "description": "The person whose relations to list" }
            },
            "required": ["person"]
        }),
        Box::new(list_related_tool),
    )?;

    o.register_tool(
        "compare_people_tool".to_string(),
        "Compares two residents and returns their occupations and birth years. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "person_a": { "type": "string", "description": "First person" },
                "person_b": { "type": "string", "description": "Second person" }
            },
            "required": ["person_a", "person_b"]
        }),
        Box::new(compare_people_tool),
    )?;

    Ok(o)
}

fn main() {
    let cfg_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config.toml");
    let content = std::fs::read_to_string(&cfg_path).expect("failed to read config.toml");
    let cfg: HarnessConfig = toml::from_str(&content).expect("failed to parse config.toml");

    unsafe {
        std::env::set_var(
            "LLMS_BENCH_OUTPUT",
            Path::new(env!("CARGO_MANIFEST_DIR")).join(&cfg.bench_output),
        );
    }

    let app_cfg = AppConfig {
        system_prompt: cfg.system_prompt.clone(),
        model: cfg.model.clone(),
        url: cfg.url.clone(),
        api_key: std::env::var("API_KEY").ok().filter(|s| !s.is_empty()),
        backend: cfg.backend.clone(),
        query: cfg.query.clone(),
        error_mode: ErrorMode::from_str(&cfg.error_mode),
    };
    config::init(app_cfg.clone());

    let mut backend = init_backend(&app_cfg).expect("failed to init backend");

    let prompts = &cfg.prompts;
    let iterations = cfg.iterations as usize;

    println!(
        "Running {} iterations with model '{}'",
        cfg.iterations, cfg.model
    );
    for i in 0..iterations {
        let prompt = prompts[i % prompts.len()].clone();
        print!("  [{}/{}] prompt: \"{}\" ... ", i + 1, iterations, prompt);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match ToolReady::prompt(&mut backend, prompt) {
            Ok(ans) => println!("OK ({} chars)", ans.len()),
            Err(e) => panic!("ERROR: {}", e),
        }
    }

    println!("Done. Results written to bench output file.");
    llms::bench::finalize();
}
