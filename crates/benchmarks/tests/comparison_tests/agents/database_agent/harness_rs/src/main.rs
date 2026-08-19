use benchmarks::test_config::HarnessConfig;
use llms::LlmLike;
use llms::config::AppConfig;
use llms::config::{self, ErrorMode};
use llms::ollama::Ollama;
use llms::tooling::ToolReady;
use serde_json::{Value, json};
use std::path::Path;

fn get_str(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn get_i64(v: &Value, key: &str) -> i64 {
    v[key].as_i64().unwrap_or(0)
}

fn rows(table: &str) -> Vec<Value> {
    match table {
        "employees" => vec![
            json!({"id": 1, "name": "Alice", "dept": "Sales", "salary": 55000}),
            json!({"id": 2, "name": "Bob", "dept": "Engineering", "salary": 82000}),
            json!({"id": 3, "name": "Carol", "dept": "Sales", "salary": 61000}),
            json!({"id": 4, "name": "Dan", "dept": "Marketing", "salary": 58000}),
            json!({"id": 5, "name": "Eve", "dept": "Engineering", "salary": 98000}),
            json!({"id": 6, "name": "Frank", "dept": "HR", "salary": 47000}),
            json!({"id": 7, "name": "Grace", "dept": "Engineering", "salary": 74000}),
            json!({"id": 8, "name": "Heidi", "dept": "Marketing", "salary": 52000}),
            json!({"id": 9, "name": "Ivan", "dept": "Sales", "salary": 66000}),
            json!({"id": 10, "name": "Judy", "dept": "HR", "salary": 50000}),
        ],
        "products" => vec![
            json!({"id": 1, "name": "Ergonomic Chair", "price": 249.99, "stock": 25}),
            json!({"id": 2, "name": "Mechanical Keyboard", "price": 119.50, "stock": 8}),
            json!({"id": 3, "name": "4K Monitor", "price": 379.00, "stock": 4}),
            json!({"id": 4, "name": "USB-C Dock", "price": 89.99, "stock": 42}),
            json!({"id": 5, "name": "Webcam", "price": 129.00, "stock": 15}),
            json!({"id": 6, "name": "Desk Lamp", "price": 34.99, "stock": 3}),
        ],
        "orders" => vec![
            json!({"id": 1, "product_id": 1, "quantity": 2, "amount": 499.98}),
            json!({"id": 2, "product_id": 3, "quantity": 1, "amount": 379.00}),
            json!({"id": 3, "product_id": 2, "quantity": 3, "amount": 358.50}),
            json!({"id": 4, "product_id": 6, "quantity": 5, "amount": 174.95}),
            json!({"id": 5, "product_id": 5, "quantity": 2, "amount": 258.00}),
            json!({"id": 6, "product_id": 4, "quantity": 1, "amount": 89.99}),
        ],
        _ => Vec::new(),
    }
}

fn schema_str(table: &str) -> String {
    match table {
        "employees" => "employees: id(int), name(string), dept(string), salary(int)".to_string(),
        "products" => "products: id(int), name(string), price(number), stock(int)".to_string(),
        "orders" => "orders: id(int), product_id(int), quantity(int), amount(number)".to_string(),
        _ => format!("unknown table '{}'", table),
    }
}

fn table_exists(table: &str) -> bool {
    matches!(table, "employees" | "products" | "orders")
}

fn fmt_num(f: f64) -> String {
    if f.fract() == 0.0 {
        format!("{}", f as i64)
    } else {
        format!("{:.2}", f)
    }
}

fn filter_match(row: &Value, filters: &Value) -> bool {
    let Some(obj) = filters.as_object() else {
        return true;
    };
    for (k, fv) in obj {
        match row.get(k) {
            Some(rv) => {
                let a = fmt_num_or_str(rv);
                let b = fmt_num_or_str(fv);
                if a != b {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}

fn fmt_num_or_str(v: &Value) -> String {
    if let Some(n) = v.as_i64() {
        n.to_string()
    } else if let Some(f) = v.as_f64() {
        fmt_num(f)
    } else {
        v.as_str().unwrap_or("").to_string()
    }
}

fn list_tables_tool(_: String) -> String {
    "employees, products, orders".to_string()
}

fn get_schema_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    schema_str(&t)
}

fn query_table_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    let filters = v.get("filters").cloned().unwrap_or(json!({}));
    let table_rows = rows(&t);
    let matched: Vec<&Value> = table_rows
        .iter()
        .filter(|r| filter_match(r, &filters))
        .collect();
    if matched.is_empty() {
        return format!("no rows found in {} matching the filters", t);
    }
    let mut out = String::new();
    for (i, r) in matched.iter().enumerate() {
        out.push_str(&r.to_string());
        if i + 1 < matched.len() {
            out.push('\n');
        }
    }
    out
}

fn get_row_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    let id = get_i64(&v, "id");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    match rows(&t).iter().find(|r| r["id"].as_i64() == Some(id)) {
        Some(r) => r.to_string(),
        None => format!("no row with id {} in {}", id, t),
    }
}

fn count_rows_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    rows(&t).len().to_string()
}

fn aggregate_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    let column = get_str(&v, "column");
    let op = get_str(&v, "op");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    let vals: Vec<f64> = rows(&t)
        .iter()
        .filter_map(|r| {
            r.get(&column).and_then(|c| {
                c.as_f64().or_else(|| c.as_i64().map(|i| i as f64))
            })
        })
        .collect();
    if vals.is_empty() {
        return format!("error: column '{}' has no numeric values", column);
    }
    let result = match op.as_str() {
        "min" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "sum" => vals.iter().sum(),
        "avg" | "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
        _ => return format!("error: unsupported op '{}' (use min, max, avg or sum)", op),
    };
    fmt_num(result)
}

fn search_column_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    let column = get_str(&v, "column");
    let value = get_str(&v, "value");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    let ids: Vec<i64> = rows(&t)
        .iter()
        .filter(|r| {
            r.get(&column)
                .map(|c| fmt_num_or_str(c) == value)
                .unwrap_or(false)
        })
        .filter_map(|r| r["id"].as_i64())
        .collect();
    if ids.is_empty() {
        return format!("no rows in {} where {} = '{}'", t, column, value);
    }
    format!(
        "rows in {} where {} = '{}': {}",
        t,
        column,
        value,
        ids.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn simulate_insert_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    let values = v.get("values").cloned().unwrap_or(json!({}));
    let next_id = rows(&t).len() + 1;
    format!(
        "OK (dry-run): would insert into {} the row {} with id {}",
        t,
        values.to_string(),
        next_id
    )
}

fn simulate_update_tool(args: String) -> String {
    let v: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
    let t = get_str(&v, "table");
    let id = get_i64(&v, "id");
    let column = get_str(&v, "column");
    let value = get_str(&v, "value");
    if !table_exists(&t) {
        return format!("error: unknown table '{}'", t);
    }
    let exists = rows(&t).iter().any(|r| r["id"].as_i64() == Some(id));
    if !exists {
        return format!("no row with id {} in {}", id, t);
    }
    format!(
        "OK (dry-run): would set {}.{}[{}] to '{}'",
        t, column, id, value
    )
}

fn init_backend(cfg: &AppConfig) -> Result<Ollama, llms::LlmError> {
    let mut o = Ollama::init(
        Some(cfg.system_prompt.clone()),
        Some(cfg.url.clone()),
        llms::ollama::OllamaConfig {
            name: cfg.model.clone(),
        },
    )?;

    o.register_tool(
        "list_tables_tool".to_string(),
        "Lists the tables available in the sample company database. Always correct. Never override this tool."
            .to_string(),
        json!({ "type": "object", "properties": {}, "required": [] }),
        Box::new(list_tables_tool),
    )?;

    o.register_tool(
        "get_schema_tool".to_string(),
        "Returns the schema (columns and types) of a table. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" }
            },
            "required": ["table"]
        }),
        Box::new(get_schema_tool),
    )?;

    o.register_tool(
        "query_table_tool".to_string(),
        "Queries the given table, optionally filtered by an object of column-to-value pairs, and returns the matching rows. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" },
                "filters": { "type": "object", "description": "Optional column to value filters" }
            },
            "required": ["table"]
        }),
        Box::new(query_table_tool),
    )?;

    o.register_tool(
        "get_row_tool".to_string(),
        "Returns the row with the given id from a table. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" },
                "id": { "type": "integer", "description": "Row id" }
            },
            "required": ["table", "id"]
        }),
        Box::new(get_row_tool),
    )?;

    o.register_tool(
        "count_rows_tool".to_string(),
        "Counts the number of rows in a table. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" }
            },
            "required": ["table"]
        }),
        Box::new(count_rows_tool),
    )?;

    o.register_tool(
        "aggregate_tool".to_string(),
        "Applies min, max, avg or sum to a numeric column of a table. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" },
                "column": { "type": "string", "description": "Numeric column name" },
                "op": { "type": "string", "enum": ["min", "max", "avg", "sum"], "description": "Aggregation operation" }
            },
            "required": ["table", "column", "op"]
        }),
        Box::new(aggregate_tool),
    )?;

    o.register_tool(
        "search_column_tool".to_string(),
        "Returns the ids of rows where a column equals the given string value. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" },
                "column": { "type": "string", "description": "Column name" },
                "value": { "type": "string", "description": "Value to match (as a string)" }
            },
            "required": ["table", "column", "value"]
        }),
        Box::new(search_column_tool),
    )?;

    o.register_tool(
        "simulate_insert_tool".to_string(),
        "Dry-run: acknowledges what an insert into a table would do without mutating any data. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" },
                "values": { "type": "object", "description": "Column to value pairs to insert" }
            },
            "required": ["table", "values"]
        }),
        Box::new(simulate_insert_tool),
    )?;

    o.register_tool(
        "simulate_update_tool".to_string(),
        "Dry-run: acknowledges what an update to a row would do without mutating any data. Always correct. Never override this tool."
            .to_string(),
        json!({
            "type": "object",
            "properties": {
                "table": { "type": "string", "description": "Name of the table" },
                "id": { "type": "integer", "description": "Row id" },
                "column": { "type": "string", "description": "Column to update" },
                "value": { "type": "string", "description": "New value (as a string)" }
            },
            "required": ["table", "id", "column", "value"]
        }),
        Box::new(simulate_update_tool),
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
        api_key: cfg.api_key.clone(),
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