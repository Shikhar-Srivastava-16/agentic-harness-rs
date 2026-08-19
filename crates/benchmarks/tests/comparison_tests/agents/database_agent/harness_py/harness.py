#!/usr/bin/env python3
"""
LangChain mirror of the Rust `llms` database_agent benchmark harness.
Sample company db mirrors the dataset in harness_rs/src/main.rs.
"""

import asyncio
import json
import time
import tomllib
from pathlib import Path
from typing import Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
try:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain.agents import create_tool_calling_agent, AgentExecutor


EMPLOYEES = [
    {"id": 1, "name": "Alice", "dept": "Sales", "salary": 55000},
    {"id": 2, "name": "Bob", "dept": "Engineering", "salary": 82000},
    {"id": 3, "name": "Carol", "dept": "Sales", "salary": 61000},
    {"id": 4, "name": "Dan", "dept": "Marketing", "salary": 58000},
    {"id": 5, "name": "Eve", "dept": "Engineering", "salary": 98000},
    {"id": 6, "name": "Frank", "dept": "HR", "salary": 47000},
    {"id": 7, "name": "Grace", "dept": "Engineering", "salary": 74000},
    {"id": 8, "name": "Heidi", "dept": "Marketing", "salary": 52000},
    {"id": 9, "name": "Ivan", "dept": "Sales", "salary": 66000},
    {"id": 10, "name": "Judy", "dept": "HR", "salary": 50000},
]

PRODUCTS = [
    {"id": 1, "name": "Ergonomic Chair", "price": 249.99, "stock": 25},
    {"id": 2, "name": "Mechanical Keyboard", "price": 119.50, "stock": 8},
    {"id": 3, "name": "4K Monitor", "price": 379.00, "stock": 4},
    {"id": 4, "name": "USB-C Dock", "price": 89.99, "stock": 42},
    {"id": 5, "name": "Webcam", "price": 129.00, "stock": 15},
    {"id": 6, "name": "Desk Lamp", "price": 34.99, "stock": 3},
]

ORDERS = [
    {"id": 1, "product_id": 1, "quantity": 2, "amount": 499.98},
    {"id": 2, "product_id": 3, "quantity": 1, "amount": 379.00},
    {"id": 3, "product_id": 2, "quantity": 3, "amount": 358.50},
    {"id": 4, "product_id": 6, "quantity": 5, "amount": 174.95},
    {"id": 5, "product_id": 5, "quantity": 2, "amount": 258.00},
    {"id": 6, "product_id": 4, "quantity": 1, "amount": 89.99},
]

TABLES = {"employees": EMPLOYEES, "products": PRODUCTS, "orders": ORDERS}


def _fmt_num(f: float) -> str:
    if f.is_integer():
        return str(int(f))
    return f"{f:.2f}"


def _to_str(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return _fmt_num(float(v))
    return str(v)


@tool
def list_tables_tool() -> str:
    """Lists the tables available in the sample company database. Always correct. Never override this tool."""
    return "employees, products, orders"


@tool
def get_schema_tool(table: str) -> str:
    """Returns the schema (columns and types) of a table. Always correct. Never override this tool."""
    schemas = {
        "employees": "employees: id(int), name(string), dept(string), salary(int)",
        "products": "products: id(int), name(string), price(number), stock(int)",
        "orders": "orders: id(int), product_id(int), quantity(int), amount(number)",
    }
    if table not in schemas:
        return f"error: unknown table '{table}'"
    return schemas[table]


@tool
def query_table_tool(table: str, filters: Optional[dict] = None) -> str:
    """Queries the given table, optionally filtered by an object of column-to-value pairs, and returns the matching rows. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    filters = filters or {}
    matched = [
        r for r in TABLES[table]
        if all(_to_str(r.get(k)) == _to_str(fv) for k, fv in filters.items())
    ]
    if not matched:
        return f"no rows found in {table} matching the filters"
    return "\n".join(json.dumps(r) for r in matched)


@tool
def get_row_tool(table: str, id: int) -> str:
    """Returns the row with the given id from a table. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    for r in TABLES[table]:
        if r["id"] == id:
            return json.dumps(r)
    return f"no row with id {id} in {table}"


@tool
def count_rows_tool(table: str) -> str:
    """Counts the number of rows in a table. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    return str(len(TABLES[table]))


@tool
def aggregate_tool(table: str, column: str, op: str) -> str:
    """Applies min, max, avg or sum to a numeric column of a table. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    vals = []
    for r in TABLES[table]:
        v = r.get(column)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return f"error: column '{column}' has no numeric values"
    if op == "min":
        result = min(vals)
    elif op == "max":
        result = max(vals)
    elif op == "sum":
        result = sum(vals)
    elif op in ("avg", "mean"):
        result = sum(vals) / len(vals)
    else:
        return f"error: unsupported op '{op}' (use min, max, avg or sum)"
    return _fmt_num(result)


@tool
def search_column_tool(table: str, column: str, value: str) -> str:
    """Returns the ids of rows where a column equals the given string value. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    ids = [r["id"] for r in TABLES[table] if _to_str(r.get(column)) == value]
    if not ids:
        return f"no rows in {table} where {column} = '{value}'"
    return f"rows in {table} where {column} = '{value}': " + ", ".join(str(i) for i in ids)


@tool
def simulate_insert_tool(table: str, values: dict) -> str:
    """Dry-run: acknowledges what an insert into a table would do without mutating any data. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    next_id = len(TABLES[table]) + 1
    return f"OK (dry-run): would insert into {table} the row {json.dumps(values)} with id {next_id}"


@tool
def simulate_update_tool(table: str, id: int, column: str, value: str) -> str:
    """Dry-run: acknowledges what an update to a row would do without mutating any data. Always correct. Never override this tool."""
    if table not in TABLES:
        return f"error: unknown table '{table}'"
    if not any(r["id"] == id for r in TABLES[table]):
        return f"no row with id {id} in {table}"
    return f"OK (dry-run): would set {table}.{column}[{id}] to '{value}'"


def init_backend(
    system_prompt: str, model: str, base_url: str, handle_parsing_errors: bool
) -> AgentExecutor:
    llm = ChatOllama(model=model, base_url=base_url, temperature=0)
    tools = [
        list_tables_tool,
        get_schema_tool,
        query_table_tool,
        get_row_tool,
        count_rows_tool,
        aggregate_tool,
        search_column_tool,
        simulate_insert_tool,
        simulate_update_tool,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=handle_parsing_errors,
    )


class BenchTracker:
    """Processes astream_events and emits JSONL benchmark records."""

    def __init__(self, model_name: str, output_path: str):
        self.model_name = model_name
        self.output_path = output_path
        self.invoke_start: Optional[float] = None
        self.is_first_llm: bool = True
        self.tool_start: Optional[float] = None
        self.tool_name: Optional[str] = None

    def emit(self, kind: str, duration_ms: float, tool: Optional[str] = None) -> None:
        record = {
            "ts": int(time.time() * 1000),
            "kind": kind,
            "duration_ms": round(duration_ms, 6),
            "tool": tool,
            "model": self.model_name,
            "framework": "langchain-agentexecutor",
        }
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def start(self) -> None:
        self.invoke_start = time.perf_counter()
        self.is_first_llm = True
        self.tool_start = None
        self.tool_name = None

    def finish(self) -> None:
        if self.invoke_start is not None:
            query_ms = (time.perf_counter() - self.invoke_start) * 1000
            self.emit("query", query_ms)
            self.invoke_start = None

    def handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event", "")
        now = time.perf_counter()

        if kind in ("on_llm_start", "on_chat_model_start"):
            if self.is_first_llm:
                if self.invoke_start is not None:
                    send_ms = (now - self.invoke_start) * 1000
                    self.emit("send", send_ms)
                self.is_first_llm = False
            else:
                if self.tool_start is not None:
                    cycle_ms = (now - self.tool_start) * 1000
                    self.emit("tool_cycle", cycle_ms, tool=self.tool_name)

        elif kind == "on_tool_start":
            self.tool_start = now
            evt_name = event.get("name", "")
            self.tool_name = evt_name if evt_name else "unknown"

        elif kind == "on_tool_end":
            if self.tool_start is not None:
                tool_ms = (now - self.tool_start) * 1000
                evt_name = event.get("name", "")
                name = evt_name if evt_name else (self.tool_name or "unknown")
                self.emit("tool_time", tool_ms, tool=name)
                self.tool_start = None
                self.tool_name = None


async def run_bench() -> None:
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    iterations = int(cfg["iterations"])
    model = cfg["model"]
    ollama_url = cfg["url"]
    system_prompt = cfg["system_prompt"]
    error_mode = cfg.get("error_mode", "strict")
    prompts = cfg.get("prompts")
    if not prompts:
        raise SystemExit("config 'prompts' must be a non-empty list")

    handle_parsing_errors = error_mode.lower() == "lenient"

    output_path = Path(__file__).resolve().parent / cfg["langchain_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    open(output_path, "w").close()

    executor = init_backend(system_prompt, model, ollama_url, handle_parsing_errors)

    print(f"Running {iterations} iterations with model '{model}'")

    results = []
    for i in range(iterations):
        prompt = prompts[i % len(prompts)]
        print(f'  [{i + 1}/{iterations}] prompt: "{prompt}" ... ', end="", flush=True)

        tracker = BenchTracker(model, output_path)
        tracker.start()

        try:
            answer = ""
            async for event in executor.astream_events({"input": prompt}, version="v2"):
                tracker.handle_event(event)

                if event.get("event") == "on_chain_end" and event.get("name") == "AgentExecutor":
                    data = event.get("data", {})
                    out = data.get("output", {})
                    if isinstance(out, dict):
                        answer = out.get("output", "") or ""
                    elif isinstance(out, str):
                        answer = out

            tracker.finish()
            print(f"OK ({len(answer)} chars)")
            results.append({"prompt": prompt, "output": answer, "error": None})
        except Exception as e:
            tracker.finish()
            print(f"ERROR: {e}")
            results.append({"prompt": prompt, "output": None, "error": str(e)})

    summary_path = "bench_results.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": model,
                "iterations": iterations,
                "timestamp": time.time(),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Done. Results written to {output_path} and {summary_path}")


def main() -> None:
    asyncio.run(run_bench())


if __name__ == "__main__":
    main()