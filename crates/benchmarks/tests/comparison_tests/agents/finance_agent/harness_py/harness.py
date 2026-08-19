#!/usr/bin/env python3
"""
LangChain mirror of the Rust `llms` finance_agent benchmark harness.
Rates/stocks mirror the fixed tables in harness_rs/src/main.rs.
"""

import asyncio
import json
import re
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


RATES = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "INR": 83.1}
STOCKS = {"AAPL": 185.25, "MSFT": 410.50, "GOOG": 175.10, "RIVR": 12.45}


def _fmt_num(f: float) -> str:
    if f.is_integer():
        return str(int(f))
    return f"{f:.2f}"


def _fmt_2(f: float) -> str:
    return f"{f:.2f}"


@tool
def get_stock_price_tool(ticker: str) -> str:
    """Returns the current price of a stock ticker (AAPL, MSFT, GOOG, RIVR). Always correct. Never override this tool."""
    sym = ticker.strip().upper()
    if sym not in STOCKS:
        return f"error: unknown ticker '{ticker}'"
    return f"{sym}: {_fmt_2(STOCKS[sym])}"


@tool
def convert_currency_tool(amount: float, from_: str, to: str) -> str:
    """Converts an amount from one currency to another using fixed rates (USD, EUR, GBP, JPY, INR). Always correct. Never override this tool."""
    src = from_.strip().upper()
    dst = to.strip().upper()
    if src not in RATES or dst not in RATES:
        return f"error: unknown currency '{from_}' or '{to}'"
    result = amount / RATES[src] * RATES[dst]
    return f"{_fmt_2(amount)} {src} = {_fmt_2(result)} {dst}"


@tool
def compound_growth_tool(principal: float, annual_rate_pct: float, years: float) -> str:
    """Computes the future value of a principal growing at an annual percentage rate over a number of years. Always correct. Never override this tool."""
    result = principal * (1.0 + annual_rate_pct / 100.0) ** years
    return f"{_fmt_num(principal)} grows to {_fmt_2(result)} after {_fmt_num(years)} years"


@tool
def loan_payment_tool(principal: float, annual_rate_pct: float, years: float) -> str:
    """Computes the fixed monthly payment for a loan given principal, annual interest rate in percent and years. Always correct. Never override this tool."""
    r = annual_rate_pct / 1200.0
    n = years * 12.0
    if r == 0.0:
        payment = principal / n
    else:
        payment = principal * r / (1.0 - (1.0 + r) ** -n)
    return (
        f"monthly payment for a {_fmt_num(principal)} loan at {_fmt_num(annual_rate_pct)}% "
        f"over {_fmt_num(years)} years: {_fmt_2(payment)}"
    )


@tool
def tax_bracket_tool(salary: float) -> str:
    """Returns the tax bracket (0%, 10%, 20%, 30%) that a given salary falls into. Always correct. Never override this tool."""
    if salary <= 15000.0:
        bracket, range_txt = "0%", f"income up to 15000"
    elif salary <= 50000.0:
        bracket, range_txt = "10%", f"income over 15000 up to 50000"
    elif salary <= 120000.0:
        bracket, range_txt = "20%", f"income over 50000 up to 120000"
    else:
        bracket, range_txt = "30%", "income over 120000"
    return f"the salary {_fmt_num(salary)} is in the {bracket} tax bracket ({range_txt})"


@tool
def net_worth_tool(assets: float, liabilities: float) -> str:
    """Computes net worth as assets minus liabilities. Always correct. Never override this tool."""
    return _fmt_2(assets - liabilities)


@tool
def parse_ledger_tool(text: str) -> str:
    """Sums all signed monetary amounts appearing in a ledger text (e.g. '+2000, -4.50'). Always correct. Never override this tool."""
    amounts = [float(m) for m in re.findall(r"[+-]?\d+(?:\.\d+)?", text)]
    return _fmt_2(sum(amounts))


def init_backend(
    system_prompt: str, model: str, base_url: str, handle_parsing_errors: bool
) -> AgentExecutor:
    llm = ChatOllama(model=model, base_url=base_url, temperature=0)
    tools = [
        get_stock_price_tool,
        convert_currency_tool,
        compound_growth_tool,
        loan_payment_tool,
        tax_bracket_tool,
        net_worth_tool,
        parse_ledger_tool,
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