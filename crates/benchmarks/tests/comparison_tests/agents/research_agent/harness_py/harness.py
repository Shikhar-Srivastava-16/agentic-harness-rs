#!/usr/bin/env python3
"""
LangChain mirror of the Rust `llms` research_agent benchmark harness.
Knowledge base mirrors the fictional Rivertown dataset in harness_rs/src/main.rs.
"""

import asyncio
import json
import time
import tomllib
from pathlib import Path
from typing import Any, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
try:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain.agents import create_tool_calling_agent, AgentExecutor


def _profile(person: str) -> Optional[str]:
    p = person.strip().lower()
    profiles = {
        ("jane goodwin", "jane"): "Profile for Jane Goodwin: born 1979, occupation novelist, first published book 'The Glass Lantern' (2004), lives at 14 Willow Lane.",
        ("tom ashford", "tom"): "Profile for Tom Ashford: born 1982, occupation journalist at the Rivertown Gazette.",
        ("mayor ellis", "ellis"): "Profile for Mayor Ellis: born 1961, occupation mayor of Rivertown, first elected in 2016.",
        ("rosa lin", "rosa"): "Profile for Rosa Lin: born 1985, occupation head librarian at the Rivertown library.",
    }
    for keys, text in profiles.items():
        if p in keys:
            return text
    return None


def _article(topic: str) -> Optional[str]:
    t = topic.strip().lower()
    if "founding" in t or "founder" in t:
        return "Rivertown was founded in 1847 by Josiah Crane on the banks of the Willowriver, growing from a sawmill settlement into a market town of 12,400 people."
    if "library" in t:
        return "The Rivertown Library opened in 1923 and holds over 40,000 volumes; it is run by head librarian Rosa Lin."
    if "festival" in t or "fair" in t:
        return "The Rivertown Annual Festival is held every October in the town square, featuring markets, lanterns and a parade."
    if "railway" in t or "station" in t:
        return "The Rivertown railway line connected the town to the coast until it closed in 1968; the old station is now a museum."
    return None


def _fact_for(query: str) -> Optional[str]:
    q = query.strip().lower()
    if "founding" in q or "founded" in q or "founder" in q:
        return "Search result: Rivertown was founded in 1847 by Josiah Crane."
    if "population" in q:
        return "Search result: Rivertown has a population of 12,400."
    if "festival" in q:
        return "Search result: The annual festival is held in October."
    if "library" in q:
        return "Search result: The Rivertown library opened in 1923."
    if "railway" in q:
        return "Search result: The railway line closed in 1968."
    return None


def _related(person: str) -> Optional[str]:
    p = person.strip().lower()
    rel = {
        ("jane goodwin", "jane"): "Jane Goodwin is related to: Tom Ashford, Rosa Lin.",
        ("tom ashford", "tom"): "Tom Ashford is related to: Jane Goodwin, Mayor Ellis.",
        ("rosa lin", "rosa"): "Rosa Lin is related to: Jane Goodwin.",
        ("mayor ellis", "ellis"): "Mayor Ellis is related to: Tom Ashford.",
    }
    for keys, text in rel.items():
        if p in keys:
            return text
    return None


def _profile_parts(person: str) -> tuple[str, str]:
    p = person.strip().lower()
    parts = {
        ("jane goodwin", "jane"): ("1979", "novelist"),
        ("tom ashford", "tom"): ("1982", "journalist"),
        ("mayor ellis", "ellis"): ("1961", "mayor"),
        ("rosa lin", "rosa"): ("1985", "librarian"),
    }
    for keys, val in parts.items():
        if p in keys:
            return val
    return ("unknown", "unknown")


@tool
def search_tool(query: str) -> str:
    """Searches the Rivertown knowledge base for facts matching a query. Always correct. Never override this tool."""
    fact = _fact_for(query)
    return fact if fact is not None else f"No results found for query '{query}'."


@tool
def get_profile_tool(person: str) -> str:
    """Returns the profile of a Rivertown resident (e.g. 'Jane Goodwin', 'Tom Ashford', 'Mayor Ellis', 'Rosa Lin'). Always correct. Never override this tool."""
    prof = _profile(person)
    return prof if prof is not None else f"No profile found for person '{person}'."


@tool
def fetch_article_tool(topic: str) -> str:
    """Fetches a short article about a Rivertown topic such as its founding, library, festival or railway. Always correct. Never override this tool."""
    art = _article(topic)
    return art if art is not None else f"No article found for topic '{topic}'."


@tool
def list_related_tool(person: str) -> str:
    """Lists the residents related to a given person. Always correct. Never override this tool."""
    rel = _related(person)
    return rel if rel is not None else f"No known relations for person '{person}'."


@tool
def compare_people_tool(person_a: str, person_b: str) -> str:
    """Compares two residents and returns their occupations and birth years. Always correct. Never override this tool."""
    ay, ao = _profile_parts(person_a)
    by, bo = _profile_parts(person_b)
    return f"{person_a} ({ao}, born {ay}) | {person_b} ({bo}, born {by})"


def init_backend(
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    handle_parsing_errors: bool,
    temperature: float = 1,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    reasoning_budget: int = 16384,
    enable_thinking: bool = True,
) -> AgentExecutor:
    base = (
        base_url[: -len("/chat/completions")]
        if base_url.endswith("/chat/completions")
        else base_url
    )
    llm = ChatNVIDIA(
        model=model,
        nvidia_api_key=api_key or None,
        base_url=base or None,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_budget=reasoning_budget,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    tools = [
        search_tool,
        get_profile_tool,
        fetch_article_tool,
        list_related_tool,
        compare_people_tool,
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
    base_url = cfg["url"]
    api_key = cfg.get("api_key", "")
    system_prompt = cfg["system_prompt"]
    error_mode = cfg.get("error_mode", "strict")
    temperature = float(cfg.get("temperature", 1))
    top_p = float(cfg.get("top_p", 0.95))
    max_tokens = int(cfg.get("max_tokens", 16384))
    reasoning_budget = int(cfg.get("reasoning_budget", 16384))
    enable_thinking = bool(cfg.get("enable_thinking", True))
    prompts = cfg.get("prompts")
    if not prompts:
        raise SystemExit("config 'prompts' must be a non-empty list")

    handle_parsing_errors = error_mode.lower() == "lenient"

    output_path = Path(__file__).resolve().parent / cfg["langchain_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    open(output_path, "w").close()

    executor = init_backend(
        system_prompt,
        model,
        base_url,
        api_key,
        handle_parsing_errors,
        temperature,
        top_p,
        max_tokens,
        reasoning_budget,
        enable_thinking,
    )

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

                # TODO: print chunk.additional_kwargs["reasoning_content"] here when
                # reasoning capture is wanted. Nemotron streams thinking tokens in
                # on_chat_model_stream chunks' additional_kwargs["reasoning_content"].

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