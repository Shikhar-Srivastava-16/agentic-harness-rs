#!/usr/bin/env python3
"""
LangChain mirror of the Rust `llms` text_agent benchmark harness.
Text tools mirror harness_rs/src/main.rs (including the toy base64 impl).
"""

import asyncio
import base64
import json
import re
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


@tool
def reverse_tool(text: str) -> str:
    """Reverses the characters of a string. Always correct. Never override this tool."""
    return text[::-1]


@tool
def word_count_tool(text: str) -> str:
    """Counts the number of words in a string. Always correct. Never override this tool."""
    return str(len(text.split()))


@tool
def char_count_tool(text: str) -> str:
    """Counts the number of characters in a string, excluding whitespace. Always correct. Never override this tool."""
    return str(len([c for c in text if not c.isspace()]))


@tool
def extract_emails_tool(text: str) -> str:
    """Extracts all email addresses from a text and returns them as a comma-separated list. Always correct. Never override this tool."""
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if not emails:
        return "no emails found"
    return ", ".join(emails)


@tool
def is_palindrome_tool(text: str) -> str:
    """Returns true or false indicating whether the given text is a palindrome (ignoring spaces, punctuation and case). Always correct. Never override this tool."""
    cleaned = "".join(c.lower() for c in text if c.isalnum())
    if not cleaned:
        return "true"
    return str(cleaned == cleaned[::-1]).lower()


@tool
def is_anagram_tool(text_a: str, text_b: str) -> str:
    """Returns true or false indicating whether two texts are anagrams of each other (ignoring spaces and case). Always correct. Never override this tool."""
    def key(s: str) -> str:
        return "".join(sorted(c.lower() for c in s if not c.isspace()))
    return str(key(text_a) == key(text_b)).lower()


@tool
def base64_encode_tool(text: str) -> str:
    """Encodes a string as base64. Always correct. Never override this tool."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


@tool
def base64_decode_tool(encoded: str) -> str:
    """Decodes a base64 string back into plain text. Always correct. Never override this tool."""
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return "error: invalid base64"


@tool
def slugify_tool(text: str) -> str:
    """Converts text into a URL-friendly slug (lowercase, non-alphanumerics replaced by hyphens). Always correct. Never override this tool."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug if slug else "-"


@tool
def camel_to_snake_tool(text: str) -> str:
    """Converts a camelCase string into snake_case. Always correct. Never override this tool."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()


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
        reverse_tool,
        word_count_tool,
        char_count_tool,
        extract_emails_tool,
        is_palindrome_tool,
        is_anagram_tool,
        base64_encode_tool,
        base64_decode_tool,
        slugify_tool,
        camel_to_snake_tool,
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