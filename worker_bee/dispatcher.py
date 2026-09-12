"""Dispatch prompts to LLMs via the shared llm module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from worker_bee.llm import (
    invoke_model,
    create_provider,
    ChatResponse,
    get_last_usage,
    reset_last_usage,
)


DEFAULT_MODEL = "ollama:qwen3.8:27b"


@dataclass
class Response:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


def dispatch(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> Response:
    """Send a single prompt to a model and return the response.

    Uses invoke_model() for simple prompt-in/response-out calls.
    Works with any supported model string: ollama:*, claude, gemini,
    api:*, vertex:*.
    """
    reset_last_usage()
    text = invoke_model(prompt, model=model, timeout=timeout)
    usage = get_last_usage()
    return Response(
        text=text,
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )


def dispatch_chat(
    messages: list[dict],
    *,
    system: str = "",
    model: str = DEFAULT_MODEL,
    tools: list[dict] | None = None,
    max_tokens: int = 8096,
    num_ctx: int | None = None,
) -> ChatResponse:
    """Send a multi-turn conversation with optional tool use.

    Uses create_provider() for the full chat API with tool support.
    Returns a ChatResponse with content blocks (TextBlock, ToolUseBlock).
    """
    provider = create_provider(model)
    return provider.send(messages, system, tools or [], max_tokens=max_tokens, num_ctx=num_ctx)


def dispatch_batch(
    prompts: list[tuple[dict, str]],
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    max_workers: int = 1,
    retries: int = 2,
) -> list[tuple[dict, Response]]:
    """Dispatch prompts concurrently, retrying failures independently.

    Results retain input order. A failed item is returned as a Response with
    ``error`` populated rather than aborting the whole batch.
    """
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    if not prompts:
        return []

    def run_one(issue_prompt):
        issue, prompt = issue_prompt
        last_error = None
        for _ in range(retries + 1):
            try:
                return issue, dispatch(prompt, model=model, timeout=timeout)
            except Exception as exc:
                last_error = exc
        return issue, Response(
            text="",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            error=f"{type(last_error).__name__}: {last_error}",
        )

    workers = max(1, min(max_workers, len(prompts)))
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, item): index for index, item in enumerate(prompts)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
