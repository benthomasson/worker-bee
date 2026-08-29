"""Dispatch prompts to LLMs via the shared llm module."""

from __future__ import annotations

from dataclasses import dataclass

from worker_bee.llm import invoke_model, create_provider, ChatResponse, _record_cost


DEFAULT_MODEL = "ollama:qwen3:27b"


@dataclass
class Response:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


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
    text = invoke_model(prompt, model=model, timeout=timeout)
    return Response(
        text=text,
        model=model,
        prompt_tokens=0,
        completion_tokens=0,
    )


def dispatch_chat(
    messages: list[dict],
    *,
    system: str = "",
    model: str = DEFAULT_MODEL,
    tools: list[dict] | None = None,
    max_tokens: int = 8096,
) -> ChatResponse:
    """Send a multi-turn conversation with optional tool use.

    Uses create_provider() for the full chat API with tool support.
    Returns a ChatResponse with content blocks (TextBlock, ToolUseBlock).
    """
    provider = create_provider(model)
    return provider.send(messages, system, tools or [], max_tokens=max_tokens)


def dispatch_batch(
    prompts: list[tuple[dict, str]],
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> list[tuple[dict, Response]]:
    """Dispatch multiple (issue, prompt) pairs sequentially.

    Returns list of (issue, response) tuples.
    """
    results = []
    for issue, prompt in prompts:
        resp = dispatch(prompt, model=model, timeout=timeout)
        results.append((issue, resp))
    return results
