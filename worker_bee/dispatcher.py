"""Dispatch prompts to a local model via Ollama."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass


DEFAULT_ENDPOINT = "http://localhost:11434"


@dataclass
class Response:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


def dispatch(
    prompt: str,
    *,
    model: str = "qwen3:27b",
    endpoint: str = DEFAULT_ENDPOINT,
) -> Response:
    """Send a prompt to Ollama and return the response."""
    url = f"{endpoint}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach Ollama at {endpoint}: {e}") from e

    return Response(
        text=body.get("response", ""),
        model=body.get("model", model),
        prompt_tokens=body.get("prompt_eval_count", 0),
        completion_tokens=body.get("eval_count", 0),
    )


def dispatch_batch(
    prompts: list[tuple[dict, str]],
    *,
    model: str = "qwen3:27b",
    endpoint: str = DEFAULT_ENDPOINT,
) -> list[tuple[dict, Response]]:
    """Dispatch multiple (issue, prompt) pairs sequentially.

    Returns list of (issue, response) tuples.
    """
    results = []
    for issue, prompt in prompts:
        resp = dispatch(prompt, model=model, endpoint=endpoint)
        results.append((issue, resp))
    return results
