"""Dispatch prompts to LLMs via the shared llm module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import inspect
import math
import threading
import time

from worker_bee.llm import (
    invoke_model,
    create_provider,
    ChatResponse,
    get_last_usage,
    reset_last_usage,
)


DEFAULT_MODEL = "ollama:qwen3.8:27b"


class _RequestRateLimiter:
    """Thread-safe spacing limiter shared by calls for one model."""

    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / requests_per_second
        self._next_request = 0.0
        self._last_request: float | None = None
        self._enabled = True
        self._lock = threading.Lock()

    def set_rate(self, requests_per_second: float | None) -> None:
        """Change or disable the rate without replacing the shared object."""
        with self._lock:
            if requests_per_second is None:
                self._enabled = False
                return
            self.interval = 1.0 / requests_per_second
            self._enabled = True
            if self._last_request is not None:
                self._next_request = max(
                    self._next_request, self._last_request + self.interval
                )

    def acquire(self) -> None:
        # Keep the lock while sleeping. Releasing it after reserving a slot
        # permits a delayed thread to start after a later reservation, which
        # can violate the configured spacing at the provider boundary.
        with self._lock:
            if not self._enabled:
                return
            now = time.monotonic()
            start = max(now, self._next_request)
            delay = start - now
            if delay > 0:
                time.sleep(delay)
            self._last_request = start
            self._next_request = start + self.interval


_rate_limiters: dict[str, _RequestRateLimiter] = {}
_rate_limiters_lock = threading.Lock()


def configure_rate_limit(model: str, requests_per_second: float | None) -> None:
    """Set or clear the process-wide request rate limit for ``model``.

    The limiter is shared by ``dispatch``, ``dispatch_batch``, ``dispatch_chat``,
    and concurrent callers in this process. It spaces request starts; it does
    not enforce provider token quotas or coordinate separate processes.
    """
    if requests_per_second is not None:
        if (
            isinstance(requests_per_second, bool)
            or not isinstance(requests_per_second, (int, float))
            or not math.isfinite(requests_per_second)
            or requests_per_second <= 0
        ):
            raise ValueError("requests_per_second must be a finite positive number")
    with _rate_limiters_lock:
        if model in _rate_limiters:
            # Retain the object so calls that already obtained it observe the
            # same configuration during reconfiguration and clearing.
            _rate_limiters[model].set_rate(
                None if requests_per_second is None else float(requests_per_second)
            )
        elif requests_per_second is not None:
            _rate_limiters[model] = _RequestRateLimiter(float(requests_per_second))


def _acquire_rate_limit(model: str) -> None:
    with _rate_limiters_lock:
        limiter = _rate_limiters.get(model)
    if limiter is not None:
        limiter.acquire()


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
    _acquire_rate_limit(model)
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
    timeout: int = 300,
) -> ChatResponse:
    """Send a multi-turn conversation with optional tool use.

    Uses create_provider() for the full chat API with tool support. ``timeout``
    is passed to providers that support the shared timeout contract.
    Returns a ChatResponse with content blocks (TextBlock, ToolUseBlock).
    """
    _acquire_rate_limit(model)
    provider = create_provider(model)
    send = provider.send
    # Keep the pre-timeout provider API usable for custom providers while
    # forwarding the timeout to providers that implement the shared contract.
    try:
        parameters = inspect.signature(send).parameters.values()
        accepts_timeout = any(
            parameter.name == "timeout"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_timeout = True
    kwargs = {"max_tokens": max_tokens, "num_ctx": num_ctx}
    if accepts_timeout:
        kwargs["timeout"] = timeout
    return send(messages, system, tools or [], **kwargs)


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
