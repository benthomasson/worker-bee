"""Shared LLM invocation via CLI subprocesses and API providers.

Supports named models (claude, gemini), Claude submodels via
'claude:<model>' syntax, ollama models via 'ollama:<model>',
and API-based providers via 'api:<model>' and 'vertex:<model>'.

CLI-based models pipe prompts to stdin and read responses from stdout.
API-based models use LangChain adapters with optional Langfuse tracing.

Cost tracking: CLI models use --output-format json to capture token
counts and costs. Use get_cost_summary() to retrieve accumulated stats.
"""

import json
import os
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import dataclass, field

try:
    from langchain_anthropic import ChatAnthropic
    _HAS_LANGCHAIN_ANTHROPIC = True
except ImportError:
    ChatAnthropic = None
    _HAS_LANGCHAIN_ANTHROPIC = False

try:
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex
    _HAS_LANGCHAIN_VERTEX = True
except ImportError:
    ChatAnthropicVertex = None
    _HAS_LANGCHAIN_VERTEX = False

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    _HAS_LANGFUSE = True
except ImportError:
    LangfuseCallbackHandler = None
    _HAS_LANGFUSE = False


MODEL_COMMANDS = {
    "claude": ["claude", "-p", "--output-format", "json"],
    "gemini": ["gemini", "--skip-trust", "-o", "json", "-p", ""],
}

_langfuse_handler = None
_langfuse_checked = False

_cost_lock = threading.Lock()

_cost_tracker = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost_usd": 0.0,
    "by_model": {},
}


def reset_cost_tracker():
    """Reset accumulated cost/token stats."""
    with _cost_lock:
        _cost_tracker["calls"] = 0
        _cost_tracker["input_tokens"] = 0
        _cost_tracker["output_tokens"] = 0
        _cost_tracker["total_cost_usd"] = 0.0
        _cost_tracker["by_model"] = {}


def get_cost_summary() -> dict:
    """Return accumulated cost/token stats across all LLM calls."""
    import copy
    with _cost_lock:
        return copy.deepcopy(_cost_tracker)


def format_cost_summary() -> str:
    """Format cost summary as a human-readable string."""
    with _cost_lock:
        s = _cost_tracker
        if s["calls"] == 0:
            return ""
        parts = []
        if s["total_cost_usd"] > 0:
            parts.append(f"${s['total_cost_usd']:.4f}")
        parts.append(f"{s['input_tokens']:,} input + {s['output_tokens']:,} output tokens")
        parts.append(f"{s['calls']} call(s)")
        return "Cost: " + " | ".join(parts)


def _record_cost(model: str, input_tokens: int, output_tokens: int, cost_usd: float):
    """Record token/cost stats from one LLM call."""
    with _cost_lock:
        _cost_tracker["calls"] += 1
        _cost_tracker["input_tokens"] += input_tokens
        _cost_tracker["output_tokens"] += output_tokens
        _cost_tracker["total_cost_usd"] += cost_usd

        if model not in _cost_tracker["by_model"]:
            _cost_tracker["by_model"][model] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0,
            }
        m = _cost_tracker["by_model"][model]
        m["calls"] += 1
        m["input_tokens"] += input_tokens
        m["output_tokens"] += output_tokens
        m["total_cost_usd"] += cost_usd


def _parse_cli_json(output: str, model: str) -> str:
    """Parse JSON output from CLI, extract response text and record costs.

    Falls back to returning raw output if JSON parsing fails.
    """
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return output

    if not isinstance(data, dict):
        return output

    if model.startswith("gemini") or model.startswith("gemini:"):
        text = data.get("response", output)
        stats = data.get("stats", {})
        input_tokens = 0
        output_tokens = 0
        for model_stats in stats.get("models", {}).values():
            tokens = model_stats.get("tokens", {})
            input_tokens += tokens.get("input", 0)
            output_tokens += tokens.get("candidates", 0)
        _record_cost(model, input_tokens, output_tokens, 0.0)
        return text

    text = data.get("result", output)
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost_usd = data.get("total_cost_usd", 0.0)
    _record_cost(model, input_tokens, output_tokens, cost_usd)
    return text


def _get_langfuse_handler():
    global _langfuse_handler, _langfuse_checked
    if _langfuse_checked:
        return _langfuse_handler
    _langfuse_checked = True
    if not _HAS_LANGFUSE:
        return None
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    _langfuse_handler = LangfuseCallbackHandler()
    return _langfuse_handler


def resolve_model_cmd(model: str) -> list[str]:
    """Resolve a model name to a CLI command list.

    Supports named models ('claude', 'gemini'), Claude submodels
    via 'claude:<model>' (e.g. 'claude:sonnet'), Gemini submodels
    via 'gemini:<model>' (e.g. 'gemini:gemini-2.5-flash'), and
    ollama models via 'ollama:<model>' syntax (e.g. 'ollama:gemma3:4b').

    API-based models ('api:<model>', 'vertex:<model>') are handled
    by _invoke_api() and should not reach this function.
    """
    if model in MODEL_COMMANDS:
        return MODEL_COMMANDS[model]
    if model.startswith("claude:"):
        submodel = model.split(":", 1)[1]
        return ["claude", "-p", "--model", submodel, "--output-format", "json"]
    if model.startswith("gemini:"):
        submodel = model.split(":", 1)[1]
        return ["gemini", "--skip-trust", "-m", submodel, "-o", "json", "-p", ""]
    if model.startswith("ollama:"):
        ollama_model = model.split(":", 1)[1]
        return ["ollama", "run", ollama_model]
    available = (
        list(MODEL_COMMANDS)
        + ["claude:<model>", "gemini:<model>", "ollama:<model>",
           "api:<model>", "vertex:<model>"]
    )
    raise ValueError(f"Unknown model: {model}. Available: {available}")


def _invoke_api(prompt: str, model: str, timeout: int = 300) -> str:
    """Invoke an LLM via LangChain API adapter. Returns response text."""
    prefix, name = model.split(":", 1)

    if prefix == "api":
        if not _HAS_LANGCHAIN_ANTHROPIC:
            raise ImportError(
                "langchain-anthropic is required for api: models. "
                "Install with: pip install langchain-anthropic"
            )
        llm = ChatAnthropic(model=name, timeout=float(timeout))
    elif prefix == "vertex":
        if not _HAS_LANGCHAIN_VERTEX:
            raise ImportError(
                "langchain-google-vertexai is required for vertex: models. "
                "Install with: pip install langchain-google-vertexai"
            )
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT environment variable is required "
                "for vertex: models"
            )
        location = os.environ.get("GOOGLE_CLOUD_REGION", "us-east5")
        llm = ChatAnthropicVertex(
            model_name=name, project=project, location=location,
            request_timeout=float(timeout),
        )
    else:
        raise ValueError(f"Unknown API prefix: {prefix}")

    config = {}
    handler = _get_langfuse_handler()
    if handler:
        config["callbacks"] = [handler]

    try:
        response = llm.invoke(prompt, config=config)
    except Exception as exc:
        exc_name = type(exc).__name__
        if "timeout" in exc_name.lower() or "timeout" in str(exc).lower():
            raise subprocess.TimeoutExpired(model, timeout) from exc
        raise RuntimeError(f"{model} failed: {exc}") from exc

    return response.content


def invoke_model(prompt: str, model: str = "claude", timeout: int = 300) -> str:
    """Invoke an LLM via CLI subprocess or API. Returns response text.

    For CLI models, uses --output-format json to capture token/cost data.
    Accumulated stats available via get_cost_summary().

    Raises FileNotFoundError if the model binary is not in PATH (CLI models).
    Raises ImportError if API dependencies are not installed (API models).
    Raises RuntimeError if the model exits non-zero or API call fails.
    Raises subprocess.TimeoutExpired on timeout.
    """
    if model.startswith("api:") or model.startswith("vertex:"):
        return _invoke_api(prompt, model, timeout)

    cmd = resolve_model_cmd(model)
    binary = cmd[0]
    if not shutil.which(binary):
        raise FileNotFoundError(f"'{binary}' CLI not found in PATH")

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{model} failed: {result.stderr}")
    output = result.stdout
    # Fragile: ollama thinking markers may change across versions
    if model.startswith("ollama:") and "Thinking...\n" in output:
        parts = output.split("...done thinking.\n", 1)
        if len(parts) == 2:
            output = parts[1]
    if model.startswith("ollama:"):
        return output
    return _parse_cli_json(output, model)


# --- Tool-use chat providers ---
# These wrap different LLM backends to support the agentic tool-use loop.
# Each provider converts between its native message/tool format and a
# common interface so the main loop doesn't care which backend is in use.
#
# The Anthropic provider passes messages straight through to the SDK.
# The Ollama provider converts between Anthropic format (used by the main
# loop) and the OpenAI-compatible format that Ollama speaks.


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class ChatResponse:
    content: list
    stop_reason: str


def create_provider(model_string):
    """Create a chat provider based on the model string.

    Model string formats:
      ollama:<model>  — Local Ollama (e.g. ollama:llama3.1)
      api:<model>     — Direct Anthropic API (e.g. api:claude-sonnet-4-20250514)
      <model>         — Vertex AI (default, e.g. claude-sonnet-4-20250514)
    """
    if model_string.startswith("ollama:"):
        ollama_model = model_string.split(":", 1)[1]
        return _OllamaProvider(ollama_model)
    if model_string.startswith("api:"):
        anthropic_model = model_string.split(":", 1)[1]
        return _AnthropicProvider(anthropic_model, use_vertex=False)
    return _AnthropicProvider(model_string, use_vertex=True)


class _AnthropicProvider:
    """Wraps the Anthropic SDK (Vertex AI or direct API) for tool-use conversations."""

    def __init__(self, model, use_vertex=True):
        self.model = model
        if use_vertex:
            from anthropic import AnthropicVertex
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            region = os.environ.get("GOOGLE_CLOUD_REGION", "us-east5")
            if not project:
                raise RuntimeError("Set GOOGLE_CLOUD_PROJECT for Vertex AI models")
            self.client = AnthropicVertex(project_id=project, region=region)
        else:
            from anthropic import Anthropic
            self.client = Anthropic()

    def send(self, messages, system, tools, max_tokens=8096):
        return self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )


class _OllamaProvider:
    """Wraps the Ollama HTTP API for tool-use conversations.

    Converts between Anthropic message format (used by the main loop)
    and the OpenAI-compatible format that Ollama expects.
    """

    def __init__(self, model, base_url=None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._tool_id_counter = 0

    def send(self, messages, system, tools, max_tokens=8096):
        ollama_messages = self._convert_messages(messages, system)
        ollama_tools = self._convert_tools(tools)

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("error", body)
            except (json.JSONDecodeError, ValueError):
                detail = body
            raise RuntimeError(f"Ollama error: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url} — is it running? ({e})"
            ) from e

        return self._parse_response(result)

    def _convert_messages(self, messages, system):
        """Convert Anthropic-format message history to Ollama/OpenAI format."""
        converted = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Plain user text
            if role == "user" and isinstance(content, str):
                converted.append({"role": "user", "content": content})

            # Tool results (Anthropic wraps these in a "user" message)
            elif role == "user" and isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        converted.append({
                            "role": "tool",
                            "content": item["content"],
                        })

            # Assistant response (may contain text + tool_use blocks)
            elif role == "assistant":
                if isinstance(content, str):
                    converted.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        # Handle both SDK objects (attrs) and dataclasses/dicts
                        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                        if btype == "text":
                            text_parts.append(getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", ""))
                        elif btype == "tool_use":
                            name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
                            inp = getattr(block, "input", {}) if not isinstance(block, dict) else block.get("input", {})
                            tool_calls.append({
                                "function": {"name": name, "arguments": inp},
                            })

                    assistant_msg = {"role": "assistant", "content": "\n".join(text_parts)}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    converted.append(assistant_msg)

        return converted

    def _convert_tools(self, tools):
        """Convert Anthropic tool schemas to OpenAI/Ollama format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def _parse_response(self, result):
        """Convert Ollama JSON response to a ChatResponse."""
        msg = result.get("message", {})
        content = []

        if msg.get("content"):
            content.append(TextBlock(text=msg["content"]))

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            self._tool_id_counter += 1
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            content.append(ToolUseBlock(
                id=f"ollama_tool_{self._tool_id_counter}",
                name=func["name"],
                input=args,
            ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return ChatResponse(content=content, stop_reason=stop_reason)
