from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace
import socket
import subprocess
import urllib.error

import pytest

import worker_bee.llm as llm


def test_cli_invocation_uses_configured_workspace_as_cwd(monkeypatch, tmp_path):
    seen = {}

    class Result:
        returncode = 0
        stdout = '{"result": "ok"}'
        stderr = ""

    monkeypatch.setattr(llm, "resolve_model_cmd", lambda model: ["claude"])
    monkeypatch.setattr(llm.shutil, "which", lambda binary: "/usr/bin/claude")
    monkeypatch.setattr(llm, "get_workspace_root", lambda: tmp_path)

    def fake_run(*args, **kwargs):
        seen["cwd"] = kwargs["cwd"]
        return Result()

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    monkeypatch.setattr(llm, "_parse_cli_json", lambda output, model: "ok")
    assert llm.invoke_model("hello", "claude") == "ok"
    assert seen["cwd"] == tmp_path


def test_openai_simple_invocation_normalizes_timeout(monkeypatch):
    class SDKTimeout(Exception):
        pass

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["timeout"] == 17
            raise SDKTimeout("request timeout")

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(llm, "_HAS_OPENAI", True)
    monkeypatch.setattr(llm._openai_mod, "OpenAI", lambda: FakeClient())

    with pytest.raises(subprocess.TimeoutExpired) as error:
        llm._invoke_openai("hello", "openai:test", timeout=17)
    assert error.value.timeout == 17


def test_openai_tool_invocation_normalizes_timeout(monkeypatch):
    class SDKTimeout(Exception):
        pass

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["timeout"] == 23
            raise SDKTimeout("request timed out")

    provider = object.__new__(llm._OpenAIProvider)
    provider.model = "test"
    provider.client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(subprocess.TimeoutExpired) as error:
        provider.send([], "", [], timeout=23)
    assert error.value.timeout == 23


def test_openai_tool_invocation_normalizes_errors_and_passes_timeout(monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["timeout"] == 23
            raise ValueError("bad request")

    provider = object.__new__(llm._OpenAIProvider)
    provider.model = "test"
    provider.client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(RuntimeError, match="openai:test failed: bad request"):
        provider.send([], "", [], timeout=23)


def test_openai_malformed_response_records_failed_call(monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(output=None)

    monkeypatch.setattr(llm, "_HAS_OPENAI", True)
    monkeypatch.setattr(llm._openai_mod, "OpenAI", lambda: SimpleNamespace(responses=FakeResponses()))
    llm.reset_cost_tracker()

    with pytest.raises(TypeError):
        llm._invoke_openai("hello", "openai:test")
    assert llm.get_cost_summary()["usage_unknown_calls"] == 1


def test_api_value_error_is_normalized(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt, config=None):
            raise ValueError("bad provider response")

    monkeypatch.setattr(llm, "_HAS_LANGCHAIN_ANTHROPIC", True)
    monkeypatch.setattr(llm, "ChatAnthropic", lambda **kwargs: FakeLLM())
    llm.reset_cost_tracker()

    with pytest.raises(RuntimeError, match="api:test failed: bad provider response"):
        llm._invoke_api("hello", "api:test")
    assert llm.get_cost_summary()["calls"] == 1


def test_api_constructor_failure_records_failed_call(monkeypatch):
    def fail_constructor(**kwargs):
        raise RuntimeError("bad configuration")

    monkeypatch.setattr(llm, "_HAS_LANGCHAIN_ANTHROPIC", True)
    monkeypatch.setattr(llm, "ChatAnthropic", fail_constructor)
    llm.reset_cost_tracker()

    with pytest.raises(RuntimeError, match="bad configuration"):
        llm._invoke_api("hello", "api:test")
    assert llm.get_cost_summary()["usage_unknown_calls"] == 1


def test_anthropic_provider_records_unknown_usage():
    llm.reset_cost_tracker()

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(usage=None)

    provider = object.__new__(llm._AnthropicProvider)
    provider.model = "claude-test"
    provider.client = SimpleNamespace(messages=FakeMessages())

    provider.send([], "system", [], timeout=19)
    assert llm.get_cost_summary()["usage_unknown_calls"] == 1


def test_anthropic_provider_passes_timeout():
    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["timeout"] == 19
            return "response"

    provider = object.__new__(llm._AnthropicProvider)
    provider.model = "claude-test"
    provider.client = SimpleNamespace(messages=FakeMessages())

    assert provider.send([], "system", [], timeout=19) == "response"


def test_anthropic_provider_normalizes_timeout():
    class SDKTimeout(Exception):
        pass

    class FakeMessages:
        def create(self, **kwargs):
            raise SDKTimeout("timed out")

    provider = object.__new__(llm._AnthropicProvider)
    provider.model = "claude-test"
    provider.client = SimpleNamespace(messages=FakeMessages())

    with pytest.raises(subprocess.TimeoutExpired) as error:
        provider.send([], "system", [], timeout=19)
    assert error.value.timeout == 19


def test_ollama_provider_passes_timeout(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"message": {"content": "ok"}}'

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        return FakeResponse()

    provider = llm._OllamaProvider("test", base_url="http://ollama")
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    provider.send([], "system", [], timeout=29)
    assert calls == [29]


def test_ollama_provider_normalizes_wrapped_socket_timeout(monkeypatch):
    provider = llm._OllamaProvider("test", base_url="http://ollama")

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(subprocess.TimeoutExpired) as error:
        provider.send([], "system", [], timeout=29)
    assert error.value.timeout == 29


def test_cli_nonzero_exit_records_failed_call(monkeypatch):
    llm.reset_cost_tracker()
    monkeypatch.setattr(llm.shutil, "which", lambda binary: "/usr/bin/" + binary)
    monkeypatch.setattr(
        llm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="provider failed"),
    )

    with pytest.raises(RuntimeError, match="claude failed"):
        llm.invoke_model("hello", "claude")
    summary = llm.get_cost_summary()
    assert summary["calls"] == 1
    assert summary["usage_unknown_calls"] == 1
    assert summary["by_model"]["claude"]["usage_unknown_calls"] == 1


def test_cli_parse_failure_records_call_without_known_usage():
    llm.reset_cost_tracker()

    output = "not valid json"
    assert llm._parse_cli_json(output, "claude") == output
    assert llm.get_cost_summary() == {
        "calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "usage_unknown_calls": 1,
        "total_cost_usd": 0.0,
        "by_model": {
            "claude": {
                "calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "usage_unknown_calls": 1,
                "total_cost_usd": 0.0,
            }
        },
    }
    assert llm.get_last_usage() == {"prompt_tokens": 0, "completion_tokens": 0}


def test_cli_non_object_json_records_call_without_known_usage():
    llm.reset_cost_tracker()

    output = "[1, 2, 3]"
    assert llm._parse_cli_json(output, "gemini") == output
    summary = llm.get_cost_summary()
    assert summary["calls"] == 1
    assert summary["usage_unknown_calls"] == 1
    assert summary["by_model"]["gemini"]["calls"] == 1
    assert summary["by_model"]["gemini"]["usage_unknown_calls"] == 1
    assert llm.get_last_usage() == {"prompt_tokens": 0, "completion_tokens": 0}


def test_cli_null_and_partial_usage_are_unknown():
    llm.reset_cost_tracker()

    llm._parse_cli_json('{"result":"ok","usage":null}', "claude")
    llm._parse_cli_json('{"response":"ok","stats":{"models":{"main":{"tokens":{"input":3}}}}}', "gemini")

    summary = llm.get_cost_summary()
    assert summary["calls"] == 2
    assert summary["usage_unknown_calls"] == 2


def test_invalid_token_values_are_normalized_and_marked_unknown():
    llm.reset_cost_tracker()

    llm._record_cost("bad", None, "10", 0.0)

    summary = llm.get_cost_summary()
    assert summary["calls"] == 1
    assert summary["usage_unknown_calls"] == 1
    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0


def test_cli_invalid_cache_usage_is_unknown():
    llm.reset_cost_tracker()

    llm._parse_cli_json(
        '{"result":"ok","usage":{"input_tokens":2,"output_tokens":3,'
        '"cache_read_input_tokens":"bad"}}',
        "claude",
    )

    summary = llm.get_cost_summary()
    assert summary["calls"] == 1
    assert summary["usage_unknown_calls"] == 1
    assert summary["input_tokens"] == 2
    assert summary["output_tokens"] == 3


def test_api_invalid_usage_is_unknown(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt, config):
            return SimpleNamespace(content="ok", usage_metadata={"input_tokens": None, "output_tokens": 3})

    monkeypatch.setattr(llm, "_HAS_LANGCHAIN_ANTHROPIC", True)
    monkeypatch.setattr(llm, "ChatAnthropic", lambda **kwargs: FakeLLM())
    llm.reset_cost_tracker()

    assert llm._invoke_api("hello", "api:test") == "ok"
    assert llm.get_cost_summary()["usage_unknown_calls"] == 1


def test_ollama_invalid_usage_is_unknown(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"response":"ok","prompt_eval_count":null,"eval_count":"3"}'

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    llm.reset_cost_tracker()

    assert llm._invoke_ollama("hello", "ollama:test") == "ok"
    assert llm.get_cost_summary()["usage_unknown_calls"] == 1


def test_openai_tool_invalid_usage_is_unknown():
    provider = object.__new__(llm._OpenAIProvider)
    provider.model = "test"
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        output=[], usage=SimpleNamespace(input_tokens=None, output_tokens=3))))
    llm.reset_cost_tracker()

    provider.send([], "", [])
    assert llm.get_cost_summary()["usage_unknown_calls"] == 1


def test_cost_summary_formats_unknown_usage_and_reset_clears_it():
    llm.reset_cost_tracker()
    llm._record_cost("test", 0, 0, 0.0, usage_known=False)

    assert "1 call(s) with unknown usage" in llm.format_cost_summary()
    llm.reset_cost_tracker()
    assert llm.get_cost_summary() == {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "usage_unknown_calls": 0,
        "total_cost_usd": 0.0,
        "by_model": {},
    }


def test_langfuse_handler_initialization_is_thread_safe(monkeypatch):
    instances = []
    instances_lock = Lock()
    started = Barrier(8)

    class FakeHandler:
        pass

    def make_handler():
        handler = FakeHandler()
        with instances_lock:
            instances.append(handler)
        return handler

    monkeypatch.setattr(llm, "_HAS_LANGFUSE", True)
    monkeypatch.setattr(llm, "LangfuseCallbackHandler", make_handler)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setattr(llm, "_langfuse_handler", None)
    monkeypatch.setattr(llm, "_langfuse_checked", False)

    def get_handler(_):
        started.wait()
        return llm._get_langfuse_handler()

    with ThreadPoolExecutor(max_workers=8) as executor:
        handlers = list(executor.map(get_handler, range(8)))

    assert len(instances) == 1
    assert all(handler is instances[0] for handler in handlers)


def test_langfuse_handler_is_cached_when_disabled(monkeypatch):
    monkeypatch.setattr(llm, "_HAS_LANGFUSE", False)
    monkeypatch.setattr(llm, "_langfuse_handler", None)
    monkeypatch.setattr(llm, "_langfuse_checked", False)

    assert llm._get_langfuse_handler() is None
    assert llm._langfuse_checked is True
