from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace
import socket
import subprocess
import urllib.error

import pytest

import worker_bee.llm as llm


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


def test_cli_parse_failure_records_call_without_known_usage():
    llm.reset_cost_tracker()

    output = "not valid json"
    assert llm._parse_cli_json(output, "claude") == output
    assert llm.get_cost_summary() == {
        "calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
        "by_model": {
            "claude": {
                "calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            }
        },
    }
    assert llm.get_last_usage() == {"prompt_tokens": 0, "completion_tokens": 0}


def test_cli_non_object_json_records_call_without_known_usage():
    llm.reset_cost_tracker()

    output = "[1, 2, 3]"
    assert llm._parse_cli_json(output, "gemini") == output
    assert llm.get_cost_summary()["calls"] == 1
    assert llm.get_cost_summary()["by_model"]["gemini"]["calls"] == 1
    assert llm.get_last_usage() == {"prompt_tokens": 0, "completion_tokens": 0}


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
