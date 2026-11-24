from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import worker_bee.llm as llm


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
