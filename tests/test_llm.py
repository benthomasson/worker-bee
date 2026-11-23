from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import worker_bee.llm as llm


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
