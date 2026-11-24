import threading
import time
from unittest.mock import patch

from worker_bee.dispatcher import Response, dispatch, dispatch_batch, dispatch_chat


def test_dispatch_returns_usage():
    with patch("worker_bee.dispatcher.invoke_model", return_value="ok"), patch(
        "worker_bee.dispatcher.get_last_usage",
        return_value={"prompt_tokens": 12, "completion_tokens": 7},
    ):
        result = dispatch("hello", model="test")
    assert result.text == "ok"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 7


def test_batch_is_parallel_ordered_and_isolates_retries():
    calls = {}

    def fake_dispatch(prompt, **kwargs):
        calls[prompt] = calls.get(prompt, 0) + 1
        if prompt == "bad":
            raise RuntimeError("nope")
        return Response(prompt, "test", 1, 2)

    with patch("worker_bee.dispatcher.dispatch", side_effect=fake_dispatch):
        result = dispatch_batch(
            [({"id": 1}, "one"), ({"id": 2}, "bad"), ({"id": 3}, "three")],
            model="test",
            max_workers=3,
            retries=2,
        )

    assert [issue["id"] for issue, _ in result] == [1, 2, 3]
    assert result[0][1].text == "one"
    assert result[2][1].text == "three"
    assert result[1][1].error == "RuntimeError: nope"
    assert calls["bad"] == 3


def test_batch_rejects_negative_retries():
    try:
        dispatch_batch([], retries=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_batch_rejects_non_positive_max_workers():
    for max_workers in (0, -1):
        try:
            dispatch_batch([], max_workers=max_workers)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_batch_rejects_non_integer_max_workers():
    for max_workers in (1.5, "2", True):
        try:
            dispatch_batch([], max_workers=max_workers)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_dispatch_uses_configured_process_wide_rate_limiter():
    with patch("worker_bee.dispatcher.invoke_model", return_value="ok"), patch(
        "worker_bee.dispatcher.get_last_usage",
        return_value={"prompt_tokens": 0, "completion_tokens": 0},
    ), patch("worker_bee.dispatcher._acquire_rate_limit") as acquire:
        from worker_bee.dispatcher import configure_rate_limit

        configure_rate_limit("test", 10)
        try:
            dispatch("hello", model="test")
        finally:
            configure_rate_limit("test", None)

    acquire.assert_called_once_with("test")


def test_rate_limit_rejects_non_positive_values():
    from worker_bee.dispatcher import configure_rate_limit

    for value in (0, -1, True, float("inf"), float("nan"), "10"):
        try:
            configure_rate_limit("test", value)
        except ValueError as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_configured_limiter_spaces_actual_dispatch_starts():
    from worker_bee.dispatcher import configure_rate_limit

    starts = []
    configure_rate_limit("timed", 20)
    try:
        with patch("worker_bee.dispatcher.invoke_model", side_effect=lambda *a, **k: starts.append(time.monotonic()) or "ok"), patch(
            "worker_bee.dispatcher.get_last_usage",
            return_value={"prompt_tokens": 0, "completion_tokens": 0},
        ):
            dispatch("one", model="timed")
            dispatch("two", model="timed")
    finally:
        configure_rate_limit("timed", None)
    assert starts[1] - starts[0] >= 0.04


def test_concurrent_dispatch_calls_share_limiter():
    from worker_bee.dispatcher import configure_rate_limit

    starts = []
    configure_rate_limit("concurrent", 50)
    try:
        def invoke(*args, **kwargs):
            starts.append(time.monotonic())
            return "ok"

        with patch("worker_bee.dispatcher.invoke_model", side_effect=invoke), patch(
            "worker_bee.dispatcher.get_last_usage",
            return_value={"prompt_tokens": 0, "completion_tokens": 0},
        ):
            threads = [threading.Thread(target=dispatch, args=(str(i),), kwargs={"model": "concurrent"}) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    finally:
        configure_rate_limit("concurrent", None)
    assert len(starts) == 3
    assert min(b - a for a, b in zip(starts, starts[1:])) >= 0.015


def test_dispatch_chat_uses_configured_limiter():
    from worker_bee.dispatcher import configure_rate_limit

    configure_rate_limit("chat-test", 10)
    try:
        with patch("worker_bee.dispatcher._acquire_rate_limit") as acquire, patch(
            "worker_bee.dispatcher.create_provider"
        ) as create:
            dispatch_chat([], model="chat-test")
    finally:
        configure_rate_limit("chat-test", None)
    acquire.assert_called_once_with("chat-test")
    create.return_value.send.assert_called_once()
