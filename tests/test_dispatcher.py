from unittest.mock import patch

from worker_bee.dispatcher import Response, dispatch, dispatch_batch


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
