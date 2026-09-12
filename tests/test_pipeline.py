"""Tests for run_pipeline wiring (dispatch_batch default + order)."""

from unittest.mock import patch

from worker_bee.dispatcher import Response, dispatch_batch
from worker_bee.pipeline import run_pipeline

ISSUES = [
    {"type": "gated", "belief_id": "b-one", "source": "docs/one.md"},
    {"type": "gated", "belief_id": "b-two", "source": "docs/two.md"},
    {"type": "gated", "belief_id": "b-three", "source": "docs/three.md"},
]


def _ok_response(belief_id: str) -> Response:
    return Response(f"answer for {belief_id}", "test-model", 10, 5)


def test_run_pipeline_uses_dispatch_batch_and_preserves_order():
    def fake_batch(prompts, **kwargs):
        # dispatch_batch returns (issue, response) pairs in input order
        return [(issue, _ok_response(issue["belief_id"])) for issue, _ in prompts]

    with (
        patch("worker_bee.pipeline.extract", return_value=list(ISSUES)),
        patch("worker_bee.pipeline.build_prompt", side_effect=lambda issue: f"prompt for {issue['belief_id']}"),
        patch("worker_bee.pipeline.dispatch_batch", side_effect=fake_batch) as mock_batch,
    ):
        results = run_pipeline("brain.db", model="test")

    assert mock_batch.call_count == 1
    _, batch_kwargs = mock_batch.call_args
    assert batch_kwargs.get("model") == "test"
    assert [r["issue"]["belief_id"] for r in results] == ["b-one", "b-two", "b-three"]
    assert [r["response"].text for r in results] == [
        "answer for b-one",
        "answer for b-two",
        "answer for b-three",
    ]


def test_run_pipeline_keeps_failed_items_with_error():
    good = _ok_response("b-one")
    bad = Response("", "test-model", 0, 0, error="RuntimeError: nope")

    with (
        patch("worker_bee.pipeline.extract", return_value=list(ISSUES)),
        patch("worker_bee.pipeline.build_prompt", side_effect=lambda issue: f"prompt for {issue['belief_id']}"),
        patch(
            "worker_bee.pipeline.dispatch_batch",
            return_value=[(ISSUES[0], good), (ISSUES[1], bad), (ISSUES[2], _ok_response("b-three"))],
        ),
    ):
        results = run_pipeline("brain.db")

    assert [r["issue"]["belief_id"] for r in results] == ["b-one", "b-two", "b-three"]
    assert results[0]["response"].error is None
    assert results[1]["response"].error == "RuntimeError: nope"


def test_run_pipeline_dry_run_does_not_dispatch():
    with (
        patch("worker_bee.pipeline.extract", return_value=list(ISSUES)),
        patch("worker_bee.pipeline.build_prompt", side_effect=lambda issue: f"prompt for {issue['belief_id']}"),
        patch("worker_bee.pipeline.dispatch_batch") as mock_batch,
    ):
        results = run_pipeline("brain.db", dry_run=True)

    mock_batch.assert_not_called()
    assert [r["issue"]["belief_id"] for r in results] == ["b-one", "b-two", "b-three"]
    assert all(r["response"] is None for r in results)
    assert all(r["prompt"] == f"prompt for {r['issue']['belief_id']}" for r in results)


def test_dispatch_batch_defaults_to_single_worker():
    import inspect

    params = inspect.signature(dispatch_batch).parameters
    assert params["max_workers"].default == 1
