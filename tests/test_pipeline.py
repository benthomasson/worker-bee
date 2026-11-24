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
    assert batch_kwargs.get("max_workers") == 1
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


def test_pipeline_clears_invocation_rate_limit():
    with (
        patch("worker_bee.pipeline.extract", return_value=[]),
        patch("worker_bee.pipeline.configure_rate_limit") as configure,
    ):
        run_pipeline("brain.db", model="test", requests_per_second=2)

    assert [call.args for call in configure.call_args_list] == [
        ("test", 2),
        ("test", None),
    ]


def test_pipeline_clears_invocation_rate_limit_on_failure():
    with (
        patch("worker_bee.pipeline.extract", side_effect=RuntimeError("broken")),
        patch("worker_bee.pipeline.configure_rate_limit") as configure,
    ):
        try:
            run_pipeline("brain.db", model="test", requests_per_second=2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected extraction failure")

    assert configure.call_args_list[-1].args == ("test", None)


def test_pipeline_progress_file_skips_completed_items_on_resume(tmp_path):
    progress = tmp_path / "progress.json"
    calls = []

    def fake_batch(prompts, **kwargs):
        calls.extend(issue["belief_id"] for issue, _ in prompts)
        return [(issue, _ok_response(issue["belief_id"])) for issue, _ in prompts]

    with (
        patch("worker_bee.pipeline.extract", return_value=list(ISSUES)),
        patch("worker_bee.pipeline.build_prompt", side_effect=lambda issue: f"prompt for {issue['belief_id']}"),
        patch("worker_bee.pipeline.dispatch_batch", side_effect=fake_batch),
    ):
        first = run_pipeline("brain.db", model="test", progress_file=progress)
        second = run_pipeline("brain.db", model="test", progress_file=progress, resume=True)

    assert calls == ["b-one", "b-two", "b-three"]
    assert [r["issue"]["belief_id"] for r in first] == ["b-one", "b-two", "b-three"]
    assert [r["issue"]["belief_id"] for r in second] == ["b-one", "b-two", "b-three"]


def test_pipeline_resume_requires_progress_file_before_extract():
    with patch("worker_bee.pipeline.extract") as extract_mock:
        try:
            run_pipeline("brain.db", resume=True)
        except ValueError as exc:
            assert "progress-file" in str(exc)
        else:
            raise AssertionError("expected resume validation error")
    extract_mock.assert_not_called()


def test_pipeline_rejects_existing_progress_file_without_resume(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text("{}", encoding="utf-8")
    with patch("worker_bee.pipeline.extract") as extract_mock:
        try:
            run_pipeline("brain.db", progress_file=progress)
        except ValueError as exc:
            assert "use --resume" in str(exc)
        else:
            raise AssertionError("expected existing progress validation error")
    extract_mock.assert_not_called()


def test_pipeline_resume_preserves_current_order_and_ignores_stale(tmp_path):
    progress = tmp_path / "progress.json"
    from worker_bee.pipeline import _serialize_result, _write_progress
    saved = {}
    for issue in (ISSUES[2], ISSUES[0], {"type": "gated", "belief_id": "stale"}):
        saved[f"{issue['type']}:{issue['belief_id']}"] = _serialize_result({
            "issue": issue, "prompt": "saved", "response": _ok_response(issue["belief_id"])
        })
    _write_progress(progress, saved)
    with patch("worker_bee.pipeline.extract", return_value=[ISSUES[0], ISSUES[2]]):
        results = run_pipeline("brain.db", model="test", progress_file=progress, resume=True)
    assert [r["issue"]["belief_id"] for r in results] == ["b-one", "b-three"]


def test_pipeline_rejects_progress_dry_run(tmp_path):
    with patch("worker_bee.pipeline.extract", return_value=list(ISSUES)):
        try:
            run_pipeline("brain.db", dry_run=True, progress_file=tmp_path / "progress.json")
        except ValueError as exc:
            assert "dry-run" in str(exc)
        else:
            raise AssertionError("expected dry-run validation error")


def test_progress_item_round_trip_and_validation():
    from worker_bee.pipeline import _result_from_saved, _serialize_result
    result = {"issue": ISSUES[0], "prompt": "p", "response": _ok_response("b-one")}
    assert _result_from_saved(_serialize_result(result)) == result
    try:
        _result_from_saved({"issue": ISSUES[0], "prompt": "p", "response": {}})
    except ValueError as exc:
        assert "progress response" in str(exc)
    else:
        raise AssertionError("expected malformed response error")


def test_progress_records_failures_and_retries_them_on_resume(tmp_path):
    progress = tmp_path / "progress.json"
    bad = Response("", "test-model", 0, 0, error="temporary")
    calls = []

    def fake_batch(prompts, **kwargs):
        issue, _ = prompts[0]
        calls.append(issue["belief_id"])
        return [(issue, bad if len(calls) == 1 else _ok_response(issue["belief_id"]))]

    with (
        patch("worker_bee.pipeline.extract", return_value=[ISSUES[0]]),
        patch("worker_bee.pipeline.build_prompt", return_value="prompt"),
        patch("worker_bee.pipeline.dispatch_batch", side_effect=fake_batch),
    ):
        first = run_pipeline("brain.db", progress_file=progress)
        second = run_pipeline("brain.db", progress_file=progress, resume=True)

    assert calls == ["b-one", "b-one"]
    assert first[0]["response"].error == "temporary"
    assert second[0]["response"].error is None


def test_resume_rejects_changed_issue_or_prompt(tmp_path):
    progress = tmp_path / "progress.json"
    from worker_bee.pipeline import _serialize_result, _write_progress
    _write_progress(progress, {
        "gated:b-one": _serialize_result({
            "issue": ISSUES[0], "prompt": "old prompt", "response": _ok_response("b-one")
        })
    })
    with (
        patch("worker_bee.pipeline.extract", return_value=[{**ISSUES[0], "description": "changed"}]),
        patch("worker_bee.pipeline.build_prompt", return_value="new prompt"),
        patch("worker_bee.pipeline.dispatch_batch", return_value=[
            (ISSUES[0], _ok_response("b-one"))
        ]) as dispatch,
    ):
        run_pipeline("brain.db", progress_file=progress, resume=True)
    dispatch.assert_called_once()


def test_progress_rejects_duplicate_issue_keys():
    duplicate = [ISSUES[0], dict(ISSUES[0])]
    with patch("worker_bee.pipeline.extract", return_value=duplicate):
        try:
            run_pipeline("brain.db")
        except ValueError as exc:
            assert "duplicate issue keys" in str(exc)
        else:
            raise AssertionError("expected duplicate key error")


def test_progress_rejects_mismatched_item_key(tmp_path):
    progress = tmp_path / "progress.json"
    from worker_bee.pipeline import _serialize_result, _write_progress
    _write_progress(progress, {
        "gated:wrong": _serialize_result({
            "issue": ISSUES[0], "prompt": "p", "response": _ok_response("b-one")
        })
    })
    try:
        run_pipeline("brain.db", progress_file=progress, resume=True)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected key consistency error")


def test_dispatch_batch_defaults_to_single_worker():
    import inspect

    params = inspect.signature(dispatch_batch).parameters
    assert params["max_workers"].default == 1
