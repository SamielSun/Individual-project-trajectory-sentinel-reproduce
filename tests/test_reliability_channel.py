"""Contracts for the tool-reliability channel r_t (telemetry v6)."""
from __future__ import annotations

import numpy as np

from derail.common import (D_TOTAL_GOAL, D_TOTAL_REL, IDX_REL_BURST,
                           IDX_REL_CONSEC_FAIL, IDX_REL_DISTINCT_FAILING,
                           IDX_REL_FAIL_FRAC, IDX_REL_NO_RECOVERY,
                           IDX_REL_REPEAT_FAIL, IDX_REL_SINCE_CLEAN,
                           CHANNEL_SLICES)
from derail.telemetry.adapter import episode_from_trace
from derail.telemetry.reliability import (ReliabilityState,
                                          reliability_features)

_B = D_TOTAL_GOAL


def _step(calls, text="working on it"):
    return {"text": text, "action": "tool_call", "output_tokens": 20,
            "latency_s": 0.4, "logprobs_available": False, "schema": 5,
            "task": "find papers about echo state networks",
            "tool_events": [
                {"name": n, "args": {"q": a}, "result": r,
                 "is_error": e, "result_chars": len(r)}
                for n, a, r, e in calls]}


def _run(steps):
    state = ReliabilityState()
    return np.array([reliability_features(s, state) for s in steps])


def test_a_trace_with_no_tool_events_is_inert() -> None:
    out = _run([{"text": "just reasoning"} for _ in range(4)])
    assert out.sum() == 0.0


def test_clean_calls_are_not_anomalous() -> None:
    out = _run([_step([("search", f"q{i}", "a document", False)])
                for i in range(5)])
    assert out.sum() == 0.0


def test_scope_grows_as_more_tools_fail() -> None:
    """distinct_failing is the dim that beat error_flag on tool cascade."""
    out = _run([
        _step([("a", "q", "Error: 503", True)]),
        _step([("b", "q", "Error: 503", True)]),
        _step([("c", "q", "Error: 503", True)]),
    ])
    col = out[:, IDX_REL_DISTINCT_FAILING - _B]
    assert col[0] == 1.0 and col[1] == 1.0 and col[2] == 1.0
    # Every tool tried so far is failing, which a per-step error flag cannot
    # distinguish from a single tool failing once.
    assert out[2, IDX_REL_CONSEC_FAIL - _B] > out[0, IDX_REL_CONSEC_FAIL - _B]


def test_a_retry_of_the_same_call_is_marked_repeat() -> None:
    out = _run([
        _step([("s", "same", "Error: 503", True)]),
        _step([("s", "same", "Error: 503", True)]),
    ])
    assert out[0, IDX_REL_REPEAT_FAIL - _B] == 0.0
    assert out[1, IDX_REL_REPEAT_FAIL - _B] == 1.0


def test_recovery_clears_the_failing_state() -> None:
    out = _run([
        _step([("s", "q1", "Error: 503", True)]),
        _step([("s", "q2", "a document", False)]),
    ])
    assert out[0, IDX_REL_NO_RECOVERY - _B] == 1.0
    assert out[1, IDX_REL_NO_RECOVERY - _B] == 0.0
    assert out[1, IDX_REL_DISTINCT_FAILING - _B] == 0.0
    assert out[1, IDX_REL_SINCE_CLEAN - _B] == 0.0


def test_time_since_clean_starts_only_after_the_first_tool_call() -> None:
    """A reasoning prefix must not read as degradation."""
    out = _run([{"text": "thinking"}, {"text": "still thinking"},
                _step([("s", "q", "a document", False)])])
    assert out[0, IDX_REL_SINCE_CLEAN - _B] == 0.0
    assert out[1, IDX_REL_SINCE_CLEAN - _B] == 0.0
    assert out[2, IDX_REL_SINCE_CLEAN - _B] == 0.0


def test_degradation_persists_across_steps_without_calls() -> None:
    out = _run([
        _step([("s", "q", "Error: 503", True)]),
        {"text": "let me think about that"},
    ])
    assert out[1, IDX_REL_SINCE_CLEAN - _B] > out[0, IDX_REL_SINCE_CLEAN - _B]
    assert out[1, IDX_REL_NO_RECOVERY - _B] == 1.0


def test_features_at_t_cannot_depend_on_a_later_step() -> None:
    head = [_step([("s", f"q{i}", "a document", False)]) for i in range(3)]
    a = _run(head + [_step([("s", "x", "a document", False)])])
    b = _run(head + [_step([("s", "x", "Error: 503", True)])])
    assert np.array_equal(a[:3], b[:3])


def test_every_dim_is_bounded_and_oriented() -> None:
    out = _run([_step([("a", "q", "Error: 503", True),
                       ("b", "q", "Error: 503", True)]) for _ in range(6)])
    assert out.min() >= 0.0 and out.max() <= 1.0
    for idx in (IDX_REL_FAIL_FRAC, IDX_REL_DISTINCT_FAILING,
                IDX_REL_CONSEC_FAIL, IDX_REL_NO_RECOVERY, IDX_REL_BURST):
        assert out[-1, idx - _B] > 0.0, "a sustained cascade must score high"


def test_appending_the_channel_leaves_the_earlier_dims_untouched() -> None:
    """[0, 64) must stay byte-identical, or every frozen artifact moves."""
    steps = [_step([("s", f"q{i}", "a document" if i % 2 else "Error: 503",
                     i % 2 == 0)]) for i in range(6)]
    without = episode_from_trace(steps, "e", use_sentence_transformers=False,
                                 goal=True)
    with_ = episode_from_trace(steps, "e", use_sentence_transformers=False,
                               reliability=True)
    assert without.X.shape[1] == D_TOTAL_GOAL
    assert with_.X.shape[1] == D_TOTAL_REL
    assert np.array_equal(without.X, with_.X[:, :D_TOTAL_GOAL])


def test_the_channel_slice_is_registered_and_disjoint() -> None:
    r = CHANNEL_SLICES["r"]
    assert (r.start, r.stop) == (D_TOTAL_GOAL, D_TOTAL_REL)
    for name, sl in CHANNEL_SLICES.items():
        if name != "r":
            assert sl.stop <= r.start or sl.start >= r.stop


def test_the_channel_reads_no_task_or_framework_identity() -> None:
    """Same calls, different task text and framework: identical features."""
    a = _run([_step([("s", "q", "Error: 503", True)], text="langgraph run")])
    b = _run([_step([("s", "q", "Error: 503", True)], text="autogen run")])
    assert np.array_equal(a, b)
