"""Do the repair fixes do what they claim, mechanically?

Not "does recovery go up on the corpus" -- the live study measures that.
These are the properties the design rests on, each tied to a failure the
forensics measured:

  a checkpoint may not resume after evidence the content channel distrusts,
  because `is_error=False` is not the same as "usable";
  evidence the rollback discards is re-issued, because a task-required lookup
  that happened after the fault used to be thrown away with it;
  a call that changes the world is never repeated, and a rollback that cannot
  undo one refuses to go behind it;
  the repair path cannot see an onset, a class, a verdict or a verifier;
  the budget binds.

Nothing here names Wikipedia, arXiv, or any task in the suite: the fixtures
are synthetic tools, so a fix that worked by recognising the corpus would
fail these.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from derail.harness.tools import SimpleTool, ToolRegistry
from derail.intervene.checkpoint import (GroundingReference,
                                         discarded_evidence_calls,
                                         last_grounded_step_before,
                                         step_is_grounded)
from derail.intervene.effects import (EFFECTFUL, IDEMPOTENT, READ_ONLY,
                                      Compensation, EffectPolicy,
                                      UncompensatedEffect,
                                      assert_no_duplicate_effects,
                                      safe_checkpoint)
from derail.intervene.real_tool_rollback import (CheckpointResult,
                                                 retry_real_task_from_checkpoint)
from derail.intervene.repair_policy import (HEADLINE_RUNGS, RUNGS, RepairBudget,
                                            ResourceLedger, repair_message,
                                            select_rung)

# --------------------------------------------------------------- fixtures

#: One grounding dim, so a test can put a step inside or outside the null by
#: setting a single number. The real reference uses all nine.
_REF = GroundingReference(dims=("char_anom",), median=(0.0,), scale=(1.0,),
                          trip=3.0, n_healthy_episodes=10, n_healthy_steps=100)


def _row(char_anom: float) -> np.ndarray:
    from derail.common import D_TOTAL_GOAL, IDX_GRD_CHAR_ANOM
    x = np.zeros(D_TOTAL_GOAL)
    x[IDX_GRD_CHAR_ANOM] = char_anom
    return x


def _step(tool: str, args: dict, result: str = "ok",
          is_error: bool = False) -> dict:
    return {"text": f"[{tool}(...) -> {result}]", "action": "tool_call",
            "latency_s": 0.1, "error": is_error,
            "tool_events": [{"name": tool, "args": args, "result": result,
                             "is_error": is_error, "latency_s": 0.1}]}


class _FakeBackend:
    """reset/step/add_tool_results double. Answers once, then stops."""

    def __init__(self, script=None, **_ignored):
        self.history: list = []
        self.script = list(script or [])
        self.calls = 0

    def reset(self, task: str) -> None:
        self.history = [{"role": "user", "content": task}]

    def add_tool_results(self, results):
        for r in results:
            self.history.append({"role": "tool", "tool_name": r["name"],
                                 "content": r["content"]})

    def step(self, t: int) -> dict:
        self.calls += 1
        if self.script:
            return self.script.pop(0)
        return {"text": "done", "action": "synthesis", "tool_uses": [],
                "stop_reason": "end_turn"}


class _Ledger:
    """A tool with a real side effect, so duplication is observable."""

    def __init__(self):
        self.entries: list[str] = []

    def book(self, ref: str) -> str:
        self.entries.append(ref)
        return f"booked {ref} (total {len(self.entries)})"

    def cancel(self, ref: str) -> str:
        self.entries.remove(ref)
        return f"cancelled {ref}"


def _registry(ledger: _Ledger | None = None) -> ToolRegistry:
    tools = [SimpleTool("lookup", "read something", {"q": "query"},
                        lambda q: f"result for {q}")]
    if ledger is not None:
        tools.append(SimpleTool("book", "change the world", {"ref": "ref"},
                                ledger.book))
        tools.append(SimpleTool("cancel", "undo a booking", {"ref": "ref"},
                                ledger.cancel))
    return ToolRegistry(tools)


# ------------------------------------------------- 8. causal checkpointing

def test_a_clean_looking_but_ungrounded_step_is_not_a_checkpoint():
    """The measured defect: a wrong document and a garbled result both arrive
    with is_error=False, and the shipped rule resumed after them."""
    steps = [_step("lookup", {"q": "a"}), _step("lookup", {"q": "b"}),
             _step("lookup", {"q": "c"})]
    X = np.array([_row(0.5), _row(99.0), _row(0.5)])
    assert step_is_grounded(steps[0], X[0], _REF)
    assert not step_is_grounded(steps[1], X[1], _REF)
    assert last_grounded_step_before(steps, X, 3, _REF) == 3
    # ...and with the ungrounded step last, the rule stops before it.
    assert last_grounded_step_before(steps, X, 2, _REF) == 1


def test_an_errored_call_carries_no_evidence():
    step = _step("lookup", {"q": "a"}, result="Error: boom", is_error=True)
    assert not step_is_grounded(step, _row(0.0), _REF)


def test_a_step_with_no_tool_call_is_not_a_checkpoint():
    """A synthesis step observed nothing, so there is nothing to resume from
    after it."""
    bare = {"text": "thinking", "action": "synthesis", "error": False,
            "tool_events": []}
    assert not step_is_grounded(bare, _row(0.0), _REF)


def test_the_checkpoint_rule_cannot_reach_ground_truth():
    for fn in (last_grounded_step_before, step_is_grounded,
               discarded_evidence_calls):
        names = set(inspect.signature(fn).parameters)
        for forbidden in ("tau", "failure_class", "verdict", "success",
                          "success_fn", "verifier", "label"):
            assert forbidden not in names, f"{fn.__name__} can read {forbidden}"


# ------------------------------------------------- 11. discarded evidence

def test_evidence_the_rollback_discards_is_offered_for_regeneration():
    """The failure this fixes: a task-required lookup happened after the
    fault, rollback threw it away, and the retry never remade it."""
    steps = [_step("lookup", {"q": "a"}), _step("lookup", {"q": "poisoned"}),
             _step("lookup", {"q": "needed-by-the-task"})]
    calls = discarded_evidence_calls(steps, 1)
    assert ("lookup", {"q": "needed-by-the-task"}) in calls
    assert ("lookup", {"q": "a"}) not in calls, "kept steps are not discarded"


def test_regeneration_collapses_repeats_and_skips_failures():
    steps = [_step("lookup", {"q": "x"}), _step("lookup", {"q": "x"}),
             _step("lookup", {"q": "y"}, result="Error: no", is_error=True)]
    assert discarded_evidence_calls(steps, 0) == [("lookup", {"q": "x"})]


def test_the_retry_reissues_discarded_evidence_live():
    """End to end through the runner: the discarded call really is executed
    again and its fresh result reaches the conversation."""
    registry = _registry()
    steps = [_step("lookup", {"q": "kept"}), _step("lookup", {"q": "dropped"})]
    checkpoint = CheckpointResult(k=1, outcome="normal", alarm_step=1,
                                  regenerate=(("lookup", {"q": "dropped"}),))
    out = retry_real_task_from_checkpoint(
        steps, "task text", checkpoint, "primary", registry=registry,
        backend_factory=_FakeBackend)
    regenerated = [s for s in out.steps if s.get("regenerated_evidence")]
    assert len(regenerated) == 1
    events = regenerated[0]["tool_events"]
    assert [e["name"] for e in events] == ["lookup"]
    assert "result for dropped" in events[0]["result"]
    assert out.ledger.tool_calls == 1


def test_regeneration_gets_a_fresh_result_rather_than_replaying_the_old_one():
    """Replaying would carry the corruption back in. Re-issuing does not."""
    registry = _registry()
    steps = [_step("lookup", {"q": "topic"}, result="CORRUPTED PAYLOAD")]
    checkpoint = CheckpointResult(k=0, outcome="checkpoint_at_start",
                                  alarm_step=0,
                                  regenerate=(("lookup", {"q": "topic"}),))
    out = retry_real_task_from_checkpoint(
        steps, "task text", checkpoint, "primary", registry=registry,
        backend_factory=_FakeBackend)
    text = "".join(e["result"] for s in out.steps
                   for e in (s.get("tool_events") or []))
    assert "CORRUPTED PAYLOAD" not in text
    assert "result for topic" in text


# ------------------------------------------- 11. state-changing tool safety

def test_an_effectful_call_is_never_reissued_by_a_retry():
    ledger = _Ledger()
    ledger.book("A")
    registry = _registry(ledger)
    policy = EffectPolicy(effects={"lookup": READ_ONLY, "book": EFFECTFUL})
    steps = [_step("book", {"ref": "A"})]
    checkpoint = CheckpointResult(k=0, outcome="checkpoint_at_start",
                                  alarm_step=0)
    backend = _FakeBackend(script=[{
        "text": "", "action": "tool_call", "stop_reason": "tool_use",
        "tool_uses": [{"id": "1", "name": "book", "input": {"ref": "A"}}]}])
    out = retry_real_task_from_checkpoint(
        steps, "task", checkpoint, "primary", registry=registry,
        backend_factory=lambda **kw: backend, effects=policy)
    assert ledger.entries == ["A"], "the retry booked the same thing twice"
    assert out.ledger.tool_calls == 0


def test_regeneration_refuses_effectful_calls_but_takes_idempotent_ones():
    policy = EffectPolicy(effects={"lookup": READ_ONLY, "book": EFFECTFUL,
                                   "sync": IDEMPOTENT})
    assert policy.is_replay_safe("lookup")
    assert policy.is_replay_safe("sync")
    assert not policy.is_replay_safe("book")


def test_an_unclassified_tool_is_treated_as_effectful():
    """The one most likely to do something irreversible is the one nobody
    got round to classifying."""
    assert EffectPolicy(effects={}).effect_of("charge_card") == EFFECTFUL
    assert not EffectPolicy(effects={}).is_replay_safe("charge_card")


def test_rollback_refuses_to_go_behind_an_effect_it_cannot_undo():
    policy = EffectPolicy(effects={"lookup": READ_ONLY, "book": EFFECTFUL})
    steps = [_step("lookup", {"q": "a"}), _step("book", {"ref": "A"}),
             _step("lookup", {"q": "b"})]
    checkpoint, blocking = safe_checkpoint(steps, 0, policy)
    assert checkpoint == 2, "rolled back past an uncompensated booking"
    assert blocking == [(1, "book")]


def test_a_declared_compensation_lets_the_rollback_proceed():
    policy = EffectPolicy(
        effects={"lookup": READ_ONLY, "book": EFFECTFUL},
        compensations={"book": Compensation(
            tool="cancel", build=lambda args: {"ref": args["ref"]})})
    steps = [_step("lookup", {"q": "a"}), _step("book", {"ref": "A"}),
             _step("book", {"ref": "B"})]
    checkpoint, blocking = safe_checkpoint(steps, 0, policy)
    assert checkpoint == 0 and blocking == []
    # Newest first: effects compose, so they are undone in reverse.
    assert policy.compensating_calls(steps, 0, 3) == [
        ("cancel", {"ref": "B"}), ("cancel", {"ref": "A"})]


def test_compensation_actually_restores_the_world():
    ledger = _Ledger()
    registry = _registry(ledger)
    ledger.book("A")
    ledger.book("B")
    assert ledger.entries == ["A", "B"]
    policy = EffectPolicy(
        effects={"book": EFFECTFUL},
        compensations={"book": Compensation(
            tool="cancel", build=lambda args: {"ref": args["ref"]})})
    steps = [_step("book", {"ref": "A"}), _step("book", {"ref": "B"})]
    for tool, args in policy.compensating_calls(steps, 0, 2):
        registry.call(tool, args, cassette=None)
    assert ledger.entries == []


def test_an_errored_effectful_call_does_not_block_a_rollback():
    """It changed nothing, so there is nothing to undo."""
    policy = EffectPolicy(effects={"book": EFFECTFUL})
    steps = [_step("book", {"ref": "A"}, result="Error: declined",
                   is_error=True)]
    assert safe_checkpoint(steps, 0, policy) == (0, [])


def test_duplicate_side_effects_are_detected():
    policy = EffectPolicy(effects={"book": EFFECTFUL, "lookup": READ_ONLY})
    before = [_step("book", {"ref": "A"})]
    assert_no_duplicate_effects(before, [_step("lookup", {"q": "x"})], policy)
    with pytest.raises(UncompensatedEffect, match="effectful"):
        assert_no_duplicate_effects(before, [_step("book", {"ref": "A"})],
                                    policy)


def test_repeating_an_idempotent_call_is_allowed():
    policy = EffectPolicy(effects={"sync": IDEMPOTENT})
    assert_no_duplicate_effects([_step("sync", {"ref": "A"})],
                                [_step("sync", {"ref": "A"})], policy)


# ------------------------------------------------- 9. the ladder, no oracle

def test_no_rung_can_see_ground_truth():
    names = set(inspect.signature(repair_message).parameters)
    for forbidden in ("tau", "failure_class", "verdict", "success",
                      "success_fn", "verifier", "findings", "label"):
        assert forbidden not in names
    assert set(inspect.signature(select_rung).parameters).isdisjoint(
        {"tau", "failure_class", "verdict", "success_fn"})


def test_the_resample_control_says_nothing():
    assert repair_message("resample") is None


def test_a_stuck_run_is_told_to_stop_calling_the_tool():
    steps = [_step("lookup", {"q": "x"}), _step("lookup", {"q": "x"}),
             _step("lookup", {"q": "x"})]
    X = np.array([_row(0.0)] * 3)
    assert select_rung(steps, X, 2, _REF) == "unstick"


def test_ungrounded_evidence_selects_the_regrounding_rung_and_names_the_fault():
    steps = [_step("lookup", {"q": "a"}), _step("lookup", {"q": "b"})]
    X = np.array([_row(0.0), _row(99.0)])
    assert select_rung(steps, X, 1, _REF) == "reground"
    message = repair_message("reground", steps=steps, X=X, alarm_step=1,
                             reference=_REF)
    assert "malformed or garbled" in message
    # Locates the property, never supplies content.
    assert "result for" not in message


def test_a_quiet_alarm_falls_back_to_the_task_independent_reminder():
    steps = [_step("lookup", {"q": "a"}), _step("lookup", {"q": "b"})]
    X = np.array([_row(0.0), _row(0.0)])
    assert select_rung(steps, X, 1, _REF) == "generic"


def test_the_coverage_rung_is_off_unless_asked_for_and_excluded_from_headlines():
    """It overlaps what the verifier checks, so it may not be pooled."""
    steps = [_step("lookup", {"q": "a"})]
    X = np.array([_row(0.0)])
    assert select_rung(steps, X, 0, _REF, required_tools=("other",)) != "coverage"
    assert select_rung(steps, X, 0, _REF, required_tools=("other",),
                       allow_coverage=True) == "coverage"
    assert "coverage" in RUNGS and "coverage" not in HEADLINE_RUNGS


def test_the_coverage_rung_names_only_tools_the_task_declares():
    steps = [_step("lookup", {"q": "a"})]
    message = repair_message("coverage", steps=steps, alarm_step=0,
                             required_tools=("lookup", "other"))
    assert "other" in message and "lookup" not in message


# ------------------------------------------------------- 10. bounded repair

def test_the_step_budget_binds():
    registry = _registry()
    never_stops = [{"text": "", "action": "tool_call", "stop_reason": "tool_use",
                    "tool_uses": []}] * 50
    out = retry_real_task_from_checkpoint(
        [_step("lookup", {"q": "a"})], "task",
        CheckpointResult(k=1, outcome="normal", alarm_step=1), "primary",
        registry=registry,
        backend_factory=lambda **kw: _FakeBackend(script=list(never_stops)),
        budget=RepairBudget(retry_steps=3))
    assert out.outcome == "halted"
    assert out.ledger.model_calls == 3


def test_the_model_call_budget_binds_before_the_step_budget():
    registry = _registry()
    never_stops = [{"text": "", "action": "tool_call", "stop_reason": "tool_use",
                    "tool_uses": []}] * 50
    out = retry_real_task_from_checkpoint(
        [_step("lookup", {"q": "a"})], "task",
        CheckpointResult(k=1, outcome="normal", alarm_step=1), "primary",
        registry=registry,
        backend_factory=lambda **kw: _FakeBackend(script=list(never_stops)),
        budget=RepairBudget(retry_steps=40, max_extra_model_calls=5))
    assert out.outcome == "budget_exhausted"
    assert out.ledger.exhausted == "max_extra_model_calls"
    assert out.ledger.model_calls == 6


def test_regeneration_stops_at_the_tool_budget():
    registry = _registry()
    calls = tuple(("lookup", {"q": f"q{i}"}) for i in range(10))
    out = retry_real_task_from_checkpoint(
        [_step("lookup", {"q": "a"})], "task",
        CheckpointResult(k=0, outcome="checkpoint_at_start", alarm_step=0,
                         regenerate=calls), "primary", registry=registry,
        backend_factory=_FakeBackend,
        budget=RepairBudget(max_extra_tool_calls=3))
    assert out.ledger.tool_calls == 3
    assert out.ledger.exhausted == "max_extra_tool_calls"


def test_every_repair_reports_what_it_spent():
    ledger = ResourceLedger(attempts=1, model_calls=4, tool_calls=6,
                            seconds=2.5, steps_run=4)
    record = ledger.as_record()
    for key in ("repair_attempts", "repair_model_calls", "repair_tool_calls",
                "repair_seconds", "repair_steps_run"):
        assert key in record
    assert ledger.over(RepairBudget()) == ""
    assert ResourceLedger(attempts=99).over(RepairBudget()) == "max_attempts"
