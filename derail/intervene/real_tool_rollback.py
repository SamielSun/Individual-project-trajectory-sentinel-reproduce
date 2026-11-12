"""Domain-neutral rollback/retry continuation for real-tool episodes.

Extends rollback.py's precedent to REAL_TASKS instead of the booking world.
`rebuild_history_from` (rollback.py) is reused unmodified -- reading it shows
it was already domain-neutral (task text + parse_step_events only, no booking
reference); `retry_from_checkpoint` was the booking-bound half, and this
module is its real-tool counterpart.

Two checkpoint rules, kept structurally separate so the oracle arm can never
be mistaken for the deployable primary result:

  compute_checkpoint_primary -- online-observable only: the frozen monitor's
                                 own causal alarm decides WHEN, and the
                                 grounding reference decides WHERE. Takes no
                                 tau and no failure_class -- enforced by
                                 signature, not by convention.
  compute_checkpoint_oracle  -- ground-truth tau. Explicitly named, reported
                                 only as arm="oracle_upper_bound".

Three things the primary rule does that the first version did not, each
because the forensics measured it failing
(`results/rollback_failure_forensics.md`):

  grounded, not merely non-error. A wrong document and a garbled result both
  arrive with `is_error=False`; resuming after them keeps the poisoned
  evidence and spends the budget that was needed to redo it.

  discarded evidence is regenerated. The lookups the rollback throws away
  are re-issued live before the model resumes, so a task-required call that
  happened after the fault is not lost with it. Re-issued rather than
  replayed: replaying would carry the corrupted results straight back in.
  Only calls the effect policy says are safe to repeat are re-issued.

  the retry is told something. A bare resample is the control, not the
  policy; the ladder in `repair_policy` picks a rung from causal evidence.

Repair is bounded by a budget declared before evaluation, and every attempt
reports what it consumed. `success_fn` scores the repaired run afterwards and
never reaches the repair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from derail.harness.tools import ToolRegistry
from derail.intervene.checkpoint import (CHECKPOINT_RULE_VERSION,
                                         GroundingReference,
                                         discarded_evidence_calls,
                                         last_grounded_step_before)
from derail.intervene.effects import EffectPolicy, safe_checkpoint
from derail.intervene.repair_policy import (RepairBudget, ResourceLedger,
                                            repair_message, select_rung)
from derail.intervene.rollback import rebuild_history_from
from derail.telemetry.events import SCHEMA_VERSION, make_tool_event, parse_step_events

PRIMARY = "primary"
ORACLE_UPPER_BOUND = "oracle_upper_bound"

#: Outcome taxonomy. `not_triggered` and `reconstruction_failed` are
#: terminal (no retry ran); the rest are set after a retry actually executes.
NOT_TRIGGERED = "not_triggered"
RECONSTRUCTION_FAILED = "reconstruction_failed"
RECOVERED = "recovered"
STILL_WRONG = "still_wrong"
HALTED = "halted"
BUDGET_EXHAUSTED = "budget_exhausted"


def _last_successful_tool_step_before(steps: list[dict], limit: int) -> int:
    """Index just after the last step with >=1 non-error tool call, strictly
    before `limit`. 0 if none exists (a valid `checkpoint_at_start`).

    The PRE-FIX rule, kept so the old result stays reproducible and so the
    grounded rule has something to be compared against. Not used by the
    primary arm any more -- see `compute_checkpoint_primary`.
    """
    last = 0
    for t, s in enumerate(steps[:limit]):
        evs, _ = parse_step_events(s)
        if evs and any(not e.is_error for e in evs):
            last = t + 1
    return last


@dataclass
class CheckpointResult:
    k: int
    outcome: str          # "normal" | "checkpoint_at_start" | "not_triggered"
    alarm_step: int | None = None
    rule: str = "last_successful/v0"
    #: Distinct successful calls in the discarded span, to be re-issued live
    #: before the model resumes. Empty when nothing is thrown away.
    regenerate: tuple[tuple[str, dict], ...] = ()
    #: Effectful calls that stopped the checkpoint moving further back.
    blocked_by_effects: tuple[tuple[int, str], ...] = ()
    #: Which channel raised the alarm: "behaviour" or "grounding_override".
    alarm_source: str = ""


def compute_checkpoint_primary(
        steps: list[dict], monitor, theta_b5: float,
        episode_from_trace: Callable, *,
        reference: GroundingReference | None = None,
        effects: EffectPolicy | None = None,
        override: bool = False) -> CheckpointResult:
    """Online-observable checkpoint. Signature is the enforcement: this
    function receives only `steps`, the frozen monitor and references fitted
    before serving -- there is no parameter through which tau, a failure
    class, a verdict or a `success_fn` could reach it.

    With no grounding reference this falls back to the pre-fix rule, so the
    old arm stays runnable from the same entry point.
    """
    ep = episode_from_trace(steps, "checkpoint-scan",
                            use_sentence_transformers=False,
                            grounding=reference is not None,
                            extended=reference is None)
    monitor.start_episode()
    alarm_step, alarm_source = None, ""
    for t in range(ep.X.shape[0]):
        # The monitor was fitted on width-51 telemetry; the grounding build
        # only appends dims, so the prefix it scores is the same vector it
        # would have been handed by the narrower build.
        if monitor.score_step(ep.X[t][:_MONITOR_WIDTH]) > theta_b5:
            alarm_step, alarm_source = t, "behaviour"
            break
        # The content channel may raise an alarm on its own, but only above
        # the level no healthy step reached, so it spends no false-alarm
        # budget and cannot make the behavioural channels less sensitive.
        if override and reference is not None and reference.should_alarm(ep.X[t]):
            alarm_step, alarm_source = t, "grounding_override"
            break
    if alarm_step is None:
        return CheckpointResult(k=0, outcome=NOT_TRIGGERED,
                                rule=(CHECKPOINT_RULE_VERSION if reference
                                      else "last_successful/v0"))
    if reference is None:
        k = _last_successful_tool_step_before(steps, alarm_step)
        return CheckpointResult(
            k=k, outcome=("checkpoint_at_start" if k == 0 else "normal"),
            alarm_step=alarm_step, rule="last_successful/v0",
            alarm_source=alarm_source)

    policy = effects or EffectPolicy()
    k = last_grounded_step_before(steps, ep.X, alarm_step, reference)
    k, blocking = safe_checkpoint(steps, k, policy)
    regenerate = [(name, args) for name, args
                  in discarded_evidence_calls(steps, k)
                  if policy.is_replay_safe(name)]
    return CheckpointResult(
        k=k, outcome=("checkpoint_at_start" if k == 0 else "normal"),
        alarm_step=alarm_step, rule=CHECKPOINT_RULE_VERSION,
        regenerate=tuple(regenerate), alarm_source=alarm_source,
        blocked_by_effects=tuple(blocking))


#: Width of the telemetry the frozen monitor's standardizer was fitted on.
_MONITOR_WIDTH = 51


def compute_checkpoint_oracle(steps: list[dict], tau: int) -> CheckpointResult:
    """Ground-truth-tau checkpoint -- SECONDARY, oracle-localized upper
    bound only. Never the primary recovery result."""
    k = _last_successful_tool_step_before(steps, tau)
    return CheckpointResult(k=k, rule="oracle_tau/v0",
                            outcome="checkpoint_at_start" if k == 0 else "normal")


@dataclass
class RetryOutcome:
    arm: str                       # PRIMARY | ORACLE_UPPER_BOUND
    checkpoint: CheckpointResult
    steps: list[dict] = field(default_factory=list)
    outcome: str = ""              # RECOVERED | STILL_WRONG | HALTED |
                                    # NOT_TRIGGERED | RECONSTRUCTION_FAILED
    success: bool | None = None
    error: str | None = None
    rung: str = "resample"
    ledger: ResourceLedger = field(default_factory=ResourceLedger)


def retry_real_task_from_checkpoint(
        steps: list[dict], task_prompt: str, checkpoint: CheckpointResult,
        arm: str, *, model: str = "qwen2.5:7b", temperature: float = 0.2,
        max_steps: int | None = None, registry: ToolRegistry,
        backend_factory: Callable[..., Any],
        rung: str = "resample",
        hint: str | None = None,
        budget: RepairBudget | None = None,
        effects: EffectPolicy | None = None,
        ledger: ResourceLedger | None = None) -> RetryOutcome:
    """Roll back to `checkpoint.k` and re-run live to completion or halt.

    Continuation tool calls pass `cassette=None` explicitly (not
    `mode="live"`) -- the ToolRegistry.call/Cassette.call dispatch calls the
    live function unconditionally whenever no cassette is given, so every
    call issued here is structurally live, never a replay, verified by the
    `source` field this same call path records.

    `backend_factory(model, tool_specs, temperature) -> OllamaBackend`-shaped
    object exposing reset/step/add_tool_results, matching rollback.py's
    existing contract -- kept as a parameter rather than importing
    OllamaBackend directly, so this stays a pure continuation runner testable
    against a fixture backend (see the equivalence proof).

    `hint` is the ladder's message, built by `repair_policy.repair_message`
    from causal evidence. It is passed in rather than built here so this
    runner never needs the reference, the alarm or the task spec, and so a
    caller cannot accidentally route ground truth through it.
    """
    budget = budget or RepairBudget()
    effects = effects or EffectPolicy()
    ledger = ledger if ledger is not None else ResourceLedger()
    steps_allowed = max_steps if max_steps is not None else budget.retry_steps

    if checkpoint.outcome == NOT_TRIGGERED:
        return RetryOutcome(arm=arm, checkpoint=checkpoint,
                            outcome=NOT_TRIGGERED, success=None, rung=rung,
                            ledger=ledger)

    ledger.attempts += 1
    started = time.perf_counter()
    try:
        backend = backend_factory(model=model, tool_specs=registry.specs(),
                                  temperature=temperature)
        rebuild_history_from(backend, task_prompt, steps,
                             range(checkpoint.k))
        if hint:
            backend.history.append({"role": "user", "content": hint})
    except Exception as exc:                              # noqa: BLE001
        ledger.seconds += time.perf_counter() - started
        return RetryOutcome(arm=arm, checkpoint=checkpoint,
                            outcome=RECONSTRUCTION_FAILED, success=None,
                            error=f"{type(exc).__name__}: {exc}", rung=rung,
                            ledger=ledger)

    out_steps = list(steps[:checkpoint.k])
    regenerated = _regenerate(checkpoint.regenerate, registry, backend,
                              budget, ledger)
    out_steps += regenerated
    for t in range(len(out_steps), len(out_steps) + steps_allowed):
        over = ledger.over(budget)
        if over:
            ledger.exhausted = over
            ledger.seconds += time.perf_counter() - started
            return RetryOutcome(arm=arm, checkpoint=checkpoint,
                                steps=out_steps, outcome=BUDGET_EXHAUSTED,
                                success=False, rung=rung, ledger=ledger)
        t0 = time.perf_counter()
        try:
            out = backend.step(t)
        except Exception as exc:                          # noqa: BLE001
            ledger.seconds += time.perf_counter() - started
            return RetryOutcome(arm=arm, checkpoint=checkpoint,
                                steps=out_steps, outcome=HALTED, success=False,
                                error=f"{type(exc).__name__}: {exc}", rung=rung,
                                ledger=ledger)
        ledger.model_calls += 1
        ledger.steps_run += 1
        lat = time.perf_counter() - t0
        bits, events, step_error, results = [], [], False, []
        for use in out.get("tool_uses", []):
            if not effects.is_replay_safe(use["name"]):
                # A retry may not re-issue a call that changes the world and
                # cannot be undone. Refusing is the honest move: the
                # alternative is a second booking, payment or message.
                results.append({"id": use["id"], "name": use["name"],
                                "content": ("Refused: this tool changes state "
                                            "and cannot be repeated during a "
                                            "recovery attempt."),
                                "is_error": True})
                step_error = True
                continue
            res = registry.call(use["name"], use["input"], cassette=None,
                                record_provenance=True)
            ledger.tool_calls += 1
            step_error = step_error or res.is_error
            bits.append(res.step_bit())
            results.append({"id": use["id"], "name": use["name"],
                            "content": res.content, "is_error": res.is_error})
            events.append(make_tool_event(use["name"], use["input"],
                                          res.content, is_error=res.is_error,
                                          latency_s=res.latency_s,
                                          call_id=use["id"], source=res.source))
            assert res.source in ("live_external", "live_local"), (
                f"continuation call for {use['name']!r} returned "
                f"source={res.source!r} -- must be live, never a replay")
        if results:
            backend.add_tool_results(results)
        record = {
            "text": out.get("text", ""), "action": out.get("action", ""),
            "latency_s": round(lat, 4), "error": step_error,
            "tool_events": events, "schema": SCHEMA_VERSION,
            "resumed_from": {"checkpoint_step": checkpoint.k,
                             "regenerated": len(regenerated),
                             "rung": rung},
        }
        out_steps.append(record)
        if out.get("stop_reason") == "end_turn":
            break
    else:
        ledger.seconds += time.perf_counter() - started
        return RetryOutcome(arm=arm, checkpoint=checkpoint, steps=out_steps,
                            outcome=HALTED, success=False, rung=rung,
                            ledger=ledger)

    ledger.seconds += time.perf_counter() - started
    return RetryOutcome(arm=arm, checkpoint=checkpoint, steps=out_steps,
                        outcome="_scored_by_caller", success=None, rung=rung,
                        ledger=ledger)


def _regenerate(calls: tuple[tuple[str, dict], ...], registry: ToolRegistry,
                backend, budget: RepairBudget,
                ledger: ResourceLedger) -> list[dict]:
    """Re-issue the discarded lookups live, and put the fresh results in the
    conversation before the model resumes.

    The agent gets its evidence back without the corruption that came with it
    the first time, and without spending model calls to rediscover it. Every
    call here is counted against the tool budget: regeneration is cheaper than
    re-reasoning, not free.

    Written into the history as tool messages attached to one assistant turn
    that says what it did. That turn is the recovery layer speaking, and it
    says so -- this is the one message in the rebuilt conversation that is not
    a replay of something the agent produced, and labelling it is how the
    trace stays honest about where it came from.
    """
    if not calls:
        return []
    uses, results, events = [], [], []
    for index, (name, args) in enumerate(calls):
        if ledger.tool_calls >= budget.max_extra_tool_calls:
            ledger.exhausted = "max_extra_tool_calls"
            break
        result = registry.call(name, args, cassette=None,
                               record_provenance=True)
        ledger.tool_calls += 1
        call_id = f"regen-{index}"
        uses.append({"function": {"name": name, "arguments": args}})
        results.append({"id": call_id, "name": name, "content": result.content,
                        "is_error": result.is_error})
        events.append(make_tool_event(name, args, result.content,
                                      is_error=result.is_error,
                                      latency_s=result.latency_s,
                                      call_id=call_id, source=result.source))
    if not events:
        return []
    backend.history.append({
        "role": "assistant",
        "content": ("Recovering: re-running the lookups that were discarded "
                    "when this run was rolled back."),
        "tool_calls": uses})
    backend.add_tool_results(results)
    return [{
        "text": "", "action": "tool_call", "latency_s": 0.0,
        "error": any(bool(e.get("is_error")) for e in events),
        "tool_events": events,
        "schema": SCHEMA_VERSION,
        "regenerated_evidence": True,
    }]


def plan_repair(steps: list[dict], X: np.ndarray, checkpoint: CheckpointResult,
                reference: GroundingReference,
                required_tools: tuple[str, ...] = (),
                allow_coverage: bool = False) -> tuple[str, str | None]:
    """(rung, hint) for this alarm, from causal evidence only.

    Split out from the runner so the decision can be tested without a model
    and so the runner never holds the task spec.
    """
    if checkpoint.alarm_step is None:
        return "resample", None
    rung = select_rung(steps, X, checkpoint.alarm_step, reference,
                       required_tools=required_tools,
                       allow_coverage=allow_coverage)
    return rung, repair_message(rung, steps=steps, X=X,
                                alarm_step=checkpoint.alarm_step,
                                reference=reference,
                                required_tools=required_tools)
