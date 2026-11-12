"""What repair is allowed to spend, and what it is allowed to say.

Two things a recovery result is worthless without.

**A budget fixed before the numbers.** A retry loop with no ceiling reaches
100% on anything eventually, and reports a capability the deployment does not
have. So the resources are declared here, in advance, and every run reports
what it actually consumed against them: attempts, model calls, tool calls,
wall clock. First-attempt success and eventual-success-within-budget are
reported separately, because they are different products.

**A ladder that knows nothing it should not.** Each rung adds exactly one
mechanism, so the study can say which one does the work, and every rung is
built from signals the monitor already has at alarm time:

  resample   rollback, same context, new sample. The control. A stochastic
             agent improves on retry alone, so no rung above may be credited
             unless it beats this.
  generic    a task-independent reminder to re-check the work before
             answering.
  unstick    for a run repeating one call or hitting a failing tool: stop
             calling it, finish with what you have. Selected from the step
             record's own repeat/error structure.
  reground   for a run whose evidence does not match what it asked for:
             the grounding reference names WHICH property is off (result
             against its query, against the run's other results, character
             shape) and the rung says to re-gather that evidence. It names
             the property, never a value and never a document.
  coverage   names the tools the TASK declares it requires that the run has
             not yet used successfully.

`coverage` is reported separately from every other rung, and never pooled into
a headline. It is legitimate in production -- the deployer wrote the task and
knows what it asks for -- but the task's `success_fn` also checks that those
tools were called, so a gain from this rung partly reflects being told what
the verifier will look at. Keeping it separate is the same distinction
`rollback.py` draws between `located` and `specific`, for the same reason.

Forbidden to every rung, and unreachable through any signature here: the
injected onset, the failure class, the final verdict, `success_fn`, and any
tool result from a step after the alarm. `success_fn` scores the repaired run
afterwards; it never shapes the repair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from derail.intervene.checkpoint import GroundingReference
from derail.telemetry.events import parse_step_events

#: Rungs in order of how much they tell the agent. `coverage` is last because
#: it is the only one that overlaps what the verifier checks.
RUNGS = ("resample", "generic", "unstick", "reground", "coverage")

#: Rungs whose evidence may be pooled into a headline recovery number.
#: `coverage` is excluded by construction, not by choice at reporting time.
HEADLINE_RUNGS = ("resample", "generic", "unstick", "reground")

REPAIR_POLICY_VERSION = "bounded_ladder/v1"

_GENERIC = ("Before you give the final answer, re-check your work: make sure "
            "you have carried out every step the task asks for, and that your "
            "answer rests on results you actually obtained.")

_UNSTICK = ("A tool is returning errors or the same result to repeated calls. "
            "Do not call it again with the same arguments. Work with what you "
            "have already obtained and give the final answer, noting anything "
            "you could not look up.")

#: What each grounding dim means, in words the agent can act on. These name a
#: PROPERTY of the evidence, never its content: no value, document or answer
#: appears here, so the rung locates a fault without supplying a result.
_REGROUND_BY_DIM = {
    "query_dis": "the results you received do not answer the queries you sent",
    "lex_miss": "the documents you received are not about what you asked for",
    "self_dis": "one result is inconsistent with the others you have gathered",
    "consec_dis": "the latest result does not follow from the previous one",
    "char_anom": "a result came back malformed or garbled",
    "json_broken": "a result came back structurally invalid",
    "reason_dis": "the results do not support what you have been concluding",
    "mem_dis": "you have drifted from the material you retrieved",
    "drift": "your results have been drifting away from the task",
}


@dataclass(frozen=True)
class RepairBudget:
    """Declared before evaluation. Nothing may exceed it to reach a number.

    `max_attempts` is 2 because the ladder's whole claim is that ONE informed
    retry beats one bare resample; a deep retry tree would answer a different
    question and would also let persistence stand in for repair.
    """

    max_attempts: int = 2
    max_extra_model_calls: int = 12
    max_extra_tool_calls: int = 24
    max_recovery_seconds: float = 300.0
    #: Steps the continuation may run. Reserved for the retry rather than
    #: shared with the original run's ceiling: on the rollback corpus a late
    #: checkpoint left one usable step out of a global twelve, so the retry
    #: was judged on a budget the original run had already spent.
    retry_steps: int = 12
    version: str = REPAIR_POLICY_VERSION

    def as_record(self) -> dict:
        return asdict(self)


@dataclass
class ResourceLedger:
    """What one repair actually consumed. Reported whether it succeeded."""

    attempts: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    steps_run: int = 0
    exhausted: str = ""

    def over(self, budget: RepairBudget) -> str:
        if self.attempts > budget.max_attempts:
            return "max_attempts"
        if self.model_calls > budget.max_extra_model_calls:
            return "max_extra_model_calls"
        if self.tool_calls > budget.max_extra_tool_calls:
            return "max_extra_tool_calls"
        if self.seconds > budget.max_recovery_seconds:
            return "max_recovery_seconds"
        return ""

    def as_record(self, prefix: str = "repair_") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def repeated_or_failing(steps: list[dict], upto: int) -> bool:
    """Is the run stuck against a tool? Repeats of an identical (name, args)
    call, or errors, in what has already happened."""
    seen, repeats, errors = set(), 0, 0
    for step in steps[:upto]:
        for event in parse_step_events(step)[0]:
            key = (event.name, event.args_key)
            repeats += key in seen
            seen.add(key)
            errors += event.is_error
    return repeats >= 2 or errors >= 2


def worst_grounding_dim(X: np.ndarray, t: int,
                        reference: GroundingReference) -> str | None:
    """Which grounding property is furthest outside the healthy reference at
    step `t`, or None when the step sits inside it."""
    z = ((np.asarray(X[t], dtype=float)[list(_indices(reference))]
          - np.asarray(reference.median)) / np.asarray(reference.scale))
    z = np.clip(z, 0.0, None)
    if float(np.max(z, initial=0.0)) <= reference.trip:
        return None
    return reference.dims[int(np.argmax(z))]


def _indices(reference: GroundingReference) -> tuple[int, ...]:
    from derail.intervene.checkpoint import _dim_indices
    return _dim_indices(reference.dims)


def missing_declared_tools(steps: list[dict], upto: int,
                           required_tools: tuple[str, ...]) -> tuple[str, ...]:
    """Tools the TASK declares it needs that have not yet succeeded.

    Reads the task's declared structure, which the deployer wrote and which is
    already in the prompt -- not the verifier. See the module docstring for
    why the rung built on this is reported separately anyway.
    """
    done = {e.name for s in steps[:upto] for e in parse_step_events(s)[0]
            if not e.is_error}
    return tuple(t for t in required_tools if t not in done)


def select_rung(steps: list[dict], X: np.ndarray, alarm_step: int,
                reference: GroundingReference,
                required_tools: tuple[str, ...] = (),
                allow_coverage: bool = False) -> str:
    """Which rung this alarm calls for, from causal evidence only.

    Order is by specificity of the evidence, not by strength of the hint: a
    stuck run is diagnosed from its own repeat structure, an ungrounded run
    from the grounding reference. When neither fires there is nothing specific
    to say and the rung is the task-independent reminder.
    """
    if allow_coverage and missing_declared_tools(steps, alarm_step + 1,
                                                 required_tools):
        return "coverage"
    if repeated_or_failing(steps, alarm_step + 1):
        return "unstick"
    if worst_grounding_dim(X, alarm_step, reference) is not None:
        return "reground"
    return "generic"


def repair_message(rung: str, *, steps: list[dict] | None = None,
                   X: np.ndarray | None = None, alarm_step: int | None = None,
                   reference: GroundingReference | None = None,
                   required_tools: tuple[str, ...] = ()) -> str | None:
    """The user-turn nudge appended at the checkpoint, or None.

    Every argument is something the run itself produced or the task declared.
    There is no parameter here through which an onset, a class, a verdict or a
    verifier could reach the prompt -- the guarantee is the signature, not a
    string test on the output.
    """
    if rung == "resample":
        return None
    if rung == "generic":
        return _GENERIC
    if rung == "unstick":
        return _UNSTICK
    if rung == "reground":
        dim = (None if X is None or alarm_step is None or reference is None
               else worst_grounding_dim(X, alarm_step, reference))
        if dim is None:
            return _GENERIC
        return (f"A consistency check of the evidence you have gathered found "
                f"that {_REGROUND_BY_DIM[dim]}. Gather that evidence again "
                f"before giving the final answer, and do not rely on the "
                f"result that failed the check.")
    if rung == "coverage":
        missing = (() if steps is None or alarm_step is None
                   else missing_declared_tools(steps, alarm_step + 1,
                                               required_tools))
        if not missing:
            return _GENERIC
        listed = ", ".join(sorted(missing))
        return (f"The task asks you to use these tools and you have not used "
                f"them successfully yet: {listed}. Do that before giving the "
                f"final answer.")
    raise ValueError(f"unknown rung {rung!r}")
