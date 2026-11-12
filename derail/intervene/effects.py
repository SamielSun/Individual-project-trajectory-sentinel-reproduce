"""What rollback is allowed to do to a tool call that changed something.

Rolling a conversation back is free. Rolling the WORLD back is not, and a
rollback framework that quietly assumes otherwise is only correct for
read-only tools. The tools this task suite ships are read-only -- search,
retrieval, file reads, weather, and a SQL tool that opens its database
`mode=ro` with `query_only` set -- so nothing in the committed corpora
exercises the problem. That is a fact about the corpus, not a property of the
design, and it is exactly the kind of gap that goes unnoticed until a
deployment attaches a tool that books, writes, pays or sends.

So the rule is declared per tool, and rollback obeys it:

  READ_ONLY   the call observed the world and changed nothing. Discarding it
              costs only the observation; re-issuing it is safe.
  IDEMPOTENT  the call changed something, but repeating it with the same
              arguments leaves the same state. Safe to re-issue; still not
              free, so it is counted.
  EFFECTFUL   the call changed something and repeating it would change it
              again. Rollback may not silently discard it, because the
              conversation would then no longer describe the world, and the
              retry may not re-issue it.

An EFFECTFUL call inside the discarded span leaves the run in a state the
retry cannot honestly start from. Two ways out, and a deployment must have
one of them:

  compensation  a declared inverse call that undoes it. Rollback issues the
                compensation, records that it did, and proceeds.
  refusal       no compensation exists, so the checkpoint is not allowed to
                move behind that call. The rollback stops there instead --
                a shorter, honest rollback rather than a longer one that
                lies about the world.

Unknown tools are treated as EFFECTFUL. A tool nobody classified is exactly
the one most likely to do something irreversible, and defaulting an unknown
to "safe" is how a framework quietly breaks the guarantee it advertises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from derail.telemetry.events import parse_step_events

READ_ONLY = "read_only"
IDEMPOTENT = "idempotent"
EFFECTFUL = "effectful"
EFFECT_CLASSES = (READ_ONLY, IDEMPOTENT, EFFECTFUL)

#: The shipped suite, classified explicitly. Every one is read-only: the
#: search and retrieval tools issue GETs, the file tools only read, and the
#: SQL tool refuses anything but a single read statement at three separate
#: layers. `python` is a sandboxed interpreter with no network and a
#: scratch filesystem, so it cannot reach anything the next call would see.
DEFAULT_EFFECTS: dict[str, str] = {
    "arxiv_search": READ_ONLY,
    "wikipedia_search": READ_ONLY,
    "web_search": READ_ONLY,
    "web_fetch": READ_ONLY,
    "get_weather": READ_ONLY,
    "read_file": READ_ONLY,
    "list_dir": READ_ONLY,
    "github_tool": READ_ONLY,
    "sql_query": READ_ONLY,
    "vector_search": READ_ONLY,
    "python": READ_ONLY,
    "calculator": READ_ONLY,
}


class UncompensatedEffect(RuntimeError):
    """Raised when a rollback would discard an effect it cannot undo."""


@dataclass(frozen=True)
class Compensation:
    """The declared inverse of an effectful call.

    `build` maps the original call's arguments to the compensating call's
    arguments. It is declared with the tool, never inferred: guessing an
    inverse is how a rollback deletes the wrong record.
    """

    tool: str
    build: Callable[[dict], dict]


@dataclass
class EffectPolicy:
    """Which tools change the world, and how to undo the ones that do."""

    effects: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EFFECTS))
    compensations: dict[str, Compensation] = field(default_factory=dict)

    def effect_of(self, tool: str) -> str:
        """Unknown means EFFECTFUL. See the module docstring."""
        return self.effects.get(tool, EFFECTFUL)

    def is_replay_safe(self, tool: str) -> bool:
        """May the retry issue this call again?"""
        return self.effect_of(tool) in (READ_ONLY, IDEMPOTENT)

    def uncompensated_effects(self, steps: list[dict], start: int,
                              stop: int) -> list[tuple[int, str]]:
        """Effectful calls in `steps[start:stop]` with no declared inverse.

        Errored calls are skipped: a call that failed did not change anything
        it needs undoing for, and treating it as an effect would block
        rollbacks that are perfectly safe.
        """
        out = []
        for t in range(start, min(stop, len(steps))):
            for event in parse_step_events(steps[t])[0]:
                if event.is_error:
                    continue
                if (self.effect_of(event.name) == EFFECTFUL
                        and event.name not in self.compensations):
                    out.append((t, event.name))
        return out

    def compensating_calls(self, steps: list[dict], start: int,
                           stop: int) -> list[tuple[str, dict]]:
        """Inverse calls for the effectful span, newest first.

        Reverse order because effects compose: undoing them in the order they
        were made can leave the world in a state the earlier inverse was not
        written for.
        """
        out = []
        for t in range(start, min(stop, len(steps))):
            for event in parse_step_events(steps[t])[0]:
                if event.is_error:
                    continue
                compensation = self.compensations.get(event.name)
                if compensation is not None:
                    out.append((compensation.tool,
                                compensation.build(dict(event.args or {}))))
        return list(reversed(out))


def safe_checkpoint(steps: list[dict], proposed: int,
                    policy: EffectPolicy) -> tuple[int, list[tuple[int, str]]]:
    """The earliest checkpoint at or after `proposed` that rollback may take.

    Returns (checkpoint, blocking) -- `blocking` is the uncompensated
    effectful calls that pushed it later, empty when the proposal stood. A
    caller that wants the rollback to happen anyway must supply a
    compensation; there is no flag here to waive the guarantee.
    """
    blocking = policy.uncompensated_effects(steps, proposed, len(steps))
    if not blocking:
        return proposed, []
    # Resume after the last call that cannot be undone: everything before it
    # is still safely discardable, and the world stays described by the
    # history the retry starts from.
    return max(t for t, _ in blocking) + 1, blocking


def assert_no_duplicate_effects(before: list[dict], after: list[dict],
                                policy: EffectPolicy) -> None:
    """Every effectful call in the repaired run happened at most once.

    Compares the (tool, arguments) multiset across the whole repaired trace.
    An idempotent call may legitimately repeat; an effectful one may not, and
    a retry that re-issues one has charged the card twice.
    """
    counts: dict[tuple[str, str], int] = {}
    for step in list(before) + list(after):
        for event in parse_step_events(step)[0]:
            if event.is_error or policy.effect_of(event.name) != EFFECTFUL:
                continue
            key = (event.name, event.args_key)
            counts[key] = counts.get(key, 0) + 1
    duplicated = {k: n for k, n in counts.items() if n > 1}
    if duplicated:
        raise UncompensatedEffect(
            f"repaired run repeats effectful calls {sorted(duplicated)}; a "
            f"retry may not re-issue a call that changes the world")
