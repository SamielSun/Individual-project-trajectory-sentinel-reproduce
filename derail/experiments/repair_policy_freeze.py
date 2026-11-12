"""Freeze what the repair path is allowed to know and to spend.

The detector freeze (`framework_monitor_freeze.json`) is immutable and stays
exactly as it is. Repair needs its own frozen inputs, and they are separate
artifacts for the same reason the detector's are: a checkpoint rule that can
be retuned after seeing recovery numbers is not a rule, and a retry budget
chosen after the fact reports a capability the deployment does not have.

Three things are frozen here, all decided before any repaired run is scored:

  grounding reference  per-dim healthy location and scale for the content
                       channel, plus the level above which a step is not
                       trusted. It decides which steps carry evidence worth
                       resuming from and which observations survive a
                       rollback. Fitted on healthy runs of the SAME TASKS the
                       repair serves, which is the per-deployment principle
                       the detector threshold already follows. Fitting it on
                       the detector freeze's own corpora was tried first and
                       does not work: those are a different task, and against
                       their reference an ordinary arXiv result sits 15
                       standard units out, putting the trip above every step
                       in the rollback corpus. The artifact records a
                       transfer check -- the rollback corpus's own healthy
                       runs scored against this reference -- so a reader can
                       see the reference applies to what it is used on.
  repair budget        attempts, model calls, tool calls, wall clock and
                       continuation steps.
  effect policy        which tools change the world. Recorded so a later
                       reader can see that this suite is entirely read-only,
                       and so adding a tool that is not forces a decision.

No failure label, onset, verdict or `success_fn` is read anywhere in this
module. `success_fn` scores repaired runs afterwards.

Writes:
  results/repair_policy_freeze.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from derail.experiments.framework_monitor_freeze import (OUT_PATH as FREEZE_PATH,
                                                         TRACES_ROOT, _git)
from derail.intervene.checkpoint import (CHECKPOINT_RULE_VERSION,
                                         fit_grounding_reference)
from derail.intervene.effects import DEFAULT_EFFECTS
from derail.intervene.repair_policy import (HEADLINE_RUNGS,
                                            REPAIR_POLICY_VERSION, RUNGS,
                                            RepairBudget)
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "results" / "repair_policy_freeze.json"


#: Healthy traffic the grounding reference is fitted on: real-tool runs of the
#: same tasks the repair path serves. Development corpora, already spent.
REFERENCE_CORPORA = ("langgraph7b_real2", "autogen7b_real2")

#: Scored against the fitted reference and recorded, never fitted on. If these
#: sat outside the null the reference would not apply to the runs it is used
#: on, and the artifact would be saying so.
TRANSFER_CHECK_CORPUS = "real_task_rollback"


def _healthy_episodes(corpus: str) -> tuple[list, list[str]]:
    """Every healthy episode of a corpus, with the grounding channel."""
    manifest = json.loads((TRACES_ROOT / corpus / "manifest.json")
                          .read_text("utf-8"))
    episodes, ids = [], []
    for entry in sorted((e for e in manifest
                         if e.get("failure_class") is None),
                        key=lambda e: e["episode_id"]):
        steps = [json.loads(line) for line in
                 (TRACES_ROOT / corpus / entry["file"])
                 .read_text("utf-8").splitlines() if line.strip()]
        episodes.append(episode_from_trace(
            steps, f"{corpus}/{entry['episode_id']}",
            use_sentence_transformers=False, grounding=True))
        ids.append(f"{corpus}/{entry['episode_id']}")
    return episodes, ids


def _load_training_episodes() -> tuple[list, list[str]]:
    episodes, ids = [], []
    for corpus in REFERENCE_CORPORA:
        corpus_episodes, corpus_ids = _healthy_episodes(corpus)
        episodes += corpus_episodes
        ids += corpus_ids
    if len(episodes) < 40:
        raise SystemExit(f"only {len(episodes)} healthy episodes across "
                         f"{REFERENCE_CORPORA} -- too few for a grounding "
                         f"reference")
    return episodes, ids


def _transfer_check(reference) -> dict:
    """How the reference reads on healthy runs it was NOT fitted on."""
    episodes, ids = _healthy_episodes(TRANSFER_CHECK_CORPUS)
    scores = [reference.score(ep.X[t]) for ep in episodes
              for t in range(ep.T)]
    return {
        "corpus": TRANSFER_CHECK_CORPUS,
        "n_episodes": len(ids), "n_steps": len(scores),
        "median": float(np.median(scores)) if scores else float("nan"),
        "max": float(max(scores)) if scores else float("nan"),
        "share_above_trip": (float(np.mean([s > reference.trip
                                            for s in scores]))
                             if scores else float("nan")),
    }


def build() -> dict:
    episodes, ids = _load_training_episodes()
    reference = fit_grounding_reference(episodes)
    budget = RepairBudget()
    return {
        "repair_policy_version": REPAIR_POLICY_VERSION,
        "checkpoint_rule_version": CHECKPOINT_RULE_VERSION,
        "detector_freeze_sha256": hashlib.sha256(
            FREEZE_PATH.read_bytes()).hexdigest(),
        "fit_episode_ids": ids,
        "fit_episode_count": len(ids),
        "fit_input_hash": hashlib.sha256(
            "\n".join(ids).encode("utf-8")).hexdigest(),
        "grounding_reference": reference.as_record(),
        "reference_corpora": list(REFERENCE_CORPORA),
        "transfer_check": _transfer_check(reference),
        "budget": budget.as_record(),
        "effect_policy": dict(sorted(DEFAULT_EFFECTS.items())),
        "rungs": list(RUNGS),
        "headline_rungs": list(HEADLINE_RUNGS),
        "code_commit": _git("rev-parse", "HEAD"),
    }


def main() -> None:
    artifact = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", "utf-8")
    reference = artifact["grounding_reference"]
    print(f"[repair-freeze] fitted on {artifact['fit_episode_count']} healthy "
          f"episodes ({reference['n_healthy_steps']} steps)")
    print(f"[repair-freeze] grounding trip = {reference['trip']:.4f}")
    print(f"[repair-freeze] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
