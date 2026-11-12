"""Forensic accounting of every closed-loop repair that did not recover.

Diagnosis only, on development data. Reads the committed outcomes of the live
rollback/retry study (`results/tables/real_task_rollback_outcomes.csv`) and
the corpus it ran on, and reconstructs, per episode and per arm, the state
the repair mechanism was actually working from -- which is what separates a
detection failure from a repair failure.

Nothing here re-runs the agent: every field is derived from what was
recorded. Where a mechanism cannot be settled from the record, the episode is
left classified as unresolved rather than guessed.

Writes:
  results/tables/rollback_failure_rootcause.csv
  results/rollback_failure_forensics.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from derail.experiments.run_real_task_rollback import (CORPUS_DIR, MAX_STEPS,
                                                       TASK_BY_NAME)
from derail.intervene.real_tool_rollback import _last_successful_tool_step_before
from derail.telemetry.events import parse_step_events

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = REPO_ROOT / "results" / "tables" / "real_task_rollback_outcomes.csv"
OUT_CSV = REPO_ROOT / "results" / "tables" / "rollback_failure_rootcause.csv"
OUT_REPORT = REPO_ROOT / "results" / "rollback_failure_forensics.md"

#: Below this many steps a continuation cannot plausibly finish a task whose
#: healthy runs take 4-12 steps; recorded as a measured budget, not a cause on
#: its own.
MIN_USEFUL_RETRY_STEPS = 4


def _tools_in(steps: list[dict]) -> set[str]:
    """Names of tools that actually executed without error in these steps."""
    return {e.name for s in steps for e in parse_step_events(s)[0]
            if not e.is_error}


def classify(row: dict) -> str:
    if row["outcome"] == "not_triggered":
        return "detection_gap_no_trigger"
    if row["outcome"] == "reconstruction_failed":
        return "state_reconstruction_failed"
    if row["outcome"] == "recovered":
        return ""
    if row["outcome"] == "halted":
        return ("retry_budget_exhausted_by_late_checkpoint"
                if row["retry_budget_steps"] < MIN_USEFUL_RETRY_STEPS
                else "retry_halted_within_adequate_budget")
    # still_wrong
    if row["required_tools_only_in_discarded_suffix"]:
        return "rollback_discards_verifier_required_tool_call"
    if row["checkpoint_after_onset"]:
        return "checkpoint_lands_after_onset"
    return "repair_policy_insufficient_bare_resample"


def main() -> None:
    outcomes = pd.read_csv(IN_CSV)
    manifest = {e["episode_id"]: e for e in
                json.loads((CORPUS_DIR / "manifest.json").read_text("utf-8"))}

    rows = []
    for _, r in outcomes.iterrows():
        entry = manifest[r["episode_id"]]
        steps = [json.loads(line) for line in
                 (CORPUS_DIR / entry["file"]).read_text("utf-8").splitlines()
                 if line.strip()]
        task = TASK_BY_NAME[r["task_name"]]
        k = int(r["checkpoint_k"]) if pd.notna(r["checkpoint_k"]) else 0
        tau = int(r["tau"])
        kept, discarded = _tools_in(steps[:k]), _tools_in(steps[k:])
        # The task's own allowlist is the closest machine-readable statement
        # of what its verifier needs executed; success_fn is a closure and
        # cannot be introspected.
        required = set(task.tools)
        row = {
            "dataset": r["dataset"],
            "episode_id": r["episode_id"], "task": r["task_name"],
            "failure_class": r["failure_class"], "tau": tau, "T": len(steps),
            "arm": r["arm"],
            "alarm_step": r["alarm_step"], "checkpoint_k": k,
            "checkpoint_outcome": r["checkpoint_outcome"],
            "outcome": r["outcome"], "success": r["success"],
            "checkpoint_after_onset": k > tau,
            "checkpoint_at_last_successful_tool":
                k == _last_successful_tool_step_before(
                    steps, int(r["alarm_step"]) if pd.notna(r["alarm_step"])
                    else tau),
            "retry_budget_steps": MAX_STEPS - k,
            "required_tools": ";".join(sorted(required)),
            "required_tools_kept": ";".join(sorted(required & kept)),
            "required_tools_only_in_discarded_suffix": bool(
                required & discarded - kept),
            "tools_in_discarded_suffix": ";".join(sorted(discarded - kept)),
        }
        row["root_cause"] = classify(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    primary = df[df["arm"] == "primary"]
    oracle = df[df["arm"] == "oracle_upper_bound"]
    lines = ["# Forensic accounting: every closed-loop repair that failed",
             "",
             "**SUPPORTING / diagnostic evidence, on development data.** "
             "Derived entirely from the committed outcomes of the live "
             "rollback/retry study; no agent was re-run. The primary arm is "
             "the deployable one -- the frozen monitor's own causal alarm "
             "chooses the checkpoint. The oracle arm is given ground-truth "
             "tau and is an upper bound only.",
             "",
             f"Retry budget is `max_steps - checkpoint_k` against a global "
             f"`MAX_STEPS = {MAX_STEPS}`: the continuation does NOT get a "
             f"fresh budget, so a late checkpoint leaves fewer steps to "
             f"finish the task than the original run had.",
             "",
             "## Primary arm", "",
             "| episode | class | tau | alarm | ck | retry budget | outcome | "
             "root cause |", "|---|---|---|---|---|---|---|---|"]
    for _, r in primary.iterrows():
        lines.append(
            f"| {r['episode_id']} | {r['failure_class']} | {r['tau']} | "
            f"{'-' if pd.isna(r['alarm_step']) else int(r['alarm_step'])} | "
            f"{r['checkpoint_k']} | {r['retry_budget_steps']} | "
            f"{r['outcome']} | {r['root_cause'] or 'recovered'} |")
    lines += ["", "### Primary root-cause counts", ""]
    for cause, n in primary[primary["root_cause"] != ""]["root_cause"] \
            .value_counts().items():
        lines.append(f"- **{cause}**: {n}")
    lines += ["", "## Oracle arm -- what remains unrepairable with a perfect "
              "trigger", "",
              "| episode | class | tau | ck | outcome | root cause |",
              "|---|---|---|---|---|---|"]
    for _, r in oracle[oracle["root_cause"] != ""].iterrows():
        lines.append(f"| {r['episode_id']} | {r['failure_class']} | "
                     f"{r['tau']} | {r['checkpoint_k']} | {r['outcome']} | "
                     f"{r['root_cause']} |")
    lines += ["", "## Per-class closed-loop recovery (primary arm)", "",
              "| class | recovered / n |", "|---|---|"]
    for cls, grp in primary.groupby("failure_class"):
        lines.append(f"| {cls} | {int((grp['outcome'] == 'recovered').sum())} "
                     f"/ {len(grp)} |")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(df[df["root_cause"] != ""].groupby(["arm", "root_cause"])
          .size().to_string())
    for path in (OUT_CSV, OUT_REPORT):
        print(f"[rollback-forensics] wrote {path}")


if __name__ == "__main__":
    main()
