"""What each repair fix is worth, measured live, one mechanism at a time.

Development data only: the 16-episode real-tool rollback corpus, already spent
when its interior was opened. The pre-fix study
(`results/real_task_rollback_report.md`, `..._outcomes.csv`) is NOT touched --
it stays as the recorded baseline, and this writes beside it.

Six arms, each adding exactly one thing to the one before it, so a gain can
be attributed:

  pre_fix       the shipped configuration: the detector freeze's monitor
                (fitted on another deployment), resume after the last
                non-error tool step, no regeneration, no hint. Re-run here so
                the comparison is against a contemporaneous control rather
                than a number collected on a different day.
  recalibrated  the same rule, but the monitor is fitted and thresholded on
                THIS deployment's own healthy runs. Isolates the calibration,
                which the transfer evidence in `repair_deployment_freeze.json`
                says is the dominant defect: the served corpus runs at 2.90s
                median step latency against 0.52s in the freeze's fit corpora,
                and latency is a dimension inside the `m` channel.
  grounded      the checkpoint also asks whether the evidence is GROUNDED, the
                content channel may raise an alarm above the level no healthy
                step reached, and the lookups the rollback discards are
                re-issued live. Still no hint.
  ladder        adds the causal repair rung (`repair_policy.select_rung`).
  coverage      adds the rung that names task-declared tools the run has not
                used. SUPPORTING ONLY, never pooled: the task's own verifier
                also checks those tools were called, so a gain here partly
                reflects being told what will be looked at.
  oracle        ground-truth tau picks the checkpoint. SUPPORTING ONLY, an
                upper bound, never a deployable result.

A retry counts as repaired only when the task's own predeclared `success_fn`
says the final answer is correct. It is called once, after the run, and never
reaches the repair.

Every arm reports what it spent against the frozen budget: attempts, extra
model calls, extra tool calls, recovery seconds. First-attempt success and
success within budget are reported separately, because a number reached by
retrying is a different product from one reached first time.

Writes:
  results/tables/repair_study_outcomes.csv
  results/tables/repair_study_summary.csv
  results/repair_study_report.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from derail.evaluation.metrics import first_alarm  # noqa: F401  (parity import)
from derail.experiments.framework_monitor_freeze import load_frozen_monitor
from derail.experiments.repair_deployment_freeze import load_deployment_monitor
from derail.experiments.repair_policy_freeze import OUT_PATH as REPAIR_FREEZE
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.tasks import REAL_TASKS
from derail.intervene.checkpoint import GroundingReference
from derail.intervene.effects import (EffectPolicy,
                                      assert_no_duplicate_effects)
from derail.intervene.real_tool_rollback import (HALTED, NOT_TRIGGERED,
                                                 RECONSTRUCTION_FAILED,
                                                 BUDGET_EXHAUSTED,
                                                 RECOVERED, STILL_WRONG,
                                                 compute_checkpoint_oracle,
                                                 compute_checkpoint_primary,
                                                 plan_repair,
                                                 retry_real_task_from_checkpoint)
from derail.intervene.repair_policy import RepairBudget, ResourceLedger
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "traces" / "real_task_rollback"
OUT_CSV = REPO_ROOT / "results" / "tables" / "repair_study_outcomes.csv"
OUT_SUMMARY = REPO_ROOT / "results" / "tables" / "repair_study_summary.csv"
OUT_REPORT = REPO_ROOT / "results" / "repair_study_report.md"

MODEL = "qwen2.5:7b"
TEMPERATURE = 0.2
TASK_BY_NAME = {t.name: t for t in REAL_TASKS}

#: (arm, deployment monitor, grounded checkpoint + override + regeneration,
#:  ladder hint, coverage rung, oracle tau)
ARMS = (
    ("pre_fix", False, False, False, False, False),
    ("recalibrated", True, False, False, False, False),
    ("grounded", True, True, False, False, False),
    ("ladder", True, True, True, False, False),
    ("coverage", True, True, True, True, False),
    ("oracle", True, False, False, False, True),
)
#: Arms whose evidence may be pooled into a headline. `coverage` overlaps the
#: verifier and `oracle` is given ground truth, so neither is.
HEADLINE_ARMS = ("pre_fix", "recalibrated", "grounded", "ladder")
SUPPORTING_ARMS = ("coverage", "oracle")


def _load_steps(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()
            if line.strip()]


def load_repair_freeze() -> tuple[GroundingReference, RepairBudget, EffectPolicy]:
    artifact = json.loads(REPAIR_FREEZE.read_text("utf-8"))
    budget_record = dict(artifact["budget"])
    budget_record.pop("version", None)
    return (GroundingReference.from_record(artifact["grounding_reference"]),
            RepairBudget(**budget_record),
            EffectPolicy(effects=dict(artifact["effect_policy"])))


def run_episode(entry: dict, arm_spec: tuple, monitors: dict,
                budget: RepairBudget, effects: EffectPolicy) -> dict:
    (arm, deployment, grounded, use_ladder, allow_coverage,
     use_oracle) = arm_spec
    monitor, theta, reference = monitors["deployment" if deployment
                                         else "frozen"]
    task = TASK_BY_NAME[entry["task_name"]]
    registry = build_registry(task.tools)
    steps = _load_steps(CORPUS_DIR / entry["file"])

    if use_oracle:
        checkpoint = compute_checkpoint_oracle(steps, tau=entry["tau"])
    else:
        checkpoint = compute_checkpoint_primary(
            steps, monitor, theta, episode_from_trace,
            reference=reference if grounded else None,
            effects=effects, override=grounded)

    row = {"dataset": CORPUS_DIR.name, "episode_id": entry["episode_id"],
           "arm": arm, "task_name": entry["task_name"],
           "failure_class": entry["failure_class"], "tau": entry["tau"],
           "checkpoint_k": checkpoint.k, "checkpoint_rule": checkpoint.rule,
           "checkpoint_outcome": checkpoint.outcome,
           "alarm_step": checkpoint.alarm_step,
           "alarm_source": checkpoint.alarm_source,
           "n_regenerated": len(checkpoint.regenerate),
           "blocked_by_effects": len(checkpoint.blocked_by_effects),
           "rung": "resample", "hint_given": False}

    if checkpoint.outcome == NOT_TRIGGERED:
        row.update(outcome=NOT_TRIGGERED, success=None,
                   **ResourceLedger().as_record())
        return row

    rung, hint = "resample", None
    if use_ladder:
        episode = episode_from_trace(steps, entry["episode_id"],
                                     use_sentence_transformers=False,
                                     grounding=True)
        rung, hint = plan_repair(steps, episode.X, checkpoint, reference,
                                 required_tools=tuple(task.tools),
                                 allow_coverage=allow_coverage)
    row["rung"], row["hint_given"] = rung, hint is not None

    ledger = ResourceLedger()
    out = retry_real_task_from_checkpoint(
        steps, task.prompt, checkpoint, arm, model=MODEL,
        temperature=TEMPERATURE, registry=registry,
        backend_factory=_ollama_backend_factory, rung=rung, hint=hint,
        budget=budget, effects=effects, ledger=ledger)

    row.update(out.ledger.as_record())
    if out.outcome in (RECONSTRUCTION_FAILED, HALTED, BUDGET_EXHAUSTED):
        row.update(outcome=out.outcome,
                   success=False if out.outcome != RECONSTRUCTION_FAILED else None,
                   error=out.error)
        return row

    # Nothing in the repaired run may have repeated a world-changing call.
    assert_no_duplicate_effects(steps[:checkpoint.k], out.steps, effects)

    final_text = out.steps[-1].get("text", "") if out.steps else ""
    success = bool(task.verify(final_text, out.steps))
    row.update(outcome=(RECOVERED if success else STILL_WRONG), success=success)
    return row


def _ollama_backend_factory(**kwargs):
    from derail.experiments.collect_traces import OllamaBackend
    return OllamaBackend(**kwargs)


def _rate_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(ci.low), float(ci.high)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in df.groupby("arm", sort=False):
        triggered = group[group["outcome"] != NOT_TRIGGERED]
        recovered = int((group["outcome"] == RECOVERED).sum())
        end_to_end = _rate_ci(recovered, len(group))
        conditional = _rate_ci(recovered, len(triggered))
        rows.append({
            "arm": arm, "n_episodes": len(group),
            "n_triggered": len(triggered),
            "trigger_rate": len(triggered) / len(group) if len(group) else float("nan"),
            "n_recovered": recovered,
            "end_to_end_recovery": end_to_end[0],
            "end_to_end_ci_lo": end_to_end[1], "end_to_end_ci_hi": end_to_end[2],
            "conditional_recovery": conditional[0],
            "conditional_ci_lo": conditional[1],
            "conditional_ci_hi": conditional[2],
            "n_still_wrong": int((group["outcome"] == STILL_WRONG).sum()),
            "n_halted": int((group["outcome"] == HALTED).sum()),
            "n_budget_exhausted": int((group["outcome"] == BUDGET_EXHAUSTED).sum()),
            "n_reconstruction_failed": int(
                (group["outcome"] == RECONSTRUCTION_FAILED).sum()),
            "extra_model_calls_mean": float(group["repair_model_calls"].mean()),
            "extra_model_calls_p95": float(
                np.percentile(group["repair_model_calls"].dropna(), 95))
            if group["repair_model_calls"].notna().any() else float("nan"),
            "extra_tool_calls_mean": float(group["repair_tool_calls"].mean()),
            "recovery_seconds_mean": float(group["repair_seconds"].mean()),
            "recovery_seconds_p95": float(
                np.percentile(group["repair_seconds"].dropna(), 95))
            if group["repair_seconds"].notna().any() else float("nan"),
            "regenerated_calls_mean": float(group["n_regenerated"].mean()),
            "headline": arm in HEADLINE_ARMS,
            "n_first_attempt_recovered": int(
                ((group["outcome"] == RECOVERED) &
                 (group["repair_attempts"] <= 1)).sum()),
            "n_grounding_override_alarms": int(
                (group["alarm_source"] == "grounding_override").sum()),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_repair_study")
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset of arms to run")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    if args.report_only:
        outcomes = pd.read_csv(OUT_CSV)
        summary = summarize(outcomes)
        summary.to_csv(OUT_SUMMARY, index=False)
        write_report(outcomes, summary)
        print(f"[repair-study] wrote {OUT_REPORT}")
        return

    _ensure_tls()
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text("utf-8"))
    selected = [e for e in manifest if e["failure_class"] is not None]
    frozen_monitor, frozen_theta, _ = load_frozen_monitor()
    frozen_reference, budget, effects = load_repair_freeze()
    dep_monitor, dep_theta, dep_reference, dep_artifact = \
        load_deployment_monitor()
    monitors = {"frozen": (frozen_monitor, frozen_theta, frozen_reference),
                "deployment": (dep_monitor, dep_theta, dep_reference)}
    wanted = set(args.arms.split(",")) if args.arms else None
    arms = [a for a in ARMS if wanted is None or a[0] in wanted]
    print(f"[repair-study] {len(selected)} injected episodes x {len(arms)} arms")
    print(f"[repair-study] frozen theta={frozen_theta:.3f} | deployment "
          f"theta={dep_theta:.3f} (fit "
          f"{len(dep_artifact['fit_episode_ids'])} / calibrate "
          f"{len(dep_artifact['calibration_episode_ids'])}), grounding "
          f"trip={dep_reference.trip:.3f} alarm_trip="
          f"{dep_reference.alarm_trip:.3f}")

    rows = []
    for arm_spec in arms:
        for index, entry in enumerate(selected):
            started = time.perf_counter()
            row = run_episode(entry, arm_spec, monitors, budget, effects)
            rows.append(row)
            print(f"[repair-study] {arm_spec[0]:9s} "
                  f"({index + 1}/{len(selected)}) {entry['episode_id'][9:]}: "
                  f"{row['outcome']} k={row['checkpoint_k']} "
                  f"rung={row['rung']} regen={row['n_regenerated']} "
                  f"({time.perf_counter() - started:.1f}s)")

    outcomes = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(OUT_CSV, index=False)
    summary = summarize(outcomes)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(outcomes, summary)
    for path in (OUT_CSV, OUT_SUMMARY, OUT_REPORT):
        print(f"[repair-study] wrote {path}")


def write_report(outcomes: pd.DataFrame, summary: pd.DataFrame) -> None:
    budget = RepairBudget()
    lines = [
        "# What each repair fix is worth",
        "",
        "**SUPPORTING / diagnostic evidence, development data only.** The "
        "16-episode real-tool rollback corpus was spent when its interior was "
        "opened. The pre-fix study "
        "(`results/real_task_rollback_report.md`) stands unchanged; this is "
        "written beside it, not over it.",
        "",
        "A retry counts as repaired only when the task's own predeclared "
        "`success_fn` says the final answer is correct. It is called once, "
        "after the run, and never reaches the repair. Every arm is bounded by "
        f"the budget frozen in `results/repair_policy_freeze.json`: "
        f"{budget.max_attempts} attempts, {budget.max_extra_model_calls} "
        f"extra model calls, {budget.max_extra_tool_calls} extra tool calls, "
        f"{budget.retry_steps} continuation steps, "
        f"{budget.max_recovery_seconds:.0f}s.",
        "",
        "`coverage` and `oracle` are reported but never pooled into a "
        "headline -- the first overlaps what the verifier checks, the second "
        "is given ground truth.",
        "",
        "| arm | headline | triggered | recovered | end-to-end (95% CI) | "
        "conditional | still wrong | halted | budget out |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['arm']} | {'yes' if r['headline'] else 'no'} | "
            f"{int(r['n_triggered'])}/{int(r['n_episodes'])} | "
            f"{int(r['n_recovered'])} | {r['end_to_end_recovery']:.3f} "
            f"({r['end_to_end_ci_lo']:.3f}-{r['end_to_end_ci_hi']:.3f}) | "
            f"{r['conditional_recovery']:.3f} | {int(r['n_still_wrong'])} | "
            f"{int(r['n_halted'])} | {int(r['n_budget_exhausted'])} |")

    lines += ["", "## What recovery cost", "",
              "Resources are per episode, counted whether or not the repair "
              "worked. An unbounded retry loop reaches 100% on anything, so "
              "the ceiling is declared first and reported against.",
              "",
              "| arm | extra model calls (mean / p95) | extra tool calls | "
              "regenerated lookups | recovery seconds (mean / p95) |",
              "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['arm']} | {r['extra_model_calls_mean']:.1f} / "
            f"{r['extra_model_calls_p95']:.0f} | "
            f"{r['extra_tool_calls_mean']:.1f} | "
            f"{r['regenerated_calls_mean']:.1f} | "
            f"{r['recovery_seconds_mean']:.1f} / "
            f"{r['recovery_seconds_p95']:.1f} |")

    lines += ["", "## Per failure class", "",
              "| arm | " + " | ".join(sorted(outcomes["failure_class"].unique()))
              + " |",
              "|---|" + "---|" * outcomes["failure_class"].nunique()]
    for arm, group in outcomes.groupby("arm", sort=False):
        cells = []
        for failure_class in sorted(outcomes["failure_class"].unique()):
            sub = group[group["failure_class"] == failure_class]
            cells.append(f"{int((sub['outcome'] == RECOVERED).sum())}/{len(sub)}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += ["", "## Which rung each alarm selected", "",
              "| arm | " + " | ".join(sorted(outcomes["rung"].unique())) + " |",
              "|---|" + "---|" * outcomes["rung"].nunique()]
    for arm, group in outcomes.groupby("arm", sort=False):
        counts = group["rung"].value_counts()
        lines.append(f"| {arm} | " + " | ".join(
            str(int(counts.get(rung, 0)))
            for rung in sorted(outcomes["rung"].unique())) + " |")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    main()
