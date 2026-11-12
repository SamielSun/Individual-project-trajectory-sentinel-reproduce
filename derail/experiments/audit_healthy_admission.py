"""What actually got into the healthy null every threshold is derived from?

A calibration population has to be admitted by a rule fixed in advance, and
the rule has to be reported with its numbers: how many runs were attempted,
how many admitted, how many rejected and why, and what tasks and shapes the
survivors cover. Otherwise a threshold is a statement about a hand-shaped
sample rather than about the deployment.

Two distinctions this module keeps apart, because collapsing them is what
produced a threshold set by a fold that was more than half zeros:

  verified healthy   the run happened with no injection AND the task's own
                     predeclared verifier says it finished the task. A run
                     that simply failed is not a healthy negative; it is an
                     unlabelled failure, and admitting it raises the bar for
                     every real one.
  scoreable          the detector had at least one valid scoring opportunity
                     on it. A trajectory shorter than the washout produces no
                     score at all, so it can never alarm; counting it as a
                     silent healthy episode is an artificial zero in the
                     threshold's own distribution and a free pass in the
                     false-alarm denominator.

Neither is a licence to drop episodes. Everything stays in the corpus and is
counted here; scoreability is recorded against the detector version that
decides it, so a detector that later handles short trajectories makes those
same episodes scoreable under the unchanged rule.

Writes:
  results/tables/healthy_admission_audit.csv
  results/healthy_admission_audit.md
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from derail.harness.tasks import REAL_TASKS
from derail.monitor.esn import _WASHOUT

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_CSV = REPO_ROOT / "results" / "tables" / "healthy_admission_audit.csv"
OUT_REPORT = REPO_ROOT / "results" / "healthy_admission_audit.md"

CORPORA = ("langgraph7b_real2", "autogen7b_real2")
TASK_BY_NAME = {t.name: t for t in REAL_TASKS}

#: Which detector decides scoreability. Recorded per row, because the answer
#: is a property of the detector and not of the episode: the same trajectory
#: becomes scoreable under a detector that needs fewer steps to produce a
#: score, with no change to the admission rule.
DETECTOR_VERSION = f"esn_cusum_max/washout={_WASHOUT}"

#: Identifiers a task verifier must not touch. It may read the recorded run
#: and nothing else -- not the monitor it is used to evaluate, not the label
#: the injector requested, not a threshold.
FORBIDDEN_IN_VERIFIER = ("monitor", "score", "theta", "alarm", "failure_class",
                         "tau", "cusum", "detect")


def verifier_provenance() -> dict:
    """Evidence that each task verifier is safe to admit episodes with.

    Predeclared: it is a committed attribute of the task, defined with the
    task itself rather than chosen once results are in. Independent: its
    source mentions nothing about the detector, the threshold, or the
    injector's label, and its only inputs are the final response text and the
    recorded steps.
    """
    out = {}
    for name in sorted({e["task"] for corpus in CORPORA
                        for e in json.loads(
                            (TRACES_ROOT / corpus / "manifest.json")
                            .read_text("utf-8"))
                        if e["failure_class"] is None}):
        task = TASK_BY_NAME[name]
        src = inspect.getsource(task.success_fn)
        out[name] = {
            "declared_in": "derail/harness/tasks.py",
            "signature": str(inspect.signature(task.success_fn)),
            "source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest()[:16],
            "mentions_forbidden": sorted(
                token for token in FORBIDDEN_IN_VERIFIER if token in src),
            "allowlisted_tools": list(task.tools),
        }
    return out


def _final_response(steps: list[dict]) -> str:
    """The last step's own prose -- what a verifier reads as the answer."""
    for step in reversed(steps):
        text = str(step.get("text", "") or "")
        if text.strip():
            return text
    return ""


def classify_episode(entry: dict, steps: list[dict]) -> dict:
    task = TASK_BY_NAME[entry["task"]]
    response = _final_response(steps)
    verdict = task.verify(response, steps)
    repeat = task.verify(response, steps)
    T = len(steps)
    opportunities = max(0, T - _WASHOUT)

    if verdict is True:
        status, rejection = "verified_healthy", ""
    elif verdict is False:
        status = "rejected_healthy_attempt"
        rejection = ("task verifier returned False: a run that did not finish "
                     "its task is an unlabelled failure, not a healthy negative")
    else:
        status = "rejected_healthy_attempt"
        rejection = "verifier could not decide (raised); quality unknown"

    if opportunities > 0:
        unscoreable = ""
    else:
        unscoreable = (f"T={T} <= washout {_WASHOUT}: the detector emits no "
                       f"score on any step, so it can neither alarm nor stay "
                       f"silent informatively")

    return {
        "dataset": entry["_corpus"], "episode_id": entry["episode_id"],
        "task": entry["task"], "model": entry["model"], "T": T,
        "admitted_because": entry.get("accepted_because"),
        "success_recorded_at_collection": entry.get("success"),
        "verifier_result": verdict,
        "verifier_deterministic": verdict == repeat,
        "admission_status": status,
        "rejection_reason": rejection,
        "detector_version": DETECTOR_VERSION,
        "scoring_opportunities": opportunities,
        "scoreable": opportunities > 0,
        "unscoreable_reason": unscoreable,
    }


def audit_corpus(corpus: str) -> tuple[list[dict], dict]:
    corpus_dir = TRACES_ROOT / corpus
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    rejected_path = corpus_dir / "landing_failures.json"
    logged_rejections = (json.loads(rejected_path.read_text("utf-8"))
                         if rejected_path.exists() else [])

    rows = []
    for entry in manifest:
        if entry["failure_class"] is not None:
            continue
        steps = [json.loads(line) for line in
                 (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                 if line.strip()]
        rows.append(classify_episode({**entry, "_corpus": corpus}, steps))

    verified = [r for r in rows if r["admission_status"] == "verified_healthy"]
    scoreable = [r for r in verified if r["scoreable"]]
    return rows, {
        "corpus": corpus,
        "attempted": len(rows),
        "verified_healthy": len(verified),
        "rejected_healthy_attempts": len(rows) - len(verified),
        "rejection_reasons": Counter(r["rejection_reason"] for r in rows
                                     if r["rejection_reason"]),
        "scoreable_verified_healthy": len(scoreable),
        "monitoring_coverage": (len(scoreable) / len(verified)
                                if verified else float("nan")),
        "unscoreable_verified": len(verified) - len(scoreable),
        "collection_rejections": Counter(r.get("reason")
                                         for r in logged_rejections),
    }


def main() -> None:
    all_rows, summaries = [], []
    for corpus in CORPORA:
        rows, summary = audit_corpus(corpus)
        all_rows += rows
        summaries.append(summary)

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    provenance = verifier_provenance()

    lines = ["# What is in the healthy null",
             "",
             "**SUPPORTING / diagnostic evidence.** Nothing here deletes, "
             "relabels or hides an episode: every admitted run stays in the "
             "corpus and is counted below. It reports the composition of the "
             "healthy populations every per-deployment threshold is derived "
             "from, by re-running each task's own predeclared `success_fn` "
             "over the committed traces.",
             "",
             "Collection admitted healthy runs by an outcome-independent rule "
             "(`collect_framework_real_traces.check_admission`), which cannot "
             "silently drop difficult-but-valid behaviour but also never "
             "consults the verifier -- so the corpus records `success = None` "
             "for every healthy episode.",
             "",
             "## The two distinctions", "",
             "**Verified healthy** means no injection AND the task's own "
             "verifier says the run finished the task. A run that simply "
             "failed is an unlabelled failure, not a healthy negative.",
             "",
             "**Scoreable** means the detector had at least one valid scoring "
             "opportunity. It is a property of the detector, recorded per row "
             f"as `{DETECTOR_VERSION}`: a detector that produces a score "
             "sooner makes the same episodes scoreable with no change to the "
             "admission rule. Unscoreable episodes must not enter a threshold "
             "distribution as artificial zeros, and must not pad a "
             "false-alarm denominator with runs that could never alarm.",
             "",
             "**Monitoring coverage** = scoreable verified-healthy / all "
             "verified-healthy. Reported because it is the fraction of "
             "healthy traffic the detector can say anything about at all.",
             ""]

    for summary in summaries:
        sub = df[df["dataset"] == summary["corpus"]]
        lines += [f"## {summary['corpus']}", "",
                  f"- attempted: **{summary['attempted']}**",
                  f"- verified healthy: **{summary['verified_healthy']}**",
                  f"- rejected healthy attempts: "
                  f"**{summary['rejected_healthy_attempts']}** (retained in "
                  f"the corpus, counted here, never used as negatives)",
                  f"- scoreable verified healthy: "
                  f"**{summary['scoreable_verified_healthy']}**",
                  f"- unscoreable verified healthy: "
                  f"**{summary['unscoreable_verified']}**",
                  f"- **monitoring coverage: "
                  f"{summary['monitoring_coverage']:.3f}**",
                  f"- task mix: " + ", ".join(
                      f"{k} {v}" for k, v in sorted(Counter(sub["task"]).items())),
                  f"- model: " + ", ".join(sorted(set(sub["model"]))),
                  f"- trajectory length: min {sub['T'].min()}, median "
                  f"{int(sub['T'].median())}, max {sub['T'].max()}",
                  ""]
        if summary["rejection_reasons"]:
            lines.append("Rejection reasons:")
            lines += [f"- {n} x {reason}" for reason, n
                      in summary["rejection_reasons"].most_common()]
            lines.append("")

    lines += ["## False-alarm denominators these numbers imply", "",
              "Both are reported for every future evaluation, because they "
              "diverge and each answers a different question: operational FA "
              "over all verified-healthy episodes is what a deployment sees; "
              "scoreable FA over those the detector could act on is what the "
              "detector is responsible for.",
              "",
              "| corpus | verified healthy (operational denominator) | "
              "scoreable (scoreable denominator) | monitoring coverage |",
              "|---|---|---|---|"]
    for summary in summaries:
        lines.append(
            f"| {summary['corpus']} | {summary['verified_healthy']} | "
            f"{summary['scoreable_verified_healthy']} | "
            f"{summary['monitoring_coverage']:.3f} |")

    lines += ["", "## Is the verifier safe to admit episodes with?", "",
              "| task | declared in | signature | source sha | mentions "
              "detector/label terms | tools |", "|---|---|---|---|---|---|"]
    for name, info in provenance.items():
        lines.append(
            f"| {name} | `{info['declared_in']}` | `{info['signature']}` | "
            f"`{info['source_sha256']}` | "
            f"{', '.join(info['mentions_forbidden']) or 'none'} | "
            f"{', '.join(info['allowlisted_tools'])} |")
    lines += ["",
              f"Determinism was re-checked by evaluating every episode twice: "
              f"{int(df['verifier_deterministic'].sum())} of {len(df)} agree. "
              f"The verifier receives only the final response text and the "
              f"recorded steps, and a recorded step carries no failure class, "
              f"no onset and no monitor output -- so it cannot see the label "
              f"or the detector it is used to evaluate.",
              ""]
    OUT_REPORT.write_text("\n".join(lines), "utf-8")

    print(df.groupby(["dataset", "admission_status", "scoreable"])
          .size().to_string())
    for path in (OUT_CSV, OUT_REPORT):
        print(f"[healthy-admission] wrote {path}")


if __name__ == "__main__":
    main()
