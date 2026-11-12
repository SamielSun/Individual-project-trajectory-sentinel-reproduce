"""Single source of truth for paths, pins and run definitions."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _upstream_root() -> Path:
    """Where the upstream tree lives.

    Two layouts are supported on purpose. In this fork the upstream sources are
    the repository itself (``derail/`` sits at the repo root). The overlay
    layout keeps a pristine checkout under ``vendor/upstream`` instead, which is
    what ``setup.ps1`` creates when the fork's sources are absent.
    """
    if (REPO_ROOT / "derail" / "experiments" / "run_experiment.py").exists():
        return REPO_ROOT
    return REPO_ROOT / "vendor" / "upstream"


UPSTREAM_DIR = _upstream_root()
OUT_DIR = REPO_ROOT / "out"
RUNS_DIR = OUT_DIR / "runs"
TABLES_DIR = OUT_DIR / "tables"

UPSTREAM_URL = "https://github.com/sunnydubey1111/agent-trajectory-sentinel.git"

# The upstream commit this fork is based on: the tree whose committed artifacts
# the paper's tables are computed from. Recorded in out/environment.json so a
# report table can never silently mix two versions of the upstream code.
#
# NOTE: the arXiv v1 text describes an *earlier* tree, commit 00c0673, with
# 2,823 episodes across 25 corpora. The current tree carries 4,022 across 35.
# That divergence is a finding in the report, not a pin we hide.
PINNED_COMMIT = "1b3e07f"
ARXIV_V1_COMMIT = "00c0673"

MASTER_SEED = 20260713
MULTISEED_SEEDS = (20260713, 7, 101, 202, 303)

# Disposable seeds for the paired baseline/modification comparison: never used
# by the published study, so a modified run cannot collide with a published
# artifact even if it wrote to results/ (it does not: runs are redirected).
PAIRED_SEEDS = (424242, 31337)

# The monitors the paper's headline table reports, in table order.
PAPER_MONITORS = (
    "esn_cusum_max",
    "delta_mahalanobis",
    "gru",
    "lstm",
    "tcn",
    "linear_ar",
)

# Metrics that make up the headline row (README / multiseed_summary.csv).
HEADLINE_METRICS = ("detection_rate", "mean_lead_all", "episode_auc")


def upstream_python() -> str:
    """Interpreter used for the runs.

    Order: an explicit ``REPRO_PYTHON`` override, then the venv ``setup.ps1``
    builds (``.venv`` at the upstream root), then the interpreter running this
    harness — which is what happens in a checkout that was never bootstrapped.
    """
    override = os.environ.get("REPRO_PYTHON")
    if override and Path(override).exists():
        return override
    win = UPSTREAM_DIR / ".venv" / "Scripts" / "python.exe"
    if win.exists():
        return str(win)
    posix = UPSTREAM_DIR / ".venv" / "bin" / "python"
    if posix.exists():
        return str(posix)
    import sys

    return sys.executable


def require_upstream() -> Path:
    if not (UPSTREAM_DIR / "derail" / "experiments" / "run_experiment.py").exists():
        raise SystemExit(
            f"upstream tree not found under {UPSTREAM_DIR}.\n"
            "In this fork the sources should be at the repository root; if they\n"
            "are missing, run:  pwsh -File setup.ps1"
        )
    return UPSTREAM_DIR


def run_dir(name: str) -> Path:
    p = RUNS_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def rel(path: Path) -> str:
    """Repo-relative path for logs; absolute is fine on any failure path."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(path)
