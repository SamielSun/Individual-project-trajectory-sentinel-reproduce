"""Render the report's figures from out/tables (matplotlib, Agg, no GUI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from repro import config as cfg  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "report" / "figures"

PRIMARY = "esn_cusum_max"


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "figure.dpi": 150,
    })


def fig_headline() -> Path | None:
    t1 = cfg.TABLES_DIR / "t1_multiseed_mine_vs_paper.csv"
    if not t1.exists():
        print("[figures] no t1 table; skipping fig1")
        return None
    df = pd.read_csv(t1)
    row = df[df["monitor"] == PRIMARY]
    if row.empty:
        print("[figures] primary monitor missing from t1; skipping fig1")
        return None
    r = row.iloc[0]

    panels = [
        ("detection rate", "detection_rate_mean", "detection_rate_std", 1.0),
        ("steps saved", "mean_lead_all_mean", "mean_lead_all_std", 1.0),
        ("episode AUC", "episode_auc_mean", "episode_auc_std", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (title, mkey, skey, _) in zip(axes, panels):
        mine = float(r[f"{mkey}_mine"])
        mine_sd = float(r[f"{skey}_mine"])
        paper = float(r[f"{mkey}_paper"]) if f"{mkey}_paper" in r else float("nan")
        paper_sd = float(r[f"{skey}_paper"]) if f"{skey}_paper" in r else 0.0
        ax.bar([0, 1], [paper, mine], yerr=[paper_sd, mine_sd], capsize=3,
               color=["#b8c4d9", "#2b6cb0"], width=0.62)
        ax.set_xticks([0, 1], ["paper", "ours"])
        ax.set_title(title, fontsize=9)
        for x, v, sd in ((0, paper, paper_sd), (1, mine, mine_sd)):
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8)
    axes[0].set_ylim(0, 1.0)
    fig.suptitle(f"{PRIMARY}: five seeds, ours vs committed artifact (mean $\\pm$ 1 sd)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig1_headline.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {cfg.rel(out)}")
    return out


def fig_modification() -> Path | None:
    t2 = cfg.TABLES_DIR / "t2_fusion_ablation.csv"
    if not t2.exists():
        print("[figures] no t2 table; skipping fig2")
        return None
    df = pd.read_csv(t2)
    order = ["baseline(max)", "top2mean", "mean"]
    order = [a for a in order if a in set(df["arm"])]
    colors = {"baseline(max)": "#2b6cb0", "top2mean": "#63a2d4", "mean": "#c05621"}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, metric, title in zip(axes,
                                 ("detection_rate", "episode_auc"),
                                 ("detection rate", "episode AUC")):
        g = df.groupby("arm")[metric]
        means = g.mean().reindex(order)
        sds = g.std(ddof=1).reindex(order).fillna(0.0)
        ax.bar(range(len(order)), means.to_numpy(), yerr=sds.to_numpy(), capsize=3,
               color=[colors[a] for a in order], width=0.6)
        ax.set_xticks(range(len(order)), [a.replace("(", "\n(") for a in order], fontsize=8)
        ax.set_title(title, fontsize=9)
        for i, v in enumerate(means.to_numpy()):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8)
    axes[0].set_ylim(0, 1.0)
    fig.suptitle("Fusion ablation, paired over two unseen seeds (mean $\\pm$ range/sd)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig2_modification.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {cfg.rel(out)}")
    return out


def main() -> None:
    _style()
    fig_headline()
    fig_modification()


if __name__ == "__main__":
    main()
