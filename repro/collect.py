"""Turn out/runs/* into the two comparison tables and the report's numbers.

Nothing here re-computes a metric from scratch: the metrics come from the
upstream ``results.json`` that each arm wrote, and the "paper" side comes from
the artifacts committed in the upstream tree (``results/tables/*.csv``) which
this harness never overwrites. Collection only joins, aggregates and diffs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro import config as cfg

PAPER_H1 = cfg.UPSTREAM_DIR / "results" / "tables" / "h1_main.csv"
PAPER_MULTI = cfg.UPSTREAM_DIR / "results" / "tables" / "multiseed_summary.csv"

# Same monitor set and aggregation rule as upstream derail.experiments.run_multiseed.
MULTI_MONITORS = ("esn_cusum_max", "esn_cusum", "esn_full", "esn_single",
                  "gru", "lstm", "tcn", "linear_ar", "delta_mahalanobis",
                  "mahalanobis", "self_drift")
METRICS = ("detection_rate", "mean_lead_all", "episode_auc", "healthy_fa_rate")


def load_run(name: str) -> dict | None:
    p = cfg.RUNS_DIR / name / "results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def per_monitor(run: dict) -> dict:
    return run["h1"]["per_monitor"]


def multiseed_from_runs() -> pd.DataFrame:
    """mean / sample-std across the five published seeds, same rule as upstream."""
    rows = []
    runs = {}
    for seed in cfg.MULTISEED_SEEDS:
        name = "baseline_master" if seed == cfg.MASTER_SEED else f"multiseed_s{seed}"
        r = load_run(name)
        if r is None:
            print(f"[collect] missing run {name}; skipping it in the 5-seed table")
            continue
        runs[seed] = r
    if not runs:
        return pd.DataFrame()

    for mon in MULTI_MONITORS:
        row: dict = {"monitor": mon, "n_seeds": len(runs)}
        ok = True
        for metric in METRICS:
            vals = []
            for seed, r in runs.items():
                pm = per_monitor(r)
                if mon not in pm or metric not in pm[mon]:
                    ok = False
                    break
                vals.append(float(pm[mon][metric]))
            if not ok:
                break
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def headline_frame(df: pd.DataFrame, monitor: str = "esn_cusum_max") -> dict:
    r = df[df["monitor"] == monitor]
    if r.empty:
        return {}
    return r.iloc[0].to_dict()


def fmt_pm(mean: float, std: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def main() -> None:
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = cfg.REPO_ROOT / "report"
    report_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- table 1a
    mine_multi = multiseed_from_runs()
    paper_multi = pd.read_csv(PAPER_MULTI) if PAPER_MULTI.exists() else pd.DataFrame()

    t1 = mine_multi.merge(paper_multi, on="monitor", how="outer",
                          suffixes=("_mine", "_paper"))
    t1.to_csv(cfg.TABLES_DIR / "t1_multiseed_mine_vs_paper.csv", index=False)

    # ---------------------------------------------------------------- table 1b
    mine_h1 = pd.read_csv(cfg.RUNS_DIR / "baseline_master" / "h1_main.csv")
    paper_h1 = pd.read_csv(PAPER_H1)
    merged = mine_h1.merge(paper_h1, on="monitor", suffixes=("_mine", "_paper"))
    numeric = [c for c in mine_h1.columns if c != "monitor"]
    diffs = {}
    for c in numeric:
        a, b = merged.get(f"{c}_mine"), merged.get(f"{c}_paper")
        if a is None or b is None:
            continue
        diffs[c] = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
    t1b = pd.DataFrame([{"column": k, "max_abs_diff": v} for k, v in diffs.items()])
    t1b.to_csv(cfg.TABLES_DIR / "t1b_master_seed_column_diffs.csv", index=False)

    # "Exact" is claimed for the columns the table reports. theta is a float
    # quantile of the healthy validation scores: it is allowed to differ at
    # floating-point level (it did, by ~1e-11) as long as no alarm flips, which
    # is visible because detection / FA / lead / AUC are all exactly 0.
    reported = [c for c in ("detection_rate", "mean_lead_all", "episode_auc",
                            "healthy_fa_rate", "early_alarm_rate") if c in diffs]
    exact_reported = bool(reported) and all(diffs[c] == 0.0 for c in reported)
    worst_col = max(diffs, key=diffs.get) if diffs else None
    worst_val = diffs[worst_col] if worst_col else float("nan")

    def _fmt_diff(v: float) -> str:
        if v == 0.0:
            return "0"
        return f"$%.1e$" % v if v >= 1e-3 else f"${v:.1e}$".replace("e-0", "e-")

    # LaTeX table for the report, generated from the diff above, never hand-written.
    headline_cols = ["detection_rate", "mean_lead_all", "episode_auc",
                     "healthy_fa_rate", "theta"]
    lines = ["% Auto-generated by repro/collect.py -- do not edit by hand.",
             "\\begin{tabular}{@{}lr@{}}", "\\toprule",
             "Column of \\texttt{h1\\_main.csv} & max $|$ours $-$ committed$|$ \\\\",
             "\\midrule"]
    for c in headline_cols:
        if c in diffs:
            lines.append("\\texttt{%s} & %s \\\\" % (c.replace("_", "\\_"), _fmt_diff(diffs[c])))
    others = [k for k in diffs if k not in headline_cols]
    if others:
        worst_other = max(diffs[k] for k in others)
        lines.append("remaining %d numeric columns & %s \\\\"
                     % (len(others), _fmt_diff(worst_other)))
    if diffs:
        lines.append("\\midrule")
        lines.append("all %d numeric columns, %d monitor rows & %s \\\\"
                     % (len(diffs), len(merged), _fmt_diff(worst_val)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (report_dir / "tab_master_diff.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- table 2
    rows = []
    for seed in cfg.PAIRED_SEEDS:
        base = load_run(f"baseline_s{seed}")
        mean = load_run(f"mean_s{seed}")
        top2 = load_run(f"top2_s{seed}")
        if not (base and mean):
            continue
        for label, run in (("baseline(max)", base), ("mean", mean), ("top2mean", top2)):
            if run is None:
                continue
            pm = per_monitor(run)["esn_cusum_max"]
            rows.append({"seed": seed, "arm": label,
                         "detection_rate": pm["detection_rate"],
                         "mean_lead_all": pm["mean_lead_all"],
                         "episode_auc": pm["episode_auc"],
                         "healthy_fa_rate": pm["healthy_fa_rate"]})
    t2 = pd.DataFrame(rows)
    if not t2.empty:
        t2.to_csv(cfg.TABLES_DIR / "t2_fusion_ablation.csv", index=False)

    # control rows: monitors that the patch must NOT touch
    ctrl_rows = []
    for seed in cfg.PAIRED_SEEDS:
        base = load_run(f"baseline_s{seed}")
        mean = load_run(f"mean_s{seed}")
        if not (base and mean):
            continue
        bpm, mpm = per_monitor(base), per_monitor(mean)
        for mon in bpm:
            if mon == "esn_cusum_max" or mon not in mpm:
                continue
            same = all(abs(float(bpm[mon][k]) - float(mpm[mon][k])) < 1e-12
                       for k in ("detection_rate", "mean_lead_all", "episode_auc"))
            ctrl_rows.append({"seed": seed, "monitor": mon, "identical": same})
    t2c = pd.DataFrame(ctrl_rows)
    if not t2c.empty:
        t2c.to_csv(cfg.TABLES_DIR / "t2b_controls_identical.csv", index=False)

    # ---------------------------------------------------------------- table 3
    rows = []
    base_master = load_run("baseline_master")
    for label, arm in (("budget=2%", "budget02_master"), ("budget=5% (base)", "baseline_master"),
                       ("budget=10%", "budget10_master")):
        run = load_run(arm)
        if run is None:
            continue
        pm = per_monitor(run)["esn_cusum_max"]
        rows.append({"budget": label,
                     "detection_rate": pm["detection_rate"],
                     "mean_lead_all": pm["mean_lead_all"],
                     "episode_auc": pm["episode_auc"],
                     "realised_fa": pm["healthy_fa_rate"]})
    t3 = pd.DataFrame(rows)
    if not t3.empty:
        t3.to_csv(cfg.TABLES_DIR / "t3_fa_budget.csv", index=False)

    # ---------------------------------------------------------------- numbers
    env = json.loads((cfg.OUT_DIR / "environment.json").read_text(encoding="utf-8"))
    manifest_path = cfg.RUNS_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:  # a copied artifact tree may carry the arm dirs without the manifest
        manifest = []
        for d in sorted(cfg.RUNS_DIR.iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                manifest.append(json.loads(meta.read_text(encoding="utf-8")))
    durations = [m["duration_s"] for m in manifest
                 if m.get("returncode") == 0 and m.get("duration_s")]

    macros: dict[str, str] = {}
    macros["PyVer"] = env["python"]
    macros["NumpyVer"] = str(env["packages"]["numpy"])
    macros["ScipyVer"] = str(env["packages"]["scipy"])
    macros["PandasVer"] = str(env["packages"]["pandas"])
    macros["SklearnVer"] = str(env["packages"]["scikit-learn"])
    macros["UpstreamHead"] = (env["upstream_head"] or "unknown")[:7]
    macros["CpuCount"] = str(env["cpu_count"])
    macros["NArms"] = str(len(manifest))
    macros["MeanRuntime"] = f"{np.mean(durations):.0f}" if durations else "n/a"

    def put(name: str, value: str) -> None:
        macros[name] = value

    h_mine = headline_frame(mine_multi)
    h_paper = headline_frame(paper_multi) if not paper_multi.empty else {}
    if h_mine and h_paper:
        put("MyDet", fmt_pm(h_mine["detection_rate_mean"], h_mine["detection_rate_std"]))
        put("MyLead", fmt_pm(h_mine["mean_lead_all_mean"], h_mine["mean_lead_all_std"], 2))
        put("MyAuc", fmt_pm(h_mine["episode_auc_mean"], h_mine["episode_auc_std"]))
        put("MyFa", fmt_pm(h_mine["healthy_fa_rate_mean"], h_mine["healthy_fa_rate_std"]))
        put("PaperDet", fmt_pm(h_paper["detection_rate_mean"], h_paper["detection_rate_std"]))
        put("PaperLead", fmt_pm(h_paper["mean_lead_all_mean"], h_paper["mean_lead_all_std"], 2))
        put("PaperAuc", fmt_pm(h_paper["episode_auc_mean"], h_paper["episode_auc_std"]))
        put("PaperFa", fmt_pm(h_paper["healthy_fa_rate_mean"], h_paper["healthy_fa_rate_std"]))
        put("DetDiff", f"{h_mine['detection_rate_mean'] - h_paper['detection_rate_mean']:+.4f}")
        put("AucDiff", f"{h_mine['episode_auc_mean'] - h_paper['episode_auc_mean']:+.4f}")
        put("LeadDiff", f"{h_mine['mean_lead_all_mean'] - h_paper['mean_lead_all_mean']:+.3f}")

    put("MasterExact", "exact for every reported metric" if exact_reported else "NOT exact")
    if diffs:
        put("MasterMaxDiff", f"{min(diffs.values()):.1e}" if max(diffs.values()) == 0
            else f"{worst_val:.1e}")
        put("WorstColumn", (worst_col or "n/a").replace("_", "\\_"))
        put("WorstDiff", f"{worst_val:.1e}")

    if not t2.empty:
        pivot = t2.pivot_table(index="arm", values="detection_rate", aggfunc="mean")
        if {"baseline(max)", "mean"} <= set(pivot.index):
            b = float(pivot.loc["baseline(max)", "detection_rate"])
            m = float(pivot.loc["mean", "detection_rate"])
            put("ModBaseDet", f"{b:.3f}")
            put("ModMeanDet", f"{m:.3f}")
            put("ModDeltaDet", f"{m - b:+.3f}")
            put("ModRelDet", f"{(m - b) / b * 100:+.1f}\\%")
        if "top2mean" in pivot.index:
            put("ModTopTwoDet", f"{float(pivot.loc['top2mean', 'detection_rate']):.3f}")
        pa = t2.pivot_table(index="arm", values="episode_auc", aggfunc="mean")
        if {"baseline(max)", "mean"} <= set(pa.index):
            put("ModBaseAuc", f"{float(pa.loc['baseline(max)', 'episode_auc']):.3f}")
            put("ModMeanAuc", f"{float(pa.loc['mean', 'episode_auc']):.3f}")
            put("ModDeltaAuc", f"{float(pa.loc['mean', 'episode_auc']) - float(pa.loc['baseline(max)', 'episode_auc']):+.3f}")
        pl = t2.pivot_table(index="arm", values="mean_lead_all", aggfunc="mean")
        if {"baseline(max)", "mean"} <= set(pl.index):
            put("ModBaseLead", f"{float(pl.loc['baseline(max)', 'mean_lead_all']):.2f}")
            put("ModMeanLead", f"{float(pl.loc['mean', 'mean_lead_all']):.2f}")
            put("ModDeltaLead", f"{float(pl.loc['mean', 'mean_lead_all']) - float(pl.loc['baseline(max)', 'mean_lead_all']):+.2f}")
        pf = t2.pivot_table(index="arm", values="healthy_fa_rate", aggfunc="mean")
        if {"baseline(max)", "mean"} <= set(pf.index):
            put("ModBaseFa", f"{float(pf.loc['baseline(max)', 'healthy_fa_rate']):.3f}")
            put("ModMeanFa", f"{float(pf.loc['mean', 'healthy_fa_rate']):.3f}")
            put("ModDeltaFa", f"{float(pf.loc['mean', 'healthy_fa_rate']) - float(pf.loc['baseline(max)', 'healthy_fa_rate']):+.3f}")
        if "top2mean" in pl.index:
            put("ModTopTwoLead", f"{float(pl.loc['top2mean', 'mean_lead_all']):.2f}")
        if "top2mean" in pa.index:
            put("ModTopTwoAuc", f"{float(pa.loc['top2mean', 'episode_auc']):.3f}")

    if not t2c.empty:
        put("CtrlRows", str(len(t2c)))
        put("CtrlIdentical", str(int(t2c["identical"].sum())))

    if not t3.empty:
        for _, r in t3.iterrows():
            key = r["budget"].split("=")[1].rstrip("%").replace(" (base)", "")
            tag = {"2": "BTwo", "5": "BFive", "10": "BTen"}.get(key.strip(), None)
            if tag:
                put(f"{tag}Det", f"{r['detection_rate']:.3f}")
                put(f"{tag}Fa", f"{r['realised_fa']:.3f}")
                put(f"{tag}Auc", f"{r['episode_auc']:.3f}")

    # paper's own headline, transcribed from the committed artifact, for the
    # report's target-claim box even if our five seeds are incomplete.
    if not paper_multi.empty:
        p = headline_frame(paper_multi)
        put("PaperDetRaw", f"{p['detection_rate_mean']:.4f}")
        put("PaperAucRaw", f"{p['episode_auc_mean']:.4f}")
        put("PaperLeadRaw", f"{p['mean_lead_all_mean']:.3f}")

    lines = ["% Auto-generated by repro/collect.py -- do not edit by hand.", ""]
    for k, v in sorted(macros.items()):
        if v is None:
            v = "n/a"
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (report_dir / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {cfg.rel(cfg.TABLES_DIR)}/*.csv and report/numbers.tex")
    for k in ("MyDet", "PaperDet", "MasterExact", "ModBaseDet", "ModMeanDet", "ModDeltaDet",
              "CtrlIdentical", "CtrlRows"):
        if k in macros:
            print(f"  {k} = {macros[k]}")


if __name__ == "__main__":
    main()
