"""Build Table 3 (baseline vs mitigation methods) from the saved evaluation files.
Needs only matplotlib. Run from the repo root after tasks 1, 2 and 4 (several seeds):

    python -m make_table3                  # rho 0.99
    python -m make_table3 --rho 0.9

Reads  outputs/results/{method}_eval{tag}_seed{S}.json   (method = baseline, reweighting, group_dro, ...)
Writes reports/tables/table3{tag}.md, reports/figures/table3_comparison{tag}.png,
       outputs/results/table3{tag}.json
Std is the sample standard deviation over seeds (n - 1). Worst group = lowest accuracy among
the four colour/digit groups on OOD-1, where all groups are about equally large.
"""
import argparse
import glob
import json
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ["baseline", "reweighting", "group_dro", "lff", "irm", "augmentation"]
LABELS = {"baseline": "Baseline", "reweighting": "Reweighting", "group_dro": "Group DRO",
          "lff": "LfF", "irm": "IRM", "augmentation": "Augmentation"}


def metrics(res):
    wg = min(res["test_ood1"]["group_acc"].values())
    id_, o1, o2 = (100 * res[k]["acc"] for k in ("test_id", "test_ood1", "test_ood2"))
    return {"ID": id_, "OOD-1": o1, "OOD-2": o2, "gap_ID_OOD1": id_ - o1, "gap_ID_OOD2": id_ - o2,
            "worst_group_OOD1": 100 * wg}


def mean_std(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(vals):
    m, s = mean_std(vals)
    return f"{m:.2f} \u00b1 {s:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--results-dir", default="./outputs/results")
    ap.add_argument("--table-dir", default="./reports/tables")
    ap.add_argument("--fig-dir", default="./reports/figures")
    a = ap.parse_args()
    tag = "" if abs(a.rho - 0.9) < 1e-9 else f"_rho{a.rho:g}"

    runs = {}  # method -> {seed: metrics}
    for m in METHODS:
        for path in glob.glob(os.path.join(a.results_dir, f"{m}_eval{tag}_seed*.json")):
            seed = int(re.search(r"_seed(\d+)\.json$", path).group(1))
            with open(path) as f:
                runs.setdefault(m, {})[seed] = metrics(json.load(f))
    if "baseline" not in runs:
        raise SystemExit("no baseline evaluation files found; run task1/task2 first")
    methods = [m for m in METHODS if m in runs]

    cols = ["ID", "OOD-1", "OOD-2", "gap_ID_OOD1", "gap_ID_OOD2", "worst_group_OOD1"]
    head = ["ID acc", "OOD-1 acc", "OOD-2 acc", "Gap ID-OOD1", "Gap ID-OOD2", "Worst group (OOD-1)"]
    L = [f"## Table 3: baseline vs mitigation (rho {a.rho:g}); accuracy in %, mean \u00b1 std over seeds",
         "", "| Method | Seeds | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(cols)]
    summary = {}
    for m in methods:
        seeds = sorted(runs[m])
        vals = {c: [runs[m][s][c] for s in seeds] for c in cols}
        summary[m] = {"seeds": seeds, **{c: mean_std(v) for c, v in vals.items()}}
        L.append(f"| {LABELS[m]} | {len(seeds)} | " + " | ".join(fmt(vals[c]) for c in cols) + " |")

    L += ["", "## Individual runs (accuracy %)", "", "| Method | Seed | ID | OOD-1 | OOD-2 | Worst group (OOD-1) |",
          "|---|---|---|---|---|---|"]
    for m in methods:
        for s in sorted(runs[m]):
            r = runs[m][s]
            L.append(f"| {LABELS[m]} | {s} | {r['ID']:.2f} | {r['OOD-1']:.2f} | {r['OOD-2']:.2f} | "
                     f"{r['worst_group_OOD1']:.2f} |")

    L += ["", "## Change vs baseline with the same seed (points)", ""]
    for m in methods:
        if m == "baseline":
            continue
        common = sorted(set(runs[m]) & set(runs["baseline"]))
        if not common:
            continue
        parts = []
        for key, name in (("ID", "ID"), ("OOD-1", "OOD-1"), ("OOD-2", "OOD-2")):
            d = [runs[m][s][key] - runs["baseline"][s][key] for s in common]
            parts.append(f"{name} {st.mean(d):+.2f} (per seed: " + ", ".join(f"{x:+.2f}" for x in d) + ")")
        L.append(f"- **{LABELS[m]}**: " + "; ".join(parts))
    text = "\n".join(L) + "\n"

    os.makedirs(a.table_dir, exist_ok=True)
    out_md = os.path.join(a.table_dir, f"table3{tag}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(a.results_dir, f"table3{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(text)
    print("saved", out_md)

    # ---------------- figure: per-seed dots + mean ----------------
    keys = [("ID", "ID test"), ("OOD-1", "OOD-1"), ("OOD-2", "OOD-2"), ("worst_group_OOD1", "worst group\n(OOD-1)")]
    allv = [runs[m][s][k] for m in methods for s in runs[m] for k, _ in keys]
    lo = max(0, int(min(allv)) - 2)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        for i, (k, _) in enumerate(keys):
            x = i + (j - (len(methods) - 1) / 2) * w
            vals = [runs[m][s][k] for s in sorted(runs[m])]
            ax.scatter([x] * len(vals), vals, s=14, color=f"C{j}", alpha=0.55, zorder=3)
            ax.plot([x - w * 0.35, x + w * 0.35], [st.mean(vals)] * 2, color=f"C{j}", linewidth=3,
                    zorder=4, label=LABELS[m] if i == 0 else None)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([n for _, n in keys], fontsize=9)
    ax.set_ylim(lo, 100.5)
    ax.set_ylabel("accuracy (%)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Baseline vs mitigation, rho {a.rho:g} (bar = mean, dots = individual seeds; y-axis starts at {lo}%)",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(a.fig_dir, exist_ok=True)
    out_fig = os.path.join(a.fig_dir, f"table3_comparison{tag}.png")
    fig.savefig(out_fig, dpi=150)
    print("saved", out_fig)


if __name__ == "__main__":
    main()