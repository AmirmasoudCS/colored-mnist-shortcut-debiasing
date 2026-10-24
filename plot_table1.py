"""Figure for the Task 2 results (reads the JSON written by task2_evaluate.py).

Run from the repo root (needs only matplotlib):
    python -m plot_table1

Output: reports/figures/table1_baseline_eval.png
Panels: (a) accuracy per set: overall / aligned / conflicting samples (95% binomial CI)
        (b) accuracy of the four colour/digit groups on the three test sets
        (c) cross-entropy loss per set
NOTE: panels (a) and (b) use a truncated y-axis (it starts near 96%) because all
accuracies are close to 100%; the axis range is printed in the panel titles.
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETS = ["train", "val", "test_id", "test_ood1", "test_ood2"]
LABELS = {"train": "train\n(rho 0.9)", "val": "val\n(rho 0.9)", "test_id": "ID test\n(rho 0.9)",
          "test_ood1": "OOD-1\n(rho 0.5)", "test_ood2": "OOD-2\n(rho 0.1)"}
GROUPS = ["red/0-4", "red/5-9", "green/0-4", "green/5-9"]
GROUP_STYLE = {  # colour = digit colour, marker shape = digit range, filled = aligned
    "red/0-4": ("#d62728", "o", True), "red/5-9": ("#d62728", "s", False),
    "green/0-4": ("#2ca02c", "o", False), "green/5-9": ("#2ca02c", "s", True)}


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./outputs/results/baseline_eval_seed0.json")
    ap.add_argument("--out", default="./reports/figures/table1_baseline_eval.png")
    ap.add_argument("--title", default="Baseline CNN (seed 0)")
    a = ap.parse_args()
    with open(a.results) as f:
        res = json.load(f)

    # shared lower bound for panels (a) and (b)
    lows = []
    for s in SETS:
        r = res[s]
        n_conf = r["n"] - r["n_aligned"]
        lows += [100 * (r["conflicting_acc"] - ci95(r["conflicting_acc"], n_conf)),
                 100 * r["aligned_acc"]]
    for s in SETS[2:]:
        lows += [100 * v for v in res[s]["group_acc"].values()]
    lo = max(0, math.floor(min(lows)) - 1)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 0.9]})
    x = list(range(len(SETS)))

    # (a) overall / aligned / conflicting
    for off, key, label, marker, color in [(-0.2, "aligned_acc", "aligned samples", "o", "#1f77b4"),
                                           (0.0, "acc", "all samples", "_", "black"),
                                           (0.2, "conflicting_acc", "conflicting samples", "s", "#ff7f0e")]:
        vals, errs = [], []
        for s in SETS:
            r = res[s]
            n = {"aligned_acc": r["n_aligned"], "acc": r["n"],
                 "conflicting_acc": r["n"] - r["n_aligned"]}[key]
            vals.append(100 * r[key])
            errs.append(100 * ci95(r[key], n))
        ax[0].errorbar([i + off for i in x], vals, yerr=errs, fmt=marker, color=color, capsize=3,
                       markersize=9 if marker == "_" else 6, label=label)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([LABELS[s] for s in SETS], fontsize=8)
    ax[0].set_ylabel("accuracy (%)")
    ax[0].set_title(f"(a) Accuracy by sample type\n(y-axis starts at {lo}%; bars = 95% CI)", fontsize=10)

    # (b) four groups on the three test sets
    tests = SETS[2:]
    for j, g in enumerate(GROUPS):
        color, marker, filled = GROUP_STYLE[g]
        kind = "aligned" if filled else "conflicting"
        ax[1].plot([i + (j - 1.5) * 0.15 for i in range(len(tests))],
                   [100 * res[s]["group_acc"][g] for s in tests], marker, color=color,
                   markerfacecolor=color if filled else "white", markersize=8,
                   label=f"{g} ({kind})")
    ax[1].set_xticks(range(len(tests)))
    ax[1].set_xticklabels([LABELS[s] for s in tests], fontsize=8)
    ax[1].set_title(f"(b) Accuracy by group\n(y-axis starts at {lo}%)", fontsize=10)

    for k in (0, 1):
        ax[k].set_ylim(lo, 100.3)
        ax[k].set_yticks(range(lo, 101))
        ax[k].grid(alpha=0.3)
        ax[k].legend(fontsize=7, loc="lower left")

    # (c) loss, bars from zero
    losses = [res[s]["loss"] for s in SETS]
    bars = ax[2].bar(x, losses, color="#7f7f7f")
    for b, v in zip(bars, losses):
        ax[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([LABELS[s] for s in SETS], fontsize=8)
    ax[2].set_ylabel("cross-entropy loss")
    ax[2].set_ylim(0, max(losses) * 1.15)
    ax[2].set_title("(c) Loss (y-axis from 0)", fontsize=10)
    ax[2].grid(alpha=0.3, axis="y")

    fig.suptitle(a.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()