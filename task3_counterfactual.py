"""Task 3.1: counterfactual tests (colour reversed / colour removed) on a trained baseline.

Run from the repo root:
    python -m task3_counterfactual                       # rho 0.9, seed 0
    python -m task3_counterfactual --rho 0.99 --seed 0

Outputs (tag = '' for rho 0.9, '_rho0.99' for rho 0.99):
    outputs/results/counterfactual{tag}_seed{S}.json
    reports/tables/table2{tag}_seed{S}.md
    reports/figures/counterfactual_examples.png      (what the transforms look like)
"""
import argparse
import os

import torch

from spurious.cue_detection import CONDITION_LABELS, CONDITIONS, counterfactual_tests, plot_examples
from spurious.data import build_datasets
from spurious.models import SimpleCNN
from spurious.utils import ensure_dir, get_device, model_name, rho_tag, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--table-dir", default="./reports/tables")
    ap.add_argument("--fig-dir", default="./reports/figures")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| seed:", a.seed)
    data = build_datasets(a.data_dir, rho=a.rho)
    test = data["test_id"]

    model = SimpleCNN().to(device)
    path = os.path.join(a.out_dir, "models", model_name(a.rho, a.seed))
    model.load_state_dict(torch.load(path, map_location=device))
    res = counterfactual_tests(model, test, device)

    tag = rho_tag(a.rho)
    save_json(res, os.path.join(a.out_dir, "results", f"counterfactual{tag}_seed{a.seed}.json"))
    ensure_dir(a.fig_dir)
    plot_examples(test, os.path.join(a.fig_dir, "counterfactual_examples.png"))

    clean = res["clean"]["acc"]
    lines = [f"## Table 2: counterfactual tests (rho {a.rho:g}, seed {a.seed}, n = {res['n']})",
             "| Condition | Accuracy (%) | Change vs clean (pts) | Loss |", "|---|---|---|---|"]
    for k in CONDITIONS:
        r = res[k]
        lines.append(f"| {CONDITION_LABELS[k]} | {100 * r['acc']:.2f} | "
                     f"{100 * (r['acc'] - clean):+.2f} | {r['loss']:.4f} |")
    lines.append("")
    lines.append(f"Colour swing = acc(all aligned) - acc(all reversed) = {100 * res['colour_swing']:.2f} pts "
                 "(0 for a pure shape model)")
    text = "\n".join(lines) + "\n"
    ensure_dir(a.table_dir)
    out = os.path.join(a.table_dir, f"table2{tag}_seed{a.seed}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()