"""Task 2: evaluate the saved baseline on the five sets (Table 1).

Run from the repo root:
    python -m task2_evaluate            # uses outputs/models/baseline_model.pth (seed 0)

Outputs:
    outputs/results/baseline_eval_seed{S}.json
    reports/tables/table1_seed{S}.md
"""
import argparse
import os

import torch

from spurious.data import build_datasets
from spurious.evaluation import GROUP_NAMES, evaluate_all
from spurious.models import SimpleCNN
from spurious.utils import ensure_dir, get_device, save_json

SET_INFO = {  # key -> (rho, description)
    "train": (0.9, "training"),
    "val": (0.9, "validation"),
    "test_id": (0.9, "ID test"),
    "test_ood1": (0.5, "OOD-1: unbiased"),
    "test_ood2": (0.1, "OOD-2: reversed"),
}


def pct(x):
    return f"{100 * x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--table-dir", default="./reports/tables")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = get_device()
    print("device:", device)
    data = build_datasets(a.data_dir)

    name = "baseline_model.pth" if a.seed == 0 else f"baseline_model_seed{a.seed}.pth"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(a.out_dir, "models", name), map_location=device))
    res = evaluate_all(model, data, device)
    save_json(res, os.path.join(a.out_dir, "results", f"baseline_eval_seed{a.seed}.json"))

    # ---- Table 1 (handout format) + aligned / conflicting breakdown ----
    t1 = ["| Set | rho | Accuracy (%) | Description |", "|---|---|---|---|"]
    for k, (rho, desc) in SET_INFO.items():
        t1.append(f"| {k} | {rho} | {pct(res[k]['acc'])} | {desc} |")

    t1b = ["| Set | Aligned acc (%) | Conflicting acc (%) | Loss |", "|---|---|---|---|"]
    for k in SET_INFO:
        r = res[k]
        t1b.append(f"| {k} | {pct(r['aligned_acc'])} | {pct(r['conflicting_acc'])} | {r['loss']:.4f} |")

    tg = ["| Set | " + " | ".join(GROUP_NAMES) + " |", "|---|" + "---|" * len(GROUP_NAMES)]
    for k in ("test_id", "test_ood1", "test_ood2"):
        tg.append(f"| {k} | " + " | ".join(pct(res[k]["group_acc"][g]) for g in GROUP_NAMES) + " |")

    text = ("## Table 1: baseline accuracy\n" + "\n".join(t1)
            + "\n\n## Aligned vs conflicting samples\n" + "\n".join(t1b)
            + "\n\n## Accuracy per group (colour/digits)\n" + "\n".join(tg) + "\n")
    ensure_dir(a.table_dir)
    path = os.path.join(a.table_dir, f"table1_seed{a.seed}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"saved {path}")


if __name__ == "__main__":
    main()