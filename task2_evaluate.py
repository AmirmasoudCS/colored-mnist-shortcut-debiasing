"""Task 2: evaluate a saved baseline on the five sets (Table 1).

Run from the repo root:
    python -m task2_evaluate                          # rho 0.9, seed 0
    python -m task2_evaluate --rho 0.99 --seed 0      # stronger-bias extension

Outputs (tag = '' for rho 0.9, '_rho0.99' for rho 0.99):
    outputs/results/baseline_eval{tag}_seed{S}.json
    reports/tables/table1{tag}_seed{S}.md
"""
import argparse
import os

import torch

from spurious.data import build_datasets
from spurious.evaluation import GROUP_NAMES, evaluate_all
from spurious.models import SimpleCNN
from spurious.utils import ensure_dir, get_device, model_name, rho_tag, save_json


def set_info(rho):  # key -> (rho, description)
    return {
        "train": (rho, "training"),
        "val": (rho, "validation"),
        "test_id": (rho, "ID test"),
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
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| seed:", a.seed)
    data = build_datasets(a.data_dir, rho=a.rho)

    model = SimpleCNN().to(device)
    path = os.path.join(a.out_dir, "models", model_name(a.rho, a.seed))
    model.load_state_dict(torch.load(path, map_location=device))
    res = evaluate_all(model, data, device)
    tag = rho_tag(a.rho)
    save_json(res, os.path.join(a.out_dir, "results", f"baseline_eval{tag}_seed{a.seed}.json"))

    info = set_info(a.rho)
    t1 = ["| Set | rho | Accuracy (%) | Description |", "|---|---|---|---|"]
    for k, (rho, desc) in info.items():
        t1.append(f"| {k} | {rho:g} | {pct(res[k]['acc'])} | {desc} |")

    t1b = ["| Set | Aligned acc (%) | Conflicting acc (%) | Loss | n conflicting |",
           "|---|---|---|---|---|"]
    for k in info:
        r = res[k]
        t1b.append(f"| {k} | {pct(r['aligned_acc'])} | {pct(r['conflicting_acc'])} | "
                   f"{r['loss']:.4f} | {r['n'] - r['n_aligned']} |")

    tg = ["| Set | " + " | ".join(GROUP_NAMES) + " |", "|---|" + "---|" * len(GROUP_NAMES)]
    for k in ("test_id", "test_ood1", "test_ood2"):
        tg.append(f"| {k} | " + " | ".join(pct(res[k]["group_acc"][g]) for g in GROUP_NAMES) + " |")

    text = (f"## Table 1: baseline accuracy (rho {a.rho:g}, seed {a.seed})\n" + "\n".join(t1)
            + "\n\n## Aligned vs conflicting samples\n" + "\n".join(t1b)
            + "\n\n## Accuracy per group (colour/digits)\n" + "\n".join(tg) + "\n")
    ensure_dir(a.table_dir)
    out = os.path.join(a.table_dir, f"table1{tag}_seed{a.seed}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()