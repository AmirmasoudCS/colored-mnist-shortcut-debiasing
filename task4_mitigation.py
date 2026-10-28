"""Task 4: train bias-mitigation methods and evaluate them on the five sets.

Run from the repo root:
    python -m task4_mitigation --rho 0.99 --methods lff --seeds 0 1 2
    python -m task4_mitigation --rho 0.99 --methods reweighting group_dro lff --seeds 0 1 2

Outputs per (method, seed), tag = '_rho0.99':
    outputs/models/{method}{tag}_seed{S}.pth
    outputs/results/{method}_eval{tag}_seed{S}.json      (same layout as baseline_eval*.json)
    outputs/results/{method}_history{tag}_seed{S}.json
    reports/figures/{method}_curves{tag}_seed{S}.png
LfF additionally: outputs/results/lff_biased_eval{tag}_seed{S}.json  (the auxiliary biased model)
"""
import argparse
import os

import torch

from spurious.data import build_datasets
from spurious.evaluation import evaluate_all
from spurious.methods import METHODS
from spurious.models import SimpleCNN
from spurious.plotting import plot_history
from spurious.utils import ensure_dir, get_device, rho_tag, save_json

LABELS = {"reweighting": "Sample reweighting", "group_dro": "Group DRO", "lff": "LfF"}


def result_line(res):
    wg = min(res["test_ood1"]["group_acc"].values())
    return "ID {:.2f} | OOD-1 {:.2f} | OOD-2 {:.2f} | worst-group on OOD-1 {:.2f}".format(
        100 * res["test_id"]["acc"], 100 * res["test_ood1"]["acc"],
        100 * res["test_ood2"]["acc"], 100 * wg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--fig-dir", default="./reports/figures")
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dro-eta", type=float, default=0.01, help="Group DRO step size for q")
    ap.add_argument("--lff-q", type=float, default=0.7, help="LfF: GCE exponent of the biased model")
    ap.add_argument("--lff-alpha", type=float, default=0.7, help="LfF: EMA factor of the per-sample losses")
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| methods:", a.methods, "| seeds:", a.seeds)
    data = build_datasets(a.data_dir, rho=a.rho)
    tag = rho_tag(a.rho)
    ensure_dir(os.path.join(a.out_dir, "models"))
    ensure_dir(a.fig_dir)
    hparams = {"reweighting": {}, "group_dro": {"eta": a.dro_eta},
               "lff": {"q": a.lff_q, "ema_alpha": a.lff_alpha}}

    for seed in a.seeds:
        for method in a.methods:
            print(f"\n===== {LABELS[method]} | rho {a.rho:g} | seed {seed} =====")
            model, hist = METHODS[method](data["train"], data["val"], device, seed,
                                          a.epochs, a.batch_size, a.lr, **hparams[method])
            biased_state = hist.pop("_biased_state", None)       # LfF only
            res = evaluate_all(model, data, device)

            torch.save(model.state_dict(), os.path.join(a.out_dir, "models", f"{method}{tag}_seed{seed}.pth"))
            save_json(res, os.path.join(a.out_dir, "results", f"{method}_eval{tag}_seed{seed}.json"))
            save_json(hist, os.path.join(a.out_dir, "results", f"{method}_history{tag}_seed{seed}.json"))
            plot_history(hist, os.path.join(a.fig_dir, f"{method}_curves{tag}_seed{seed}.png"),
                         f"{LABELS[method]} (rho {a.rho:g}, seed {seed})")
            print("RESULT  " + result_line(res))

            if biased_state is not None:                          # diagnostic: is the biased model biased?
                biased = SimpleCNN().to(device)
                biased.load_state_dict(biased_state)
                res_b = evaluate_all(biased, data, device)
                save_json(res_b, os.path.join(a.out_dir, "results", f"lff_biased_eval{tag}_seed{seed}.json"))
                print("BIASED MODEL (auxiliary)  " + result_line(res_b))


if __name__ == "__main__":
    main()