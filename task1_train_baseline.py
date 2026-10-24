"""Task 1: train the baseline CNN on Colored MNIST.

Run from the repo root:
    python -m task1_train_baseline                        # handout setting: rho 0.9, seed 0
    python -m task1_train_baseline --rho 0.99 --seed 0    # stronger-bias extension
    python -m task1_train_baseline --epochs 1             # quick smoke test

Outputs (tag = '' for rho 0.9, '_rho0.99' for rho 0.99):
    outputs/models/baseline_model.pth  or  baseline_model{tag}_seed{S}.pth
    outputs/results/baseline_history{tag}_seed{S}.json
    reports/figures/baseline_curves{tag}_seed{S}.png
"""
import argparse
import os

import torch
import torch.nn.functional as F

from spurious.data import batches, build_datasets
from spurious.evaluation import evaluate
from spurious.models import SimpleCNN
from spurious.plotting import plot_history
from spurious.utils import ensure_dir, get_device, model_name, rho_tag, save_json, set_seed


def train(model, train_set, val_set, device, epochs, bs, lr, seed):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)  # reproducible shuffling
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        for xb, yb, _ in batches(train_set, bs, shuffle=True, device=device, generator=gen):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

        # end-of-epoch metrics, both measured in eval mode (dropout off)
        tl, ta = evaluate(model, train_set, device)
        vl, va = evaluate(model, val_set, device)
        for key, value in zip(hist, (tl, ta, vl, va)):
            hist[key].append(value)
        print(f"epoch {ep + 1:02d}/{epochs}  train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f}", flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--fig-dir", default="./reports/figures")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| seed:", a.seed)
    set_seed(a.seed)

    data = build_datasets(a.data_dir, rho=a.rho)
    model = SimpleCNN().to(device)
    hist = train(model, data["train"], data["val"], device,
                 a.epochs, a.batch_size, a.lr, a.seed)

    tag = rho_tag(a.rho)
    path = os.path.join(a.out_dir, "models", model_name(a.rho, a.seed))
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    save_json(hist, os.path.join(a.out_dir, "results", f"baseline_history{tag}_seed{a.seed}.json"))
    ensure_dir(a.fig_dir)
    plot_history(hist, os.path.join(a.fig_dir, f"baseline_curves{tag}_seed{a.seed}.png"),
                 f"Baseline CNN (rho {a.rho:g}, seed {a.seed})")

    print(f"\nsaved weights -> {path}")
    print(f"final val accuracy: {hist['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()