"""Task 3.4: misspecification tests on a trained baseline.

Shape cue changed: rotate the digit by 90 degrees / hide the right half of the image.
Colour cue changed: natural colour, reversed colour, or removed (grayscale, full brightness).
All 3 x 3 combinations are evaluated on n random test digits (the handout asks for 1000).
Because the colour is always re-painted, the result does not depend on the test set's colouring.

Run from the repo root:
    python -m task3_misspecification                       # rho 0.9, seed 0, n = 1000
    python -m task3_misspecification --rho 0.99 --seed 0
    python -m task3_misspecification --n 10000             # all test digits, tighter intervals

Outputs (tag = '' for rho 0.9, '_rho0.99' for rho 0.99):
    outputs/results/misspec{tag}_seed{S}.json
    reports/tables/misspec{tag}_seed{S}.md
    reports/figures/misspec_examples.png
"""
import argparse
import math
import os

import torch

from spurious.cue_detection import (COLOUR_CONDITIONS, SHAPE_CONDITIONS, misspecification_tests,
                                    plot_misspec_examples)
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
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| seed:", a.seed, "| n:", a.n)
    data = build_datasets(a.data_dir, rho=a.rho)
    test = data["test_id"]

    model = SimpleCNN().to(device)
    path = os.path.join(a.out_dir, "models", model_name(a.rho, a.seed))
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    res = misspecification_tests(model, test, device, n=a.n, seed=a.seed)
    tag = rho_tag(a.rho)
    save_json(res, os.path.join(a.out_dir, "results", f"misspec{tag}_seed{a.seed}.json"))
    ensure_dir(a.fig_dir)
    plot_misspec_examples(test, os.path.join(a.fig_dir, "misspec_examples.png"))

    ci = 196 * math.sqrt(0.25 / res["n"])   # worst-case 95% half-width in points (p = 0.5)
    shapes, colours = list(SHAPE_CONDITIONS), list(COLOUR_CONDITIONS)
    L = [f"## Misspecification tests (rho {a.rho:g}, seed {a.seed}, n = {res['n']}; 95% CI at most +-{ci:.1f} pts)",
         "", "**Accuracy (%)**", "| Shape | " + " | ".join(colours) + " |", "|---|" + "---|" * len(colours)]
    for s in shapes:
        L.append(f"| {s} | " + " | ".join(f"{100 * res[f'{s}/{c}']['acc']:.1f}" for c in colours) + " |")
    L += ["", "**Prediction follows the colour (%)** (about 50 = colour has no pull)",
          "| Shape | natural | reversed |", "|---|---|---|"]
    for s in shapes:
        L.append(f"| {s} | {100 * res[f'{s}/natural']['follows_colour']:.1f} | "
                 f"{100 * res[f'{s}/reversed']['follows_colour']:.1f} |")
    text = "\n".join(L) + "\n"
    ensure_dir(a.table_dir)
    out = os.path.join(a.table_dir, f"misspec{tag}_seed{a.seed}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()