"""Task 3.2 / 3.3: Gradient x Input saliency and Grad-CAM.

For the same 100 test digits (10 per class) the maps are computed twice: with the natural
colour (aligned) and with the reversed colour (conflicting). The shapes are identical, so
any difference is caused by colour alone. The explained class is the TRUE label; titles show
the true label y and the model's prediction (y-hat).

Run from the repo root:
    python -m task3_attribution                       # rho 0.9, seed 0
    python -m task3_attribution --rho 0.99 --seed 0

Outputs (tag = '' for rho 0.9, '_rho0.99' for rho 0.99):
    reports/figures/saliency{tag}_seed{S}.png      reports/figures/gradcam{tag}_seed{S}.png
    reports/tables/attribution_stats{tag}_seed{S}.md
    outputs/results/attribution{tag}_seed{S}.json
"""
import argparse
import os

import torch

from spurious.cue_detection import (anti_color, attribute, natural_color, plot_attribution,
                                    select_per_class)
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
    ap.add_argument("--per-class", type=int, default=10)
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| seed:", a.seed)
    data = build_datasets(a.data_dir, rho=a.rho)
    test = data["test_id"]

    model = SimpleCNN().to(device)
    path = os.path.join(a.out_dir, "models", model_name(a.rho, a.seed))
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    sel = select_per_class(test, a.per_class)
    x0 = test[0][sel].float().div(255.0).to(device)
    y = test[1][sel].to(device)
    c0 = test[2][sel].to(device)

    maps, stats = {}, {}
    for cond, tf in (("natural", natural_color), ("reversed", anti_color)):
        x = tf(x0, y, c0)                                  # same digits, colour re-painted
        c = (y >= 5).long() if cond == "natural" else 1 - (y >= 5).long()
        maps[cond], stats[cond] = attribute(model, x, y, c)

    tag = rho_tag(a.rho)
    ensure_dir(a.fig_dir)
    cols = list(range(0, 10 * a.per_class, a.per_class))   # one digit per class in the figures
    plot_attribution(maps, cols,
                     os.path.join(a.fig_dir, f"saliency{tag}_seed{a.seed}.png"),
                     os.path.join(a.fig_dir, f"gradcam{tag}_seed{a.seed}.png"),
                     f"rho {a.rho:g}, seed {a.seed}")
    save_json(stats, os.path.join(a.out_dir, "results", f"attribution{tag}_seed{a.seed}.json"))

    n = 10 * a.per_class
    lines = [f"## Attribution statistics (rho {a.rho:g}, seed {a.seed}, {n} digits, same digits in both rows)",
             "| Colour | Accuracy (%) | Grad-CAM mass on digit | Gradient mass on digit | "
             "Gradient share in digit's own colour channel |", "|---|---|---|---|---|"]
    for cond in ("natural", "reversed"):
        s = stats[cond]
        lines.append(f"| {cond} | {100 * s['accuracy']:.1f} | {s['gradcam_mass_on_digit']:.3f} | "
                     f"{s['gradient_mass_on_digit']:.3f} | {s['gradient_share_own_colour_channel']:.3f} |")
    lines.append("")
    lines.append(f"Digit pixels cover {stats['natural']['digit_area_fraction']:.3f} of the image "
                 "(value expected if the map were spread uniformly).")
    text = "\n".join(lines) + "\n"
    ensure_dir(a.table_dir)
    out = os.path.join(a.table_dir, f"attribution_stats{tag}_seed{a.seed}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"saved {out} and the two figures in {a.fig_dir}")


if __name__ == "__main__":
    main()