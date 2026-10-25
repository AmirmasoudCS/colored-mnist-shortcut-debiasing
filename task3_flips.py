"""Variant of Task 3.2/3.3: analyse the colour-flipped digits.

A digit is "colour-flipped" if the model classifies it correctly in its natural colour
(0-4 red, 5-9 green) but wrongly after the colour is reversed. For those digits we ask:
  * how many are there, and in which direction (0-4 painted green vs 5-9 painted red)?
  * does the wrong answer follow the colour? (crosses the 0-4 / 5-9 boundary)
  * what do Grad-CAM / saliency show when the model explains the WRONG class?

Run from the repo root:
    python -m task3_flips                         # rho 0.9, seed 0
    python -m task3_flips --rho 0.99 --seed 0

Outputs (tag = '' for rho 0.9, '_rho0.99' for rho 0.99):
    reports/figures/flips{tag}_seed{S}.png         (first 10 flipped digits, in dataset order)
    reports/tables/flips{tag}_seed{S}.md
    outputs/results/flips{tag}_seed{S}.json
"""
import argparse
import os

import torch

from spurious.cue_detection import (anti_color, attribute, find_flips, grayscale_max,
                                    natural_color, plot_flips)
from spurious.evaluation import predict_logits
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
    ap.add_argument("--n-show", type=int, default=10, help="digits shown in the figure")
    ap.add_argument("--max-stats", type=int, default=1000, help="max flipped digits used for map statistics")
    a = ap.parse_args()

    device = get_device()
    print("device:", device, "| rho:", a.rho, "| seed:", a.seed)
    data = build_datasets(a.data_dir, rho=a.rho)
    test = data["test_id"]
    model = SimpleCNN().to(device)
    path = os.path.join(a.out_dir, "models", model_name(a.rho, a.seed))
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # ---------- which digits flip, and in which direction ----------
    flipped, pred_nat, pred_rev = find_flips(model, test, device)
    y_all = test[1]
    n_total, n_flip = len(y_all), int(flipped.sum())
    if n_flip == 0:
        print("no colour-flipped digits found")
        return
    nat_ok = pred_nat == y_all
    low, high = y_all < 5, y_all >= 5
    res = {"n_digits": n_total, "n_natural_correct": int(nat_ok.sum()), "n_flipped": n_flip,
           "flip_rate_among_natural_correct": n_flip / int(nat_ok.sum())}
    for name, grp in (("digits_0_4_painted_green", low), ("digits_5_9_painted_red", high)):
        denom = int((nat_ok & grp).sum())
        cnt = int((flipped & grp).sum())
        res[name] = {"flipped": cnt, "natural_correct": denom, "rate": cnt / max(denom, 1)}

    wrong_all, y_f = pred_rev[flipped], y_all[flipped]
    res["wrong_answer_crosses_boundary"] = ((wrong_all >= 5) != (y_f >= 5)).float().mean().item()
    # control: errors the same model makes on colour-free (grayscale) images. Ordinary
    # look-alike confusions (4/9, 3/5, 2/7 ...) also straddle the 0-4 / 5-9 boundary, so a
    # high crossing rate alone is not proof of colour influence.
    pred_gray = predict_logits(model, test, device, transform=grayscale_max).argmax(dim=1)
    err = pred_gray != y_all
    res["control_gray_errors"] = {
        "n_errors": int(err.sum()),
        "crosses_boundary": ((pred_gray[err] >= 5) != (y_all[err] >= 5)).float().mean().item()}
    pair = torch.bincount(y_f * 10 + wrong_all, minlength=100)
    top = torch.topk(pair.float(), 5)
    res["top_confusions"] = [{"true": int(i) // 10, "predicted": int(i) % 10, "count": int(c)}
                             for c, i in zip(top.values, top.indices) if c > 0]

    # ---------- maps for the flipped digits ----------
    sel = torch.nonzero(flipped).squeeze(1)[:a.max_stats]
    x0 = test[0][sel].float().div(255.0).to(device)
    y = y_all[sel].to(device)
    wrong = pred_rev[sel].to(device)
    c0 = test[2][sel].to(device)
    c_nat = (y >= 5).long()
    x_nat, x_rev = natural_color(x0, y, c0), anti_color(x0, y, c0)

    nat, s_nat = attribute(model, x_nat, y, c_nat)                  # natural colour, true class
    rev_t, s_rev_t = attribute(model, x_rev, y, 1 - c_nat)          # reversed colour, true class
    rev_w, s_rev_w = attribute(model, x_rev, wrong, 1 - c_nat)      # reversed colour, wrong class
    res["map_stats"] = {"natural_true_class": s_nat, "reversed_true_class": s_rev_t,
                        "reversed_wrong_class": s_rev_w, "n_digits_in_stats": int(len(sel))}

    tag = rho_tag(a.rho)
    ensure_dir(a.fig_dir)
    k = min(a.n_show, len(sel))
    def first(m):
        return {key: (v[:k] if isinstance(v, list) or hasattr(v, "shape") else v) for key, v in m.items()}
    plot_flips(first(nat), first(rev_t), first(rev_w), y[:k].cpu().tolist(), wrong[:k].cpu().tolist(),
               os.path.join(a.fig_dir, f"flips{tag}_seed{a.seed}.png"),
               f"Colour-flipped digits - rho {a.rho:g}, seed {a.seed} (first {k} in dataset order)")
    save_json(res, os.path.join(a.out_dir, "results", f"flips{tag}_seed{a.seed}.json"))

    # ---------- markdown ----------
    L = [f"## Colour-flipped digits (rho {a.rho:g}, seed {a.seed})",
         f"{n_flip} of {res['n_natural_correct']} naturally-correct digits ({100 * res['flip_rate_among_natural_correct']:.2f}%) "
         "are classified wrongly once their colour is reversed.", "",
         "| Direction | Flipped | Naturally correct | Flip rate (%) |", "|---|---|---|---|"]
    for name, label in (("digits_0_4_painted_green", "digits 0-4 painted green"),
                        ("digits_5_9_painted_red", "digits 5-9 painted red")):
        r = res[name]
        L.append(f"| {label} | {r['flipped']} | {r['natural_correct']} | {100 * r['rate']:.2f} |")
    L += ["", f"Wrong answer crosses the 0-4 / 5-9 boundary (follows the colour): "
          f"{100 * res['wrong_answer_crosses_boundary']:.1f}%.", "",
          f"Control: among the {res['control_gray_errors']['n_errors']} errors this model makes on colour-free "
          f"(grayscale) images, {100 * res['control_gray_errors']['crosses_boundary']:.1f}% cross the boundary "
          "(look-alike confusions such as 4/9 or 3/5 straddle it too).", "",
          "Top confusions (true -> predicted): " + ", ".join(
              f"{t['true']}->{t['predicted']} ({t['count']})" for t in res["top_confusions"]), "",
          f"| Map statistics over {len(sel)} flipped digits | Grad-CAM mass on digit | Gradient mass on digit | "
          "Gradient share in own colour channel |", "|---|---|---|---|"]
    for key, label in (("natural_true_class", "natural colour, explaining true class"),
                       ("reversed_true_class", "reversed colour, explaining true class"),
                       ("reversed_wrong_class", "reversed colour, explaining wrong class")):
        s = res["map_stats"][key]
        L.append(f"| {label} | {s['gradcam_mass_on_digit']:.3f} | {s['gradient_mass_on_digit']:.3f} | "
                 f"{s['gradient_share_own_colour_channel']:.3f} |")
    L.append("")
    L.append(f"Digit pixels cover {s_nat['digit_area_fraction']:.3f} of the image (uniform-spread reference).")
    text = "\n".join(L) + "\n"
    ensure_dir(a.table_dir)
    out = os.path.join(a.table_dir, f"flips{tag}_seed{a.seed}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"saved {out} and the figure in {a.fig_dir}")


if __name__ == "__main__":
    main()