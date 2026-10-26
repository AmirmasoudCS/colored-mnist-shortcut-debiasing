"""Task 3: which cue does the model use?  (3.1: counterfactual tests)

All transforms take a float batch x (B, 3, 28, 28) in [0, 1] plus labels y and colours c,
and return a new batch. Colour is a whole channel (0 = red, 1 = green), so the digit's
shape is recovered as x.max over channels.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .evaluation import evaluate, predict_logits  # noqa: E402


def set_color(x, c):
    """Re-paint every digit in colour c[i] (0 red, 1 green), keeping its shape."""
    gray = x.max(dim=1).values                       # (B, 28, 28)
    out = torch.zeros_like(x)
    out[torch.arange(x.shape[0], device=x.device), c] = gray
    return out


def natural_color(x, y, c):
    """Reference: 0-4 red, 5-9 green (every sample aligned)."""
    return set_color(x, (y >= 5).long())


def anti_color(x, y, c):
    """Test A: 0-4 green, 5-9 red (every sample conflicting)."""
    return set_color(x, 1 - (y >= 5).long())


def grayscale_mean(x, y, c):
    """Test B (handout): all three channels = mean of the channels.
    NOTE: this also divides the digit's intensity by 3 (a pure red digit becomes 1/3 grey)."""
    return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


def grayscale_max(x, y, c):
    """Control for Test B: colour removed but full intensity kept (max over channels)."""
    return x.max(dim=1, keepdim=True).values.repeat(1, 3, 1, 1)


CONDITIONS = {
    "clean": None,
    "natural_color": natural_color,        # all samples aligned
    "A_anti_color": anti_color,            # all samples conflicting
    "B_grayscale_mean": grayscale_mean,    # handout's Test B
    "B2_grayscale_max": grayscale_max,     # Test B with intensity preserved
}

CONDITION_LABELS = {
    "clean": "clean test set",
    "natural_color": "all aligned (reference)",
    "A_anti_color": "Test A: colours reversed",
    "B_grayscale_mean": "Test B: grayscale (mean)",
    "B2_grayscale_max": "Test B': grayscale (max, control)",
}


def counterfactual_tests(model, split, device):
    """Accuracy and loss of `model` on `split` under each condition.

    Because every transform overwrites the colour, the transformed results depend only
    on the digit shapes, not on the split's original colouring.
    colour_swing = acc(all aligned) - acc(all conflicting): 0 for a pure shape model.
    """
    res = {"n": int(split[1].shape[0])}
    for name, tf in CONDITIONS.items():
        loss, acc = evaluate(model, split, device, transform=tf)
        res[name] = {"acc": acc, "loss": loss}
    res["colour_swing"] = res["natural_color"]["acc"] - res["A_anti_color"]["acc"]
    return res


def plot_examples(split, path, n=8):
    """Visual check of the transforms (no model needed): original, A, B, B'."""
    x = split[0][:n].float().div(255.0)
    y, c = split[1][:n], split[2][:n]
    rows = [("original", x), ("A: reversed", anti_color(x, y, c)),
            ("B: gray (mean)", grayscale_mean(x, y, c)), ("B': gray (max)", grayscale_max(x, y, c))]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.3 * n + 1.6, 1.3 * len(rows) + 0.6), squeeze=False)
    for r, (label, batch) in enumerate(rows):
        for i in range(n):
            ax = axes[r][i]
            ax.imshow(batch[i].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"label {int(y[i])}", fontsize=8)
        axes[r][0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# Task 3.2 / 3.3: Gradient x Input saliency and Grad-CAM
# =====================================================================
def select_per_class(split, per_class=10):
    """Indices of the first `per_class` samples of each digit class, ordered by class."""
    y = split[1]
    return torch.cat([torch.nonzero(y == k).squeeze(1)[:per_class] for k in range(10)])


def input_gradients(model, x, target):
    """Plain gradient and Gradient x Input of the target-class logit w.r.t. the input."""
    model.eval()
    x = x.clone().requires_grad_(True)
    score = model(x).gather(1, target[:, None]).sum()
    (g,) = torch.autograd.grad(score, x)
    return g.detach(), (g * x).detach()


def grad_cam(model, x, target):
    """Grad-CAM on the last conv block (7x7), bilinearly up-sampled to 28x28 and
    max-normalised per sample."""
    model.eval()
    feats = model.features(x)                                   # (B, 128, 7, 7)
    score = model.head(feats).gather(1, target[:, None]).sum()
    (grads,) = torch.autograd.grad(score, feats)
    alpha = grads.mean(dim=(2, 3), keepdim=True)                # channel importance
    cam = torch.relu((alpha * feats).sum(dim=1, keepdim=True)).detach()
    cam = torch.nn.functional.interpolate(cam, size=28, mode="bilinear", align_corners=False)[:, 0]
    return cam / (cam.flatten(1).max(dim=1).values[:, None, None] + 1e-12)


def _norm01(m):
    return m / (m.flatten(1).max(dim=1).values[:, None, None] + 1e-12)


def attribute(model, x, y, c):
    """Maps + summary statistics for a batch.
    x: float (B, 3, 28, 28) on the model's device, y: true labels (the explained class),
    c: colour of each image (0 red / 1 green)."""
    g, gxi = input_gradients(model, x, y)
    cam = grad_cam(model, x, y)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)

    mask = (x.max(dim=1).values > 0.1).float()                  # digit pixels
    def on_digit(m):
        return (m * mask).flatten(1).sum(1) / (m.flatten(1).sum(1) + 1e-12)

    grad_map = g.abs().sum(dim=1)
    ch = g.abs().flatten(2).sum(dim=2)                          # gradient mass per colour channel
    ar = torch.arange(len(y), device=x.device)
    own, other = ch[ar, c], ch[ar, 1 - c]
    gabs = g.abs() / (g.abs().flatten(1).max(dim=1).values[:, None, None, None] + 1e-12)

    stats = {
        "accuracy": (pred == y).float().mean().item(),
        "digit_area_fraction": mask.flatten(1).mean().item(),
        "gradcam_mass_on_digit": on_digit(cam).mean().item(),
        "gradient_mass_on_digit": on_digit(grad_map).mean().item(),
        "gradient_share_own_colour_channel": (own / (own + other + 1e-12)).mean().item(),
    }
    maps = {
        "img": x.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1),
        "gxi": _norm01(gxi.abs().max(dim=1).values).cpu().numpy(),
        "grad_red": gabs[:, 0].cpu().numpy(),
        "grad_green": gabs[:, 1].cpu().numpy(),
        "cam": cam.cpu().numpy(),
        "pred": pred.cpu().tolist(),
        "y": y.cpu().tolist(),
    }
    return maps, stats


def _overlay(img, cam):
    heat = plt.cm.jet(cam)[..., :3]
    return (0.5 * img + 0.5 * heat).clip(0, 1)


def _plot_grid(rows, labels, titles, path, suptitle, cell=1.15):
    """rows: list of (list_of_arrays, cmap); titles: {row_index: [titles per column]}."""
    nr, nc = len(rows), len(rows[0][0])
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc + 2.2, cell * nr + 0.9), squeeze=False)
    for r, (arrs, cmap) in enumerate(rows):
        for k, a in enumerate(arrs):
            ax = axes[r][k]
            ax.imshow(a, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r in titles:
                ax.set_title(titles[r][k], fontsize=7)
        axes[r][0].set_ylabel(labels[r], fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attribution(maps_by_cond, cols, saliency_path, gradcam_path, title):
    """maps_by_cond: {'natural': maps, 'reversed': maps}; cols: sample indices to show."""
    sal_rows, sal_labels, sal_titles = [], [], {}
    cam_rows, cam_labels, cam_titles = [], [], {}
    for cond, m in maps_by_cond.items():
        pick = lambda key: [m[key][i] for i in cols]            # noqa: E731
        ttl = [f"y={m['y'][i]} \u0177={m['pred'][i]}" for i in cols]
        sal_titles[len(sal_rows)] = ttl
        sal_rows += [(pick("img"), None), (pick("gxi"), "hot"),
                     (pick("grad_red"), "hot"), (pick("grad_green"), "hot")]
        sal_labels += [f"{cond}: image", f"{cond}: Grad\u00d7Input", f"{cond}: |grad| red ch.",
                       f"{cond}: |grad| green ch."]
        cam_titles[len(cam_rows)] = ttl
        cam_rows += [(pick("img"), None), ([_overlay(m["img"][i], m["cam"][i]) for i in cols], None)]
        cam_labels += [f"{cond}: image", f"{cond}: Grad-CAM"]
    _plot_grid(sal_rows, sal_labels, sal_titles, saliency_path, f"Saliency - {title}")
    _plot_grid(cam_rows, cam_labels, cam_titles, gradcam_path, f"Grad-CAM - {title}")


# =====================================================================
# Variant: what drives the WRONG answer on colour-flipped digits?
# =====================================================================
def find_flips(model, split, device):
    """Digits classified correctly in their natural colour but wrongly when the colour is
    reversed ("colour-flipped"). Returns (bool mask, pred natural, pred reversed)."""
    y = split[1]
    pred_nat = predict_logits(model, split, device, transform=natural_color).argmax(dim=1)
    pred_rev = predict_logits(model, split, device, transform=anti_color).argmax(dim=1)
    flipped = (pred_nat == y) & (pred_rev != y)
    return flipped, pred_nat, pred_rev


def plot_flips(nat, rev_true, rev_wrong, y, wrong, path, title):
    """6 rows x n columns. nat / rev_true / rev_wrong are `attribute` outputs for the same
    digits: natural colour explaining the true class, reversed colour explaining the true
    class, reversed colour explaining the (wrong) predicted class."""
    cols = range(len(y))
    ov = lambda m: [_overlay(m["img"][i], m["cam"][i]) for i in cols]       # noqa: E731
    pick = lambda m, k: [m[k][i] for i in cols]                              # noqa: E731
    rows = [(pick(nat, "img"), None), (ov(nat), None),
            (pick(rev_true, "img"), None), (ov(rev_true), None),
            (ov(rev_wrong), None), (pick(rev_wrong, "gxi"), "hot")]
    labels = ["natural: image", "natural: Grad-CAM\n(true class)",
              "reversed: image", "reversed: Grad-CAM\n(true class)",
              "reversed: Grad-CAM\n(wrong class \u0177)", "reversed: Grad\u00d7Input\n(wrong class \u0177)"]
    titles = {0: [f"true {y[i]}" for i in cols], 4: [f"\u0177 = {wrong[i]}" for i in cols]}
    _plot_grid(rows, labels, titles, path, title)


# =====================================================================
# Task 3.4: misspecification tests (change the shape cue and the colour cue)
# =====================================================================
def rotate90(x, y, c):
    """Shape perturbation: rotate the digit by 90 degrees."""
    return torch.rot90(x, 1, dims=(2, 3))


def occlude_right_half(x, y, c):
    """Shape perturbation: delete the right half of the image (part of the digit)."""
    x = x.clone()
    x[:, :, :, 14:] = 0
    return x


SHAPE_CONDITIONS = {"clean": None, "rot90": rotate90, "occlude_right_half": occlude_right_half}
COLOUR_CONDITIONS = {"natural": natural_color, "reversed": anti_color, "grayscale": grayscale_max}


def misspecification_tests(model, split, device, n=1000, seed=0):
    """Accuracy for every (shape perturbation) x (colour condition) on n random test digits.

    For natural / reversed colour we also report `follows_colour`: how often the predicted
    group (0-4 vs 5-9) is the group the colour points to (green -> 5-9, red -> 0-4).
    ~0.5 = the colour has no pull; ~1.0 = the model obeys the colour.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(split[1]), generator=g)[:n]
    sub = (split[0][idx], split[1][idx], split[2][idx])
    y = sub[1]
    res = {"n": int(len(y))}
    for sname, stf in SHAPE_CONDITIONS.items():
        for cname, ctf in COLOUR_CONDITIONS.items():
            def tf(x, yy, c, stf=stf, ctf=ctf):
                if stf is not None:
                    x = stf(x, yy, c)          # destroy / alter the shape first ...
                return ctf(x, yy, c)           # ... then re-paint the colour
            pred = predict_logits(model, sub, device, transform=tf).argmax(dim=1)
            entry = {"acc": (pred == y).float().mean().item()}
            if cname != "grayscale":
                shown_green = (y >= 5) if cname == "natural" else (y < 5)
                entry["follows_colour"] = ((pred >= 5) == shown_green).float().mean().item()
            res[f"{sname}/{cname}"] = entry
    return res


def plot_misspec_examples(split, path, n=8):
    """Visual check: first n digits under each shape perturbation, natural and reversed colour."""
    x = split[0][:n].float().div(255.0)
    y, c = split[1][:n], split[2][:n]
    rows, labels = [], []
    for sname, stf in SHAPE_CONDITIONS.items():
        xs = x if stf is None else stf(x, y, c)
        for cname in ("natural", "reversed"):
            xc = COLOUR_CONDITIONS[cname](xs, y, c)
            rows.append(([xc[i].permute(1, 2, 0).clamp(0, 1).numpy() for i in range(n)], None))
            labels.append(f"{sname}\n{cname}")
    _plot_grid(rows, labels, {0: [f"label {int(y[i])}" for i in range(n)]}, path,
               "Misspecification tests: shape perturbation x colour", cell=1.0)