"""Task 4: bias-mitigation training procedures.

Every method uses exactly the baseline's setup (SimpleCNN, Adam, lr 1e-3, batch 64,
20 epochs, final-epoch weights) and only changes the training loss. With the same seed,
a method therefore starts from the same initial weights and sees the same batch order as
the baseline, so a (baseline, method) pair differs only by the method itself.

Final-epoch weights are used for every method: choosing an epoch on the OOD sets would be
cheating, and the validation set (same rho as training) cannot reveal the shortcut.
"""
import torch
import torch.nn.functional as F

from .data import batches
from .evaluation import evaluate
from .models import SimpleCNN
from .utils import set_seed


def fit(model, train_set, val_set, device, epochs, bs, lr, seed, step_fn, tag):
    """Generic Adam loop. step_fn(model, x, y, c) -> scalar training loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        for xb, yb, cb in batches(train_set, bs, shuffle=True, device=device, generator=gen):
            opt.zero_grad()
            loss = step_fn(model, xb, yb, cb)
            loss.backward()
            opt.step()
        tl, ta = evaluate(model, train_set, device)     # plain CE, dropout off
        vl, va = evaluate(model, val_set, device)
        for key, value in zip(hist, (tl, ta, vl, va)):
            hist[key].append(value)
        print(f"[{tag}] epoch {ep + 1:02d}/{epochs}  train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f}", flush=True)
    return hist


# ---------------------------------------------------------------- reweighting
def sample_weight_table(train_set):
    """(10, 2) table with w[digit, colour] = 1 / P(colour | digit), estimated on the
    training set and rescaled so that the average training sample has weight 1.
    Rare (conflicting) combinations get large weights, common ones small weights."""
    _, y, c = train_set
    counts = torch.bincount(y * 2 + c, minlength=20).view(10, 2).float().clamp(min=1)
    p = counts / counts.sum(dim=1, keepdim=True)         # P(colour | digit)
    w = 1.0 / p
    return w / w[y, c].mean()


def train_reweighting(train_set, val_set, device, seed, epochs, bs, lr):
    set_seed(seed)
    model = SimpleCNN().to(device)
    table = sample_weight_table(train_set)
    _, y, c = train_set
    aligned = c == (y >= 5).long()
    w_all = table[y, c]
    info = {"mean_weight_aligned": w_all[aligned].mean().item(),
            "mean_weight_conflicting": w_all[~aligned].mean().item()}
    print(f"[reweighting] mean weight: aligned {info['mean_weight_aligned']:.3f}, "
          f"conflicting {info['mean_weight_conflicting']:.1f}")
    table = table.to(device)

    def step(m, x, yb, cb):
        ce = F.cross_entropy(m(x), yb, reduction="none")
        return (table[yb, cb] * ce).mean()

    hist = fit(model, train_set, val_set, device, epochs, bs, lr, seed, step, "reweighting")
    hist.update(info)
    return model, hist


# ------------------------------------------------------------------ Group DRO
def train_group_dro(train_set, val_set, device, seed, epochs, bs, lr, eta=0.01):
    """Online Group DRO (Sagawa et al., 2020).
    Four groups g = 2*colour + (digit >= 5): red/0-4, red/5-9, green/0-4, green/5-9.
    Each step: q_g <- q_g * exp(eta * loss_g) for the groups present in the batch, q is
    renormalised, and the model minimises sum_g q_g * loss_g, i.e. it is pushed toward the
    group it currently fits worst."""
    set_seed(seed)
    model = SimpleCNN().to(device)
    q = torch.ones(4, device=device) / 4

    def step(m, x, yb, cb):
        losses = F.cross_entropy(m(x), yb, reduction="none")
        grp = cb * 2 + (yb >= 5).long()
        onehot = F.one_hot(grp, 4).float()
        counts = onehot.sum(dim=0)
        group_loss = (onehot * losses[:, None]).sum(dim=0) / counts.clamp(min=1)
        present = counts > 0
        with torch.no_grad():
            q[present] = q[present] * torch.exp(eta * group_loss[present])
            q.div_(q.sum())
        return (q * group_loss).sum()

    hist = fit(model, train_set, val_set, device, epochs, bs, lr, seed, step, "group_dro")
    hist["final_q"] = q.tolist()
    print("[group_dro] final group weights q (red/0-4, red/5-9, green/0-4, green/5-9):",
          [round(v, 3) for v in hist["final_q"]])
    return model, hist


METHODS = {"reweighting": train_reweighting, "group_dro": train_group_dro}