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


# --------------------------------------------------- Learning from Failure (LfF)
def train_lff(train_set, val_set, device, seed, epochs, bs, lr, q=0.7, ema_alpha=0.7):
    """Learning from Failure (Nam et al., 2020) - needs NO bias (colour) labels.

    Idea: a model that is trained to be *as biased as possible* learns the easy shortcut
    (colour) first. The samples it gets WRONG are exactly the ones the shortcut does not
    explain (conflicting samples). A second, "debiased" model then focuses on those samples.

    Two models are trained at the same time on the same batches:

    1. BIASED model f_b, trained with the Generalised Cross-Entropy (GCE) loss
           GCE = (1 - p_y**q) / q      (p_y = probability given to the true class, q = 0.7)
       Its gradient is p_y**q times the usual cross-entropy gradient, so samples the model
       already gets right count MORE and hard samples count LESS. This amplifies whatever
       is easy to learn: the colour.

    2. DEBIASED model f_d, trained with cross-entropy where each sample is multiplied by its
       relative-difficulty score
           W(x) = CE_b(x) / (CE_b(x) + CE_d(x))
       W is close to 1 when the biased model fails on x (large CE_b) while f_d does better,
       and close to 0 when the biased model already handles x easily.
       Conflicting samples (colour misleading) are exactly those where the biased model
       fails, so they receive large weights - without ever being told which they are.

    Two stabilising details from the paper:
      * CE_b and CE_d are replaced by an exponential moving average (EMA) of each sample's
        loss over the epochs (ema = alpha * ema + (1 - alpha) * CE): a single noisy step
        should not decide a weight.
      * Each EMA is divided by the largest EMA of the same digit class, so that classes that
        are intrinsically harder (like 8) are not mistaken for 'bias-conflicting'.

    The colour label is used ONLY to log the diagnostic 'mean W on aligned vs conflicting
    samples' (it should show conflicting samples receiving clearly higher weights); it
    never enters the training loss.
    The returned model is the debiased one; the biased one is returned inside the history
    under '_biased_state' for diagnostics.
    """
    set_seed(seed)
    model_d = SimpleCNN().to(device)     # created first: same initial weights as the baseline
    model_b = SimpleCNN().to(device)
    opt_d = torch.optim.Adam(model_d.parameters(), lr=lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr)
    y_all = train_set[1].to(device)
    n = len(y_all)
    ema_b = torch.zeros(n, device=device)    # per-sample loss history of the biased model
    ema_d = torch.zeros(n, device=device)    # per-sample loss history of the debiased model
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "biased_val_acc": [], "mean_W_aligned": [], "mean_W_conflicting": []}

    for ep in range(epochs):
        model_b.train()
        model_d.train()
        w_sum = torch.zeros(2, device=device)      # [aligned, conflicting] diagnostics only
        w_cnt = torch.zeros(2, device=device)
        for xb, yb, cb, ib in batches(train_set, bs, shuffle=True, device=device,
                                      generator=gen, return_index=True):
            logit_b, logit_d = model_b(xb), model_d(xb)
            ce_b = F.cross_entropy(logit_b, yb, reduction="none")
            ce_d = F.cross_entropy(logit_d, yb, reduction="none")

            with torch.no_grad():                   # --- relative difficulty W(x) ---
                ema_b[ib] = ema_alpha * ema_b[ib] + (1 - ema_alpha) * ce_b
                ema_d[ib] = ema_alpha * ema_d[ib] + (1 - ema_alpha) * ce_d
                max_b = torch.zeros(10, device=device).scatter_reduce(0, y_all, ema_b, reduce="amax")
                max_d = torch.zeros(10, device=device).scatter_reduce(0, y_all, ema_d, reduce="amax")
                nb = ema_b[ib] / (max_b[yb] + 1e-8)  # class-wise max normalisation
                nd = ema_d[ib] / (max_d[yb] + 1e-8)
                w = nb / (nb + nd + 1e-8)
                al = cb == (yb >= 5).long()          # diagnostics only (uses the colour label)
                w_sum[0] += w[al].sum()
                w_sum[1] += w[~al].sum()
                w_cnt[0] += al.sum()
                w_cnt[1] += (~al).sum()

            loss_d = (ce_d * w).mean()               # debiased model: weighted cross-entropy
            p_b = F.softmax(logit_b, dim=1).gather(1, yb[:, None]).squeeze(1).clamp_min(1e-8)
            loss_b = ((1.0 - p_b ** q) / q).mean()   # biased model: GCE

            opt_b.zero_grad()
            loss_b.backward()
            opt_b.step()
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        tl, ta = evaluate(model_d, train_set, device)
        vl, va = evaluate(model_d, val_set, device)
        bva = evaluate(model_b, val_set, device)[1]
        wa, wc = (w_sum / w_cnt.clamp(min=1)).tolist()
        for key, value in zip(hist, (tl, ta, vl, va, bva, wa, wc)):
            hist[key].append(value)
        print(f"[lff] epoch {ep + 1:02d}/{epochs}  debiased: train acc {ta:.4f} val acc {va:.4f} | "
              f"biased val acc {bva:.4f} | mean W aligned {wa:.3f} conflicting {wc:.3f}", flush=True)

    hist["_biased_state"] = {k: v.cpu() for k, v in model_b.state_dict().items()}
    return model_d, hist


METHODS = {"reweighting": train_reweighting, "group_dro": train_group_dro, "lff": train_lff}