import torch
import torch.nn.functional as F

from .data import batches

GROUP_NAMES = ["red/0-4", "red/5-9", "green/0-4", "green/5-9"]  # group = 2*colour + (digit >= 5)


@torch.no_grad()
def predict_logits(model, split, device, bs=2000):
    """Logits (on CPU) for every sample of a (x, y, c) split. Model is set to eval mode."""
    model.eval()
    outs = []
    for xb, _, _ in batches(split, bs, shuffle=False, device=device):
        outs.append(model(xb).cpu())
    return torch.cat(outs)


def evaluate(model, split, device, bs=2000):
    """Return (mean cross-entropy loss, accuracy) on a (x, y, c) split."""
    logits = predict_logits(model, split, device, bs)
    y = split[1].cpu()
    return F.cross_entropy(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()


def _acc(correct, mask):
    return correct[mask].float().mean().item() if mask.any() else float("nan")


def evaluate_all(model, data, device):
    """Per set: accuracy, loss, accuracy on aligned vs conflicting samples, and the
    accuracy of the four (colour, digit-group) groups.

    aligned     = colour equals the natural colour of the digit (0-4 red, 5-9 green)
    conflicting = colour is the opposite one
    """
    results = {}
    for name, split in data.items():
        _, y, c = split
        y, c = y.cpu(), c.cpu()
        logits = predict_logits(model, split, device)
        correct = logits.argmax(dim=1) == y
        high = (y >= 5).long()
        aligned = c == high
        group = c * 2 + high
        results[name] = {
            "n": int(y.shape[0]),
            "acc": correct.float().mean().item(),
            "loss": F.cross_entropy(logits, y).item(),
            "n_aligned": int(aligned.sum()),
            "aligned_acc": _acc(correct, aligned),
            "conflicting_acc": _acc(correct, ~aligned),
            "group_acc": {g: _acc(correct, group == i) for i, g in enumerate(GROUP_NAMES)},
        }
    return results