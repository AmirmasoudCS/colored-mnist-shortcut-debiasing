import torch
import torch.nn.functional as F

from .data import batches


@torch.no_grad()
def evaluate(model, split, device, bs=2000):
    """Return (mean cross-entropy loss, accuracy) of `model` on a (x, y, c) split.

    The model is put in eval mode (dropout off). Callers that continue training
    must call model.train() again.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb, _ in batches(split, bs, shuffle=False, device=device):
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += yb.shape[0]
    return total_loss / n, correct / n