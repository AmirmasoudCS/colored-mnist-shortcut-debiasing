import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rho_tag(rho):
    """'' for the handout setting (rho = 0.9) so old file names stay unchanged,
    otherwise '_rho0.99' etc."""
    return "" if abs(rho - 0.9) < 1e-9 else f"_rho{rho:g}"


def model_name(rho, seed):
    """baseline_model.pth for (rho 0.9, seed 0); baseline_model_rho0.99_seed1.pth otherwise."""
    tag = rho_tag(rho)
    if tag == "" and seed == 0:
        return "baseline_model.pth"
    return f"baseline_model{tag}_seed{seed}.pth"