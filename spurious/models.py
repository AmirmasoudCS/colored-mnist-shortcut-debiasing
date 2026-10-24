"""The CNN from the handout.

The forward pass is split into features() and head() so that Grad-CAM (Step 4)
can take gradients with respect to the last convolutional feature map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        return F.relu(self.conv3(x))          # 7x7  (Grad-CAM target)

    def head(self, f):
        x = self.pool(f)                      # 7x7 -> 3x3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    # quick sanity check:  python -m spurious.models
    model = SimpleCNN()
    out = model(torch.randn(2, 3, 28, 28))
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", tuple(out.shape), "(expected (2, 10))")
    print("parameters:", n_params, "(expected 390986)")