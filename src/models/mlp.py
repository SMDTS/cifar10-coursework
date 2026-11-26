"""Exercise 1: Multi-Layer Perceptron with three trainable (fully connected) layers."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim=3 * 32 * 32, hidden1=512, hidden2=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                      # 3x32x32 image -> 3072-dim vector
            nn.Linear(input_dim, hidden1),     # trainable layer 1
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),       # trainable layer 2
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),   # trainable layer 3 (logits; no softmax)
        )

    def forward(self, x):
        return self.net(x)