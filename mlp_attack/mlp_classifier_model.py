"""
MLP classifier for CIFAR10 - vulnerable to membership inference attacks

This classifier trains directly on raw pixels (3072 dimensions for CIFAR10).
This makes it more vulnerable to membership inference because:
1. No information bottleneck
2. Higher capacity to memorize training data
3. Direct gradient flow from labels to inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Simple MLP classifier trained directly on raw CIFAR10 pixels

    This is intentionally designed to be vulnerable to membership inference:
    - Large hidden layers (more memorization capacity)
    - No bottleneck
    - Direct training on high-dimensional inputs
    """
    def __init__(self, hidden_dim=512):
        super(MLPClassifier, self).__init__()

        # Network architecture: 3072 -> hidden_dim -> 256 -> 10
        self.fc1 = nn.Linear(3072, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images, shape (batch_size, 3072)

        Returns:
            logits: Raw logits, shape (batch_size, 10)
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)

        return logits
