"""
CNN classifier for CIFAR10 - vulnerable to membership inference attacks

This classifier uses a CNN architecture:
- Two convolution and max pooling layers
- A fully connected layer of size 128
- A SoftMax layer (output)
- Activation function: Tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """
    CNN classifier for CIFAR10

    Architecture:
    - Conv1: 3 -> 32 channels, 3x3 kernel
    - Tanh activation
    - MaxPool: 2x2
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - Tanh activation
    - MaxPool: 2x2
    - Flatten: 64 * 8 * 8 = 4096
    - FC1: 4096 -> 128
    - Tanh activation
    - FC2: 128 -> 10 (output)
    """
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling layers: 32x32 -> 16x16 -> 8x8
        # 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images, shape (batch_size, 3, 32, 32)

        Returns:
            logits: Raw logits, shape (batch_size, 10)
        """
        # First conv + pool block
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool(x)  # 32x32 -> 16x16

        # Second conv + pool block
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool(x)  # 16x16 -> 8x8

        # Flatten
        x = x.view(-1, 64 * 8 * 8)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.tanh(x)
        logits = self.fc2(x)

        return logits
