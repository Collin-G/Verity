import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling to 1x1 output
        self.fc1 = nn.Linear(64, 128)  # Input size now corresponds to 64 (channels) after avg pooling
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Define Leaky ReLU

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))  # Use Leaky ReLU
        x = self.pool(self.leaky_relu(self.conv2(x)))  # Use Leaky ReLU
        x = self.avg_pool(x)  # Output: (batch_size, 64, 1, 1)
        x = x.view(-1, 64)  # Flatten the tensor
        x = self.leaky_relu(self.fc1(x))  # Use Leaky ReLU for the first fully connected layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary output
        return x