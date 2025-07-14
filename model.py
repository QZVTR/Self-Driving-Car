import torch
import torch.nn as nn
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, stateSize, actionSize, device):
        super(QNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(stateSize, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, actionSize)
        self.to(device)  # Move model to GPU or CPU

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)