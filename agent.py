import torch
import torch.optim as optim
import numpy as np
from model import QNetwork, ReplayBuffer

class DQNAgent:
    def __init__(self, stateSize, actionSize, device):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.device = device
        self.qNetwork = QNetwork(stateSize, actionSize, device)
        self.targetNetwork = QNetwork(stateSize, actionSize, device)
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
        self.optimizer = optim.Adam(self.qNetwork.parameters(), lr=0.001)
        self.replayBuffer = ReplayBuffer(10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.995
        self.batchSize = 64
        self.updateTargetEvery = 100
        self.stepCounter = 0

    def selectAction(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actionSize)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qValues = self.qNetwork(state)
        return qValues.argmax().item()

    def train(self):
        if len(self.replayBuffer) < self.batchSize:
            return
        batch = self.replayBuffer.sample(self.batchSize)
        states, actions, rewards, nextStates, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        nextStates = torch.FloatTensor(np.array(nextStates)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        qValues = self.qNetwork(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        nextQValues = self.targetNetwork(nextStates).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * nextQValues
        loss = torch.nn.MSELoss()(qValues, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.stepCounter += 1
        if self.stepCounter % self.updateTargetEvery == 0:
            self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def saveModel(self, path):
        torch.save(self.qNetwork.state_dict(), path)

    def loadModel(self, path):
        self.qNetwork.load_state_dict(torch.load(path, map_location=self.device))
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())