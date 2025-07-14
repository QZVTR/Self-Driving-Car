import pygame
import matplotlib.pyplot as plt
from environment import Environment, clock
from agent import DQNAgent
import torch

env = Environment()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
stateSize = 9 #15 if using just above and side # [lane_onehot (3), dists (3), speeds(3)]
actionSize = 3 # stay, left, right 
agent = DQNAgent(stateSize, actionSize, device)
#agent.loadModel('dqn_model.pth')
episodes = 1000
rewards = []


for episode in range(episodes):
    state = env.reset()
    totalReward = 0
    done = False 
    while not done:
        action = agent.selectAction(state)
        nextState, reward, done = env.step(action)
        agent.replayBuffer.push(state, action, reward, nextState, done)
        state = nextState 
        totalReward += reward 
        agent.train()
        env.render(episode + 1, totalReward) # Comment to not see sim
        clock.tick(60)
    
    rewards.append(totalReward)

    if episode % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {totalReward}")


# Save the model
agent.saveModel('dqn_model.pth')
print("Model saved to dqn_model.pth")

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

env.close()
