import pygame
import matplotlib.pyplot as plt
import torch
from environment import Environment, clock
from agent import DQNAgent

env = Environment()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stateSize = 20 #[lane_onehot (3), dists_ahead (3), speeds_ahead (3), dists_behind (3), speeds_behind (3), dists_sides (2), speeds_sides (2)] #9  [lane_onehot (3), dists_ahead (3), speeds_ahead (3)]
actionSize = 5  # stay, left, right, up, down
agent = DQNAgent(stateSize, actionSize, device)

def train():
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
            env.render(episode + 1, totalReward)  # Comment to disable simulation rendering
            clock.tick(60)
        
        rewards.append(totalReward)
        if episode % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {totalReward}")

    # Save the model
    agent.saveModel('dqn_model_360.pth')
    print("Model saved to dqn_model.pth")

    # Plot training progress
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

def test():
    # Load the trained model
    try:
        agent.loadModel('dqn_model_360.pth')
        print("Loaded model from dqn_model_360.pth")
    except FileNotFoundError:
        print("Error: dqn_model.pth not found. Please train the model first.")
        env.close()
        return

    # Set epsilon to 0 for greedy policy (no exploration)
    agent.epsilon = 0.0

    # Run indefinitely until window is closed
    state = env.reset()
    totalReward = 0
    episode = 1
    running = True

    while running:
       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        
        action = agent.selectAction(state)
        nextState, reward, done = env.step(action)
        state = nextState
        totalReward += reward

     
        env.render(episode, totalReward)
        clock.tick(60)

     
        if done:
            state = env.reset()
            print(f"Episode {episode}, Total Reward: {totalReward}")
            episode += 1
            totalReward = 0

# Run testing (comment out to train instead)
test()

# Uncomment to train the model
#train()

# Cleanup
env.close()
