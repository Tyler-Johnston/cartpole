from DeepQLearning import DeepQLearning
import gym
import torch
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="human")

# Select the parameters
gamma = 1
epsilon = 0.1
numberEpisodes = 250
maxMemorySize = 50000

# Create an object
LearningQDeep = DeepQLearning(env, gamma, epsilon, numberEpisodes, maxMemorySize)
# Run the learning process
LearningQDeep.train()

# Get the obtained rewards in every episode
sumRewards = LearningQDeep.sumRewardsEpisode
print("Obtained Rewards:", sumRewards)

# Plotting the rewards
plt.plot(sumRewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
