import torch
import random
import torch.optim as optim
from collections import deque
from Model import Model
import numpy as np
import math

class DeepQLearning:
    def __init__(self, env, gamma, epsilon, numberEpisodes, maxMemorySize, render=True, min_epsilon=0.01, epsilon_decay=0.001, decay_rate=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_rate = decay_rate
        self.numberEpisodes = numberEpisodes
        self.stateSize = env.observation_space.shape[0]
        self.outputLayerSize = env.action_space.n
        self.sumRewardsEpisode = []
        self.experienceStorage = deque(maxlen=maxMemorySize)
        self.render = render
        
        # Initialize the PyTorch model and optimizer
        self.model = Model(self.stateSize, self.outputLayerSize)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.lossFunction = torch.nn.MSELoss()

    def train(self):
        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode = []
            print("Simulating episode {}".format(indexEpisode))

            currentState = self.env.reset()
            currentState = torch.tensor(currentState[0], dtype=torch.float32)

            done = False
            while not done:
                if self.render:
                    self.env.render()
                
                action = self.chooseAction(currentState)
                nextState, reward, done, _, _ = self.env.step(action)
                nextState = torch.tensor(nextState, dtype=torch.float32)
                rewardsEpisode.append(reward)

                # Add current state, action, reward, next state, and terminal flag to the experience storage medium
                self.experienceStorage.append((currentState, action, reward, nextState, done))

                # Train the ANN
                self.updateModel()

                # Set the current state for the next step
                currentState = nextState

            # Epsilon decay after each episode
            self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * math.exp(-self.decay_rate * indexEpisode)

            print("Sum of rewards {}: {}".format(indexEpisode, sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(sum(rewardsEpisode))
        self.env.close()

    def chooseAction(self, state):
        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            stateTensor = state.unsqueeze(0) if len(state.shape) == 1 else state
            Qvalues = self.model(stateTensor)
            return torch.argmax(Qvalues, dim=1).item()

    def updateModel(self, batchSize=50):
        if len(self.experienceStorage) < batchSize:
            return

        minibatch = random.sample(self.experienceStorage, batchSize)
        for state, action, reward, nextState, done in minibatch:
            # Ensure state and nextState are tensors with a batch dimension
            stateTensor = state.unsqueeze(0) if len(state.shape) == 1 else state
            nextStateTensor = nextState.unsqueeze(0) if len(nextState.shape) == 1 else nextState

            # Calculate target
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(nextStateTensor)).item()

            # Predict the current Q-values
            currentQValues = self.model(stateTensor)

            # Clone the current Q-values and update the value for the action taken
            targetQValues = currentQValues.clone()
            targetQValues[0][action] = target

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss = self.lossFunction(currentQValues, targetQValues)
            loss.backward()
            self.optimizer.step()