import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, stateSize, outputLayerSize):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(stateSize, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, outputLayerSize)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
