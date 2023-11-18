import torch.nn as nn
import torch.nn.functional as F
import torch

class ActionNet(nn.Module):
    def __init__(self, dimension, input_size, num_layers = 2, num_actions=13):
        super(ActionNet, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=dimension, num_layers=num_layers)
        self.fc = torch.nn.Linear(dimension * num_layers, dimension)
        self.fc2 = torch.nn.Linear(dimension, num_actions)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        hidden = torch.flatten(hidden)
        out = self.fc(hidden)
        out = F.relu(out)
        out = self.fc2(out)
        #out = F.sigmoid(out)
        out = F.relu(out)
        return out
