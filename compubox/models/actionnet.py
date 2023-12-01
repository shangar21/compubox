import torch.nn as nn
import torch.nn.functional as F
import torch

class ActionNet(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers = 2, num_actions=13):
        super(ActionNet, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_layers = num_actions

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = torch.nn.Linear(hidden_size * 2, num_actions)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers + 1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers + 1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)
        return out
