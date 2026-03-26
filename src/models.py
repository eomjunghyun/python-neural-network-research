import torch.nn as nn


class MLP_L3_Tanh_NoBN(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.readout = nn.Linear(bottleneck_dim, 1, bias=False)

    def forward(self, x):
        h = self.encoder(x)
        y_hat = self.readout(h)
        return y_hat, h


class MLP_L3_ReLU_BN(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
        )
        self.readout = nn.Linear(bottleneck_dim, 1, bias=False)

    def forward(self, x):
        h = self.encoder(x)
        y_hat = self.readout(h)
        return y_hat, h
