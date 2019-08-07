import torch
import torch.nn as nn

from utils import get_device


class Net(nn.Module):
    def __init__(self, type, latent_size, data_size, n_hidden=128, lr=1e-4):
        super().__init__()

        if type is 'G':
            self.main = nn.Sequential(
                nn.Linear(latent_size, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, data_size)
            )
        elif type is 'D':
            self.main = nn.Sequential(
                nn.Linear(data_size, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError

        self.to(get_device())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.FloatTensor(x)
        return self.main(x.to(get_device()))

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()