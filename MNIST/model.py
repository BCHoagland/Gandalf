import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, type, latent_size, data_size, n_hidden=256, lr=1e-4):
        super().__init__()

        if type is 'G':
            self.main = nn.Sequential(
                nn.Linear(latent_size, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, data_size),
                nn.Sigmoid()
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

        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.FloatTensor(x)
        return self.main(x.to(device))

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()