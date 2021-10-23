import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, model, lr):
        super().__init__()

        self.main = model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.FloatTensor(x)
        return self.main(x.to(device))

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def minimize(self, loss):
        self._optimize(loss)

    def maximize(self, loss):
        self._optimize(-loss)
