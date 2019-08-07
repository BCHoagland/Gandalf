import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x.to(device))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x.to(device))
