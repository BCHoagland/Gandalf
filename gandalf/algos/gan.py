import torch
import torch.nn as nn

from gandalf.algos.base import Algo

class GAN(Algo):
    def models(self):
        G = nn.Sequential(
            nn.Linear(self.latent_size, self.n_hidden),
            nn.ELU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ELU(),
            nn.Linear(self.n_hidden, self.data_size)
        )

        D = nn.Sequential(
            nn.Linear(self.data_size, self.n_hidden),
            nn.ELU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ELU(),
            nn.Linear(self.n_hidden, 1),
            nn.Sigmoid()
        )
        
        return G, D

    def optimize_D(self, x):
        z = self.noise()
        self.d_loss = (torch.log(self.D(x)) + torch.log(1 - self.D(self.G(z)))).mean()
        self.D.maximize(self.d_loss)

    def optimize_G(self):
        z = self.noise()
        # self.g_loss = torch.log(self.D(self.G(z))).mean()                                           # w/ trick            # TODO: decide whether or not to keep
        # self.G.maximize(self.g_loss)
        self.g_loss = torch.log(1 - self.D(self.G(z))).mean()                                       # w/out trick
        self.G.minimize(self.g_loss)