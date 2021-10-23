import torch
import torch.nn as nn

from gandalf.algos.base import Algo

class WGAN(Algo):
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
            nn.Linear(self.n_hidden, 1)
        )
        
        return G, D

    def optimize_D(self, x):
        z = self.noise()
        self.d_loss = self.D(x).mean() - self.D(self.G(z)).mean()
        self.D.maximize(self.d_loss)

        for p in self.D.parameters():
            p.data.clamp_(-self.clip, self.clip)

    def optimize_G(self):
        z = self.noise()
        g_loss = -self.D(self.G(z)).mean()
        self.G.minimize(g_loss)