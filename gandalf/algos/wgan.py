import torch
import torch.nn as nn

from gandalf.algos.base import Algo
from gandalf.visualize import plot, hist

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
        self.d_loss = ( self.D(x) - self.D(self.G(z)) ).mean()
        self.D.optimize(-self.d_loss)

        for p in self.D.parameters():
            p.data.clamp_(-self.clip, self.clip)

    def optimize_G(self):
        z = self.noise()
        self.g_loss = self.D(self.G(z)).mean()
        self.G.optimize(-self.g_loss)

    def visualize(self, epoch):
        plot(epoch, self.d_loss, 'Earth Mover Estimate', 'EM')

        # z = self.noise(100)
        # x = self.sample_data(100)
        # hist(x, 'Real')
        # hist(self.G(z), 'Generated')                                                      # TODO: decide whether or not to keep