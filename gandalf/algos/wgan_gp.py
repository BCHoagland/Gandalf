import torch
import torch.nn as nn
from torch.autograd import grad

from gandalf.algos.base import Algo

class WGAN_GP(Algo):
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

    def get_stats(self):
        return ([self.wasserstein, self.grad_penalty], 'Wasserstein Estimate', ['Distance', 'Grad Penalty'])

    def optimize_D(self, x):
        # network inputs
        z = self.noise()
        x_gen = self.G(z)
        x_inter = self.interpolate(x, x_gen)

        # gradient penalty calculation
        out = self.D(x_inter)
        grads = grad(out, x_inter, torch.ones(out.shape).to('cuda:0'), create_graph=True)[0]
        self.grad_penalty = self.config.λ * ((self.norm(grads) - 1) ** 2).mean()

        # optimization
        self.wasserstein = self.D(x).mean() - self.D(x_gen).mean()
        d_loss = self.wasserstein - self.grad_penalty
        self.D.maximize(d_loss)

    def optimize_G(self):
        z = self.noise()
        g_loss = -self.D(self.G(z)).mean()
        self.G.minimize(g_loss)