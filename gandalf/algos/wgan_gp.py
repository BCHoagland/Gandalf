import torch
import torch.nn as nn
from torch.autograd import grad

from gandalf.algos.base import Algo

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class WGAN_GP(Algo):

    def optimize_D(self, G, D, x):
        # network inputs
        z = self.noise()
        x_gen = G(z)
        x_inter = self.interpolate(x, x_gen)

        # gradient penalty calculation
        out = D(x_inter)
        grads = grad(out, x_inter, torch.ones(out.shape).to(device), create_graph=True)[0]
        grad_penalty = self.λ * ((self.norm(grads) - 1) ** 2).mean()

        # optimization
        wasserstein = D(x).mean() - D(x_gen).mean()
        d_loss = wasserstein - grad_penalty
        D.maximize(d_loss)

    def optimize_G(self, G, D):
        z = self.noise()
        g_loss = -D(G(z)).mean()
        G.minimize(g_loss)
