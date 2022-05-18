import torch
import torch.nn as nn

from gandalf.algos.base import Algo


class GAN(Algo):

    def optimize_D(self, G, D, x):
        z = self.noise()
        d_loss = (torch.log(D(x)) + torch.log(1 - D(G(z)))).mean()
        D.maximize(d_loss)

    def optimize_G(self, G, D):
        z = self.noise()
        g_loss = torch.log(D(G(z))).mean()                                           # w/ trick
        G.maximize(g_loss)
        # g_loss = torch.log(1 - D(G(z))).mean()                                       # w/out trick
        # G.minimize(g_loss)
