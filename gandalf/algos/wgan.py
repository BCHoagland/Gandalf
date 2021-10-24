import torch
import torch.nn as nn

from gandalf.algos.base import Algo


class WGAN(Algo):

    def optimize_D(self, G, D, x):
        z = self.noise()
        d_loss = D(x).mean() - D(G(z)).mean()
        D.maximize(d_loss)

        for p in D.parameters():
            p.data.clamp_(-self.clip, self.clip)

    def optimize_G(self, G, D):
        z = self.noise()
        g_loss = -D(G(z)).mean()
        G.minimize(g_loss)
