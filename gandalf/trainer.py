import torch
from gandalf.model import Net
from gandalf.algos.gan import GAN

class Trainer:
    def __init__(self, algo, epochs=1e4, k=1, vis_iter=100, lr=1e-4):
        self.algo = algo
        self.epochs = epochs
        self.k = k
        self.vis_iter = vis_iter
        self.lr = lr

    def train(self):
        G, D = self.algo.models()
        G, D = Net(G, self.lr), Net(D, self.lr)
        self.algo.setup(G, D)

        for epoch in range(int(self.epochs)):
            for _ in range(self.k):
                self.algo.optimize_D()

            self.algo.optimize_G()

            if epoch % self.vis_iter == self.vis_iter - 1:
                with torch.no_grad():
                    self.algo.visualize(epoch)