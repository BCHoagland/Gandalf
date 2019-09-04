import torch
from gandalf.visualize import hist

class MultiGaussian:
    data_size = 1

    def __init__(self, config):
        self.config = config

    def batches(self):
        batch_size = self.config.m // 2
        d1 = torch.randn(batch_size, self.data_size) * 5 + 100
        d2 = torch.randn(batch_size, self.data_size) * 2 + 20
        yield torch.cat((d1, d2), 0)

    def save(self, epoch, G):
        d1 = torch.randn(50, self.data_size) * 5 + 100
        d2 = torch.randn(50, self.data_size) * 2 + 20
        data = torch.cat((d1, d2), 0)

        z = torch.randn(100, self.config.latent_size)

        hist(G(z), 'Generated')
        hist(data, 'Real')