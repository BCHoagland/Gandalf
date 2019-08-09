import torch

class Algo:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, k):
        return getattr(self.config, k)

    def setup(self, G, D):
        self.G = G
        self.D = D

    def noise(self, batch_size=None):
        if batch_size is None: batch_size = self.m
        return torch.randn(batch_size, self.latent_size)

    def sample_data(self, batch_size=None):
        if batch_size is None: batch_size = self.m
        return torch.randn(batch_size, self.data_size) * 5 + 100