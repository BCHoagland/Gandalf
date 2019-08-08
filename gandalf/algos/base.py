import torch

class Algo:
    # def __init__(self, latent_size, data_size, m, n_hidden):
    def __init__(self, config):
        self.config = config
        # self.latent_size = latent_size
        # self.data_size = data_size
        # self.m = m
        # self.n_hidden = n_hidden

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