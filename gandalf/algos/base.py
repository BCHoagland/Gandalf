import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Algo:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, k):
        return getattr(self.config, k)

    def noise(self, batch_size=None):
        if batch_size is None: batch_size = self.m
        return torch.randn(batch_size, self.latent_size)

    def sample_data(self, batch_size=None):
        if batch_size is None: batch_size = self.m
        return torch.randn(batch_size, self.data_size) * 5 + 100

    def interpolate(self, x1, x2):                                                          #TODO remove device calls when possible
        i = torch.rand_like(x1).to(device)
        return (i * x1.to(device)) + ((1 - i) * x2.to(device))

    def norm(self, x):                                                                      #* only Euclidian for now
        return torch.sqrt(torch.sum(x.pow(2), dim=-1) + 1e-10)
