import torch
from gandalf.visualize import hist

class Gaussian:
    data_size = 1

    def __init__(self, config):
        self.config = config

    def batches(self):
        yield torch.randn(self.config.m, self.data_size) * 5 + 100

    def save(self, epoch, G):
        pass