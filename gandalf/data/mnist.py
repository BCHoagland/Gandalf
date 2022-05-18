import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from pathlib import Path


class MNIST:
    data_size = 28 * 28

    def __init__(self, config):
        self.config = config

        dataset = datasets.MNIST('gandalf/data', transform=transforms.ToTensor(), download=True)
        #! remove this eventually
        N = 5000
        dataset, _ = torch.utils.data.random_split(dataset, [N, 60000-N])
        self.dataloader = DataLoader(dataset, batch_size=config.m, shuffle=True)

    def batches(self):
        for img, _ in self.dataloader:
            yield img.view(img.size(0), -1)
    
    # def num_batches(self):
    #     return len(self.dataloader)

    def save(self, epoch, G):
        z = torch.randn(96, self.config.latent_size)                                                                    # TODO: maybe interpolate instead
        # z = torch.linspace(0, 1, 96).unsqueeze(1).repeat(1, self.config.latent_size)
        img = G(z).view(96, 1, 28, 28)

        path = f'img/MNIST/{self.config.algo}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        save_image(img, path + str(epoch + 1) + '_epochs.png')
