from gandalf import data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import gandalf


n_h = 128
n_latent = 10


class AE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(28 * 28, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_latent)
        )

        self.dec = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, 28 * 28)
        )
    
    def encode(self, x):
        return self.enc(x)
    
    def decode(self, x):
        return self.dec(x)
    
    def forward(self, x):
        return self.dec(self.enc(x))


class Config:
    m = 128
    latent_size = n_latent


def tsne_vis(pts):
    #* take batch data, compress to 2D, and plot
    with torch.no_grad():
        pts = pts.numpy()
        pts = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(pts)
        x_coords = pts[:,0]
        y_coords = pts[:,1]
        plt.scatter(x_coords, y_coords)
        plt.show()


def show_first_img(images):
    #* take first image from batch and show it
    with torch.no_grad():
        img = images.view(Config.m, 1, 28, 28).numpy()[0, 0, :, :]
        print(img.shape)
        plt.imshow(img)
        plt.show()

dataloader = gandalf.data.MNIST(Config)
model = AE()

for x in dataloader.batches():
    show_first_img(x)
    z = model.encode(x)
    tsne_vis(z)

    out = model.decode(z)
    show_first_img(out)

    quit()
