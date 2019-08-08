import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import *
from visualize import *


latent_size = 10
data_size = 28 * 28
lr = 1e-4
num_epochs = 100
m = 64
save_iter = 10

# get images from MNIST database
dataset = MNIST('data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=m, shuffle=True)

# create GAN and optimizers for it
G = Net(type='G', latent_size=latent_size, data_size=data_size)
D = Net(type='D', latent_size=latent_size, data_size=data_size)

# start training
for epoch in range(num_epochs):

    # get minibatches of size m from dataset
    for img, _ in dataloader:

        # gradient ascent for discriminator
        x = img.view(img.size(0), -1)                                           # reshape image to be 1D
        z = torch.randn(img.size(0), latent_size)                               # samples of noise
        d_loss = (torch.log(D(x)) + torch.log(1 - D(G(z)))).mean()              # L = E[ log D(x) - log(1 - D(G(z))) ]
        D.optimize(-d_loss)                                                     # gradient ASCENT on L

        # gradient ascent for generator
        z = torch.randn(img.size(0), latent_size)                               # new samples of noise
        g_loss = torch.log(1 - D(G(z))).mean()                                  # L = E[ log(1 - D(G(z))) ]
        G.optimize(g_loss)                                                      # gradient DESCENT on L

    # plot loss
    update_viz(epoch, d_loss.item(), g_loss.item())

    # save images periodically
    if epoch % save_iter == save_iter - 1:
        z = torch.randn(96, latent_size)
        # z = torch.linspace(0, 1, 96).unsqueeze(1).repeat(1, latent_size)        # interpolate over latent space
        img = G(z).view(96, 1, 28, 28)                                          # format images into correct dimensions
        save_image(img, './img/' + str(epoch + 1) + '_epochs.png')