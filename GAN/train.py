import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np

from model import *
from visualize import *

latent_size = 64
lr = 1e-4
num_epochs = 100
m = 128
save_iter = 10

# get images from MNIST database
dataset = MNIST('data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=m, shuffle=True)

# create GAN and optimizers for it
G = Generator(latent_size)
D = Discriminator()
g_optimizer = optim.Adam(G.parameters(), lr=lr)
d_optimizer = optim.Adam(D.parameters(), lr=lr)

# start training
for epoch in range(num_epochs):

    # get minibatches of size m from dataset
    for img, labels in dataloader:

        # resize each image to 1D
        img = img.view(img.size(0), -1)

        # generate m examples of noise to train the discriminator
        z = torch.FloatTensor(np.random.normal(0, 1, (img.size(0), latent_size)))

        # gradient ascent for discriminator
        # loss = avg of [log(D(x))  + log(1 - D(G(z)))]
        # where D(x) is P(x came from real data)
        d_loss = -(torch.log(D(img)) + torch.log(1 - D(G(z)))).mean()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # generate another m examples of noise to train the generator
        z = torch.FloatTensor(np.random.normal(0, 1, (img.size(0), latent_size)))

        # gradient descent for generator
        # loss = avg of [log(1 - D(G(z)))]
        g_loss = torch.log(1 - D(G(z))).mean()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # plot loss
    update_viz(epoch, d_loss.item(), g_loss.item())

    # save images periodically
    if epoch % save_iter == save_iter - 1:
        z = torch.FloatTensor(np.random.normal(0, 1, (96, latent_size)))
        img = G(z).view(96, 1, 28, 28)
        save_image(img, './img/' + str(epoch + 1) + '_epochs.png')
