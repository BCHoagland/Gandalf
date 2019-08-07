import torch

from model import Net, Net
from utils import get_device
from visualize import plot, hist


epochs = 2e4
vis_iter = 100

latent_size = 10
data_size = 40
m = 100

G = Net(type='G', latent_size=latent_size, data_size=data_size)
D = Net(type='D', latent_size=latent_size, data_size=data_size)


def noise(batch_size=m):
    return torch.randn(batch_size, latent_size)


def sample_data(batch_size=m):
    shape = (batch_size // 2, data_size)
    a = torch.randn(shape) * 4 - 10
    b = torch.randn(shape) + 5
    c = torch.cat((a, b))
    return c


for epoch in range(int(epochs)):

    # optimize discriminator
    z, x = noise(), sample_data()
    d_loss = -( torch.log(D(x)) + torch.log(1 - D(G(z))) ).mean()
    D.optimize(d_loss)

    # optimize generator
    z = noise()
    g_loss = -torch.log(D(G(z))).mean()
    G.optimize(g_loss)

    # visualize progress occasionally
    if epoch % vis_iter == vis_iter - 1:

        # plot loss
        plot(epoch, -d_loss, -g_loss)

        # draw histograms of real and generated data
        with torch.no_grad():
            z, x = noise(100), sample_data(100)
            hist(x, 'Real')
            hist(G(z), 'Generated')