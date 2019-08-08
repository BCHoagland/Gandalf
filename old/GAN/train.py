import torch
from model import Net
from visualize import plot, hist


epochs = 5e3
vis_iter = 100

latent_size = 10
data_size = 1
m = 100

G = Net(type='G', latent_size=latent_size, data_size=data_size)
D = Net(type='D', latent_size=latent_size, data_size=data_size)


def noise(batch_size=m):
    return torch.randn(batch_size, latent_size)

def sample_data(batch_size=m):
    return torch.randn(batch_size, data_size) * 5 + 100

for epoch in range(int(epochs)):

    # optimize discriminator
    z = noise()
    x = sample_data()
    d_loss = (torch.log(D(x)) + torch.log(1 - D(G(z)))).mean()
    D.optimize(-d_loss)

    # optimize generator
    z = noise()
    g_loss = torch.log(D(G(z))).mean()
    G.optimize(-g_loss)

    # visualize progress occasionally
    if epoch % vis_iter == vis_iter - 1:

        # plot loss
        plot(epoch, d_loss, g_loss)

        # draw histograms of real and generated data
        with torch.no_grad():
            z, x = noise(100), sample_data(100)
            hist(x, 'Real')
            hist(G(z), 'Generated')