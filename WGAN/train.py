import torch
from model import Net
from visualize import plot, hist


epochs = 5e3
vis_iter = 500

latent_size = 10
data_size = 1
m = 64
n = 5
lr = 5e-5
c = 0.01

g = Net(type='G', latent_size=latent_size, data_size=data_size, lr=lr)
f = Net(type='D', latent_size=latent_size, data_size=data_size, lr=lr)


def noise(batch_size=m):
    return torch.randn(batch_size, latent_size)

def sample_data(batch_size=m):
    return torch.randn(batch_size, data_size) * 5 + 100

for epoch in range(int(epochs)):

    # optimize discriminator
    for _ in range(n):
        z, x = noise(), sample_data()
        f_loss = ( f(x) - f(g(z)) ).mean()
        f.optimize(-f_loss)

        # clip discriminator weights
        for p in f.parameters():
            p.data.clamp_(-c, c)

    # optimize generator
    z = noise()
    g_loss = f(g(z)).mean()
    g.optimize(-g_loss)

    # visualize progress occasionally
    if epoch % vis_iter == vis_iter - 1:

        # plot loss
        plot(epoch, f_loss)

        # draw histograms of real and generated data
        with torch.no_grad():
            z, x = noise(100), sample_data(100)
            hist(x, 'Real')
            hist(g(z), 'Generated')