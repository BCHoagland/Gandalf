import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from gandalf.algos.base import Algo

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class TopGAN(Algo):

    def optimize_D(self, G, D, x):
        # network inputs
        z = self.noise()
        x_gen = G(z)
        x_inter = self.interpolate(x, x_gen)

        # gradient penalty calculation
        out = D(x_inter)
        grads = grad(out, x_inter, torch.ones(out.shape).to(device), create_graph=True)[0]
        grad_penalty = self.λ * ((self.norm(grads) - 1) ** 2).mean()

        # optimization
        wasserstein = D(x).mean() - D(x_gen).mean()
        d_loss = wasserstein - grad_penalty
        D.maximize(d_loss)

    def optimize_G(self, G, D):
        z = self.noise()
        g_loss = -D(G(z)).mean()
        G.minimize(g_loss)

    def progress(self, G, D, x):
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
                img = images.view(-1, 1, 28, 28).numpy()[0, 0, :, :]
                print(img.shape)
                plt.imshow(img)
                plt.show()
        
        show_first_img(x)
        out = G(self.noise())
        show_first_img(out)
        quit()
