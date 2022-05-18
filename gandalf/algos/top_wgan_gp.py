import torch
import torch.nn as nn
from torch.autograd import grad
import gudhi as gd
from gudhi.representations import Landscape

from gandalf.algos.base import Algo

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def make_rips_complex(points, diameter):
    skeleton = gd.RipsComplex(points=points, max_edge_length=diameter)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    return simplex_tree

def get_persistence_features(simplex_tree):
    simplex_tree.persistence()
    # barcodes = simplex_tree.persistence()
    zero_dim_features = simplex_tree.persistence_intervals_in_dimension(0)
    one_dim_features = simplex_tree.persistence_intervals_in_dimension(1)
    two_dim_features = simplex_tree.persistence_intervals_in_dimension(2)

    features = [zero_dim_features, one_dim_features, two_dim_features]
    # features = normalize_features(features, 1000)
    return features

def get_landscape(x):
    with torch.no_grad():
        rips = make_rips_complex(x, diameter=1000)          #! diameter
        rips.persistence()
        zero_dim_features = rips.persistence_intervals_in_dimension(0)[:-1]     # remove the last one since it's [0,inf)
        l = torch.FloatTensor(Landscape(num_landscapes=2, resolution=10).fit_transform([zero_dim_features]))       # returns vector of shape (1, num_landscapes * resolution)
    return l

def augment_with_landscape(x):
    batch_size = x.shape[0]
    l = get_landscape(x)
    l_repeated = l.repeat(batch_size, 1)
    return torch.cat((x, l_repeated), 1)

class Top_WGAN_GP(Algo):

    def optimize_D(self, G, D, x):
        x = augment_with_landscape(x)

        # network inputs
        z = self.noise()
        x_gen = augment_with_landscape(G(z))
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
        gen_augmented = augment_with_landscape(G(z))
        g_loss = -D(gen_augmented).mean()
        G.minimize(g_loss)
