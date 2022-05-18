import torch.nn as nn

import gandalf


class Config:
    latent_size = 10        # dimension of latent space
    m = 128                 # batch size
    n_hidden = 128          # size of hidden layer in networks
    clip = 0.01             # clipping cutoff for WGAN

    epochs = 100            # total number of training epochs
    k = 1                   # number of discriminator updates per epoch                                 # should be higher for WGAN
    vis_iter = 5            # generate and save examples every 'vis_iter' epochs
    progress_iter = 5       # give progress reports every 'progress_iter' epochs
    lr = 1e-4               # learning rate for networks
    λ = 10                  # gradient penalty for WGAN-GP

    data = 'MNIST'
    algo = 'GAN'

    G = nn.Sequential(
            nn.Linear(latent_size, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 28 * 28)
    )
    D = nn.Sequential(
            nn.Linear(28 * 28, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()                        # include for GAN, not for WGAN
    )

trainer = gandalf.Trainer(Config)
trainer.train()
