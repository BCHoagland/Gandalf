import gandalf

class Config:
    latent_size = 10
    m = 128
    n_hidden = 128
    clip = 0.01

    epochs = 100
    k = 5
    vis_iter = 1
    save_iter = 10
    lr = 1e-4
    λ = 10

    data = 'MNIST'
    algo = 'WGAN-GP'

trainer = gandalf.Trainer(Config)
trainer.train()