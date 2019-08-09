import gandalf

class Config:
    latent_size = 100
    m = 64
    n_hidden = 128
    clip = 0.01

    epochs = 100
    k = 5
    vis_iter = 1
    save_iter = 10
    lr = 5e-5

    data = 'MNIST'
    algo = 'WGAN'

trainer = gandalf.Trainer(Config)
trainer.train()