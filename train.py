import gandalf

class Config:
    latent_size = 10
    data_size = 1
    m = 64
    n_hidden = 128
    clip = 0.01

# algo = gandalf.algos.GAN(Config)
algo = gandalf.algos.WGAN(Config)
trainer = gandalf.Trainer(algo)

trainer.train()