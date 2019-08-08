# import importlib
# def get_class(module, name):
#     m = importlib.import_module(module)
#     return getattr(m, name)
# s = 'wgan'.upper()
# print(get_class('gandalf.algos', s))

import gandalf

class Config:
    latent_size = 10
    data_size = 28 * 28
    m = 64
    n_hidden = 128
    clip = 0.01

    epochs = 100
    k = 5
    vis_iter = 1
    save_iter = 10
    lr = 1e-4

trainer = gandalf.Trainer(gandalf.data.MNIST, gandalf.algos.WGAN, Config)
trainer.train()