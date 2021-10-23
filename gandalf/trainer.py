import torch
import importlib
from termcolor import colored

from gandalf.model import Net


def get_class(module, class_name):
    mod = importlib.import_module(module)
    return getattr(mod, class_name)


class Trainer:
    def __init__(self, config, data=None):
        self.config = config

        # set algorithm
        self.config.algo = self.config.algo.replace('-', '_')
        self.algo = get_class('gandalf.algos', self.config.algo)(config)

        # set data
        if data is None:
            self.config.data = self.config.data
            self.data = get_class('gandalf.data', self.config.data)(config)
        else:
            self.data = data
        self.config.data_size = self.data.data_size


    def __getattr__(self, k):
        return getattr(self.config, k)


    def train(self):
        try:
            # set up the generator and discriminator based on the specifications of the algo
            G, D = self.algo.models()
            G, D = Net(G, self.lr), Net(D, self.lr)
            self.algo.setup(G, D)

            # training epochs
            for epoch in range(int(self.epochs)):

                # loop through batches
                for x in self.data.batches():

                    # make sure batch is a full batch, otherwise we'll get matrix dimensionality errors
                    if x.shape[0] == self.config.m:

                        # optimize discriminator
                        for _ in range(self.k):
                            self.algo.optimize_D(x)

                        # optimize generator
                        self.algo.optimize_G()

                with torch.no_grad():
                    # save generated examples occasionally
                    if epoch % self.save_iter == self.save_iter - 1:
                        self.data.save(epoch, G)

        except KeyboardInterrupt:
            com = colored('Training cancelled', 'red')
            print(f'\r{com}')
            quit()
            
        except:
            raise
