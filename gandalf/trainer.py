import torch
import importlib
from termcolor import colored

from gandalf.model import Net
from gandalf.visualize import plot


def get_class(module, class_name):
    mod = importlib.import_module(module)
    return getattr(mod, class_name)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.config.data = self.config.data.upper()
        self.config.algo = self.config.algo.upper()

        self.algo = get_class('gandalf.algos', config.algo)(config)

        self.data = get_class('gandalf.data', config.data)(config)
        self.config.data_size = self.data.data_size


    def __getattr__(self, k):
        return getattr(self.config, k)


    def train(self):
        try:
            G, D = self.algo.models()
            G, D = Net(G, self.lr), Net(D, self.lr)
            self.algo.setup(G, D)

            for epoch in range(int(self.epochs)):

                # keep track of algorithm stats throughout batches
                stats, title, names = [], '', []

                # loop through batches
                for x in self.data.batches():
                    # make sure batch is a full batch, otherwise we'll get matrix dimensionality errors
                    if x.shape[0] == self.config.m:

                        # optimize discriminator
                        for _ in range(self.k):
                            self.algo.optimize_D(x)

                        # optimize generator
                        self.algo.optimize_G()

                        # stats bookkeeping
                        vals, title, names = self.algo.get_stats()
                        stats.append(torch.stack(vals))

                # visualize stats occasionally
                if epoch % self.vis_iter == self.vis_iter - 1:
                    with torch.no_grad():
                        # self.algo.visualize(epoch + 1)                                                      # TODO: make sure this isn't wack
                        plot(epoch, torch.stack(stats), title, names)
                        del stats[:]

                # save generated examples occasionally
                if epoch % self.save_iter == self.save_iter - 1:
                    self.data.save(epoch, G)

        except KeyboardInterrupt:
            com = colored('You killed my man Gandy :(', 'red')
            print(f'\r{com}')
            quit()
            
        except:
            raise