import torch
import importlib
from termcolor import colored
from tqdm import tqdm

from gandalf.model import Net


def get_class(module, class_name):
    mod = importlib.import_module(module)
    return getattr(mod, class_name)


class Trainer:
    def __init__(self, config):
        self.config = config

        # set algorithm
        self.algo = get_class('gandalf.algos', self.config.algo)(config)

        # set data
        self.data = get_class('gandalf.data', self.config.data)(config)
        self.config.data_size = self.data.data_size


    def __getattr__(self, k):
        return getattr(self.config, k)


    def train(self):
        try:
            # set up the generator and discriminator based on the specifications of the algo
            G = Net(self.G, self.lr)
            D = Net(self.D, self.lr)

            # training epochs`
            for epoch in range(int(self.epochs)):

                # loop through batches
                for x in tqdm(self.data.batches()):

                    # make sure batch is a full batch, otherwise we'll get matrix dimensionality errors
                    if x.shape[0] == self.m:

                        # optimize discriminator
                        for _ in range(self.k):
                            self.algo.optimize_D(G, D, x)

                        # optimize generator
                        self.algo.optimize_G(G, D)

                with torch.no_grad():
                    # save generated examples occasionally
                    if epoch % self.vis_iter == self.vis_iter - 1:
                        self.data.save(epoch, G)
                    
                    # analyze the progress of the model occasionally
                    if epoch % self.progress_iter == self.progress_iter - 1:
                        self.algo.progress(G, D, x)
            
                    # print training stats
                    print(f'Epoch {epoch + 1}/{int(self.epochs)} completed')

        except KeyboardInterrupt:
            com = colored('Training cancelled', 'red')
            print(f'\r{com}')
            quit()
            
        except:
            raise
