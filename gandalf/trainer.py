import torch
from termcolor import colored

from gandalf.model import Net


class Trainer:
    def __init__(self, data, algo, config):
        self.data = data(config)
        self.algo = algo(config)
        self.config = config


    def __getattr__(self, k):
        return getattr(self.config, k)


    def train(self):
        try:
            G, D = self.algo.models()
            G, D = Net(G, self.lr), Net(D, self.lr)
            self.algo.setup(G, D)

            for epoch in range(int(self.epochs)):

                for x in self.data.batches():
                    if x.shape[0] == self.config.m:

                        for _ in range(self.k):
                            self.algo.optimize_D(x)

                        self.algo.optimize_G()

                if epoch % self.vis_iter == self.vis_iter - 1:
                    with torch.no_grad():
                        self.algo.visualize(epoch)

                if epoch % self.save_iter == self.save_iter - 1:
                    self.data.save(epoch, G)

        except KeyboardInterrupt:
            com = colored('You killed my man Gandy :(', 'red')
            print(f'\r{com}')
            quit()
            
        except:
            raise