import numpy as np
from visdom import Visdom

viz = Visdom()

title = 'GAN Loss by Epoch'
win = None

def update_viz(epoch, d_loss, g_loss):
    global win, title

    if win is None:
        title = title

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([[d_loss, g_loss]]),
            win=title,
            opts=dict(
                title=title,
                legend=['Discriminator', 'Generator']
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([[d_loss, g_loss]]),
            win=win,
            update='append'
        )
