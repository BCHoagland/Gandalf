import numpy as np
from visdom import Visdom

viz = Visdom()

win = None
def plot(epoch, d_loss, g_loss):
    global win, title

    if win is None:
        title = 'Loss'

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([[d_loss.item(), g_loss.item()]]),
            win='GAN Loss',
            opts=dict(
                title=title,
                legend=['Discriminator', 'Generator']
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([[d_loss.item(), g_loss.item()]]),
            win=win,
            update='append'
        )


def hist(points, name):
    points = points.flatten().cpu()

    title = name
    viz.histogram(
        X=np.array(points),
        win=title,
        opts=dict(
            title=title
        )
    )