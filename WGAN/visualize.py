import numpy as np
from visdom import Visdom

viz = Visdom()

win = None
def plot(epoch, w):
    global win, title

    if win is None:
        title = 'Earth-Mover Estimate'

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([w.item()]),
            win='WGAN Loss',
            opts=dict(
                title=title,
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([w.item()]),
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