import numpy as np
from visdom import Visdom
from math import sqrt

from gandalf.visualize.utils import get_line

d = {}
viz = Visdom()

colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']


def format_y(y):
    if not isinstance(y, float):
        # if given a single number, save its float
        if len(y.shape) == 0:
            return float(y)
        # if given a set of numbers, save their mean
        else:
            return y.cpu().mean().item()


def plot(x, y_all, data_type, names):

    # ensure data type spot in dictionary exists
    if data_type not in d:
        d[data_type] = {}

    # loop through given data
    for n_plot in range(y_all.shape[1]):
        y = y_all[:,n_plot]
        name = names[n_plot]
        color = colors[n_plot]

        # ensure dictionary spots for the data exists
        if name not in d[data_type]:
            d[data_type][name] = {'points': [], 'color': color}

        # save the modified data
        y = format_y(y)
        d[data_type][name]['points'].append((x, y))

    # plot all the given data at once
    win = data_type
    title = data_type
    data = []
    for name in d[data_type]:
        x, y = zip(*d[data_type][name]['points'])
        data.append(get_line(x, y, name, color=d[data_type][name]['color'], showlegend=True))

    layout = dict(
        title=title,
        xaxis={'title': 'Epochs'},
        yaxis={'title': data_type}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


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