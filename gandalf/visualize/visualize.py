import numpy as np
from visdom import Visdom

from gandalf.visualize.utils import get_line

d = {}
viz = Visdom()

def plot(x, y, data_type, name, color='#000', refresh=True):
    # create dictionary spots if they don't already exist
    if data_type not in d:
        d[data_type] = {}
    if name not in d[data_type]:
        d[data_type][name] = {'points': [], 'color': color}
    
    # if given a single number, save its float
    if not isinstance(y, float):
        if len(y.shape) == 0:
            y = float(y)
        # if given a set of numbers, save their mean and confidence interval info
        else:
            y = y.cpu()
            mean, std = y.mean().item(), 3.291 * y.std().item() / sqrt(len(y))
            lower, upper = mean - std, mean + std
            y = (lower, mean, upper)

    # save the modified data
    d[data_type][name]['points'].append((x, y))

    # the actual plotting
    if refresh:
        win = data_type
        title = data_type
        data = []
        for name in d[data_type]:
            x, y = zip(*d[data_type][name]['points'])

            # if extracting mean and confidence internval info, plot the mean with error shading
            if isinstance(y[0], tuple):
                lower, mean, upper = zip(*y)
                data.append(get_line(x, lower, '', color='transparent'))
                data.append(get_line(x, upper, '', color='transparent', isFilled=True, fillcolor=d[data_type][name]['color'] + '44'))
                data.append(get_line(x, mean, name, color=d[data_type][name]['color'], showlegend=True))
            # if extracting single values, plot them as a single line
            else:
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