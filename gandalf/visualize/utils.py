def get_line(x, y, name, color='#000', isFilled=False, fillcolor='transparent', width=2, showlegend=False):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(
                color=color,
                width=width),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=showlegend
        )