from utils.analysis import calc_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def scatter_preds(labels, preds, by_name=None, by_col=None, ax=None,
                  title=None, figsize=(5, 5)):
    '''Creates a scatter plot of labels vs. preds, overlayed with regression line.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - by_name: str, name of col
    - by_col: np.array, shape [N]
    - ax: matplotlib.axes.Axes
    - figsize: tuple of (width, height)
    '''
    data = {'labels': labels, 'preds': preds}
    if by_name is not None:
        assert by_col is not None
        data[by_name] = by_col
    df = pd.DataFrame(data)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # y=x
    lo, hi = min(min(labels), min(preds)) - 0.2, max(max(labels), max(preds)) + 0.2
    ax.plot([lo, hi], [lo, hi], '-y')

    # scatterplot
    sns.scatterplot(data=df, x='labels', y='preds', hue=by_name, ax=ax, s=10,
                    linewidth=0)

    # regression line
    r2 = calc_score(labels=labels, preds=preds, metric='r2')
    m, b = np.polyfit(labels, preds, 1)
    ax.plot(labels, m * labels + b, ':k', label=f'$r^2={r2:.3g}$')

    ax.set_aspect('equal')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    if title is not None:
        ax.set_title(title)