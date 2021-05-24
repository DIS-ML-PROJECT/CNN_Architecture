def plot_locs(locs, fig=None, pos=(1, 1, 1), figsize=(15, 15), title=None,
              colors=None, cbar_label=None, show_cbar=True, **scatter_kws):
    '''
    Args
    - locs: np.array, shape [N, 2], each row is [lat, lon]
    - fig: matplotlib.figure.Figure
    - pos: 3-tuple of int, axes position (nrows, ncols, index)
    - figsize: list, [width, height] in inches, only used if fig is None
    - title: str
    - colors: list of int, length N
    - cbar_label: str, label for the colorbar
    - show_cbar: bool, whether to show the colorbar
    - scatter_kws: other arguments for ax.scatter
    Returns: matplotlib.axes.Axes
    '''
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = setup_ax(fig, pos)
    if title is not None:
        ax.set_title(title)

    if 's' not in scatter_kws:
        scatter_kws['s'] = 2
    pc = ax.scatter(locs[:, 1], locs[:, 0], c=colors, **scatter_kws)
    if colors is not None and show_cbar:
        cbar = fig.colorbar(pc, ax=ax, fraction=0.03)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    return ax