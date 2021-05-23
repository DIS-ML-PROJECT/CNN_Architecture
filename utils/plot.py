def plot_image_by_band(img, band_order, nrows, title, rgb=None, colorbar=False):
    '''
    Args
    - img: np.array, shape [H, W, C], type float, normalized
    - band_order: list of str, names of the bands in order
    - nrows: int, desired number of rows in the created figure
    - title: str, or None
    - rgb: one of [None, 'merge', 'add']
        - None: do not create a separate RGB image
        - 'merge': plot the RGB bands as a merged image
        - 'add': plot all bands, but also add a merged RGB image
    - colorbar: bool, whether to show colorbar
    '''
    nbands = img.shape[2]
    rgb_to_naxs = {
        None: nbands,
        'merge': nbands - 2,
        'add': nbands + 1
    }
    nplots = rgb_to_naxs[rgb]
    ncols = int(np.ceil(nplots / float(nrows)))
    fig_w = min(15, 3*ncols)
    fig_h = min(15, 3*nrows)
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                            figsize=[fig_w, fig_h], constrained_layout=True)
    if title is not None:
        fig.suptitle(title, y=1.03)

    # scale image to [0,1]: 0 = -3 std, 0.5 = mean, 1 = +3 std
    scaled_img = np.clip(img / 6.0 + 0.5, a_min=0, a_max=1)
    bands = {band_name: scaled_img[:, :, b] for b, band_name in enumerate(band_order)}

    plots = []
    plot_titles = []
    if rgb is not None:
        r, g, b = bands['RED'], bands['GREEN'], bands['BLUE']
        rgb_img = np.stack([r,g,b], axis=2)
        plots.append(rgb_img)
        plot_titles.append('RGB')

    if rgb == 'merge':
        for band_name in band_order:
            if band_name not in ['RED', 'GREEN', 'BLUE']:
                plots.append(bands[band_name])
                plot_titles.append(band_name)
    else:
        plots += [bands[band_name] for band_name in band_order]
        plot_titles += band_order

    for b in range(len(plots)):
        if len(axs.shape) == 1:
            ax = axs[b]
        else:
            ax = axs[b // ncols, b % ncols]
        # set origin='lower' to match lat/lon direction
        im = ax.imshow(plots[b], origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(plot_titles[b])

    if colorbar:
        fig.colorbar(im, orientation='vertical', ax=axs)
    plt.show()