import matplotlib.pyplot as plt


def plot_histograms_by_last_axis(
    arr, names=None, colors=None, bins=50,
    bin_range=None, max_cols=4, figsize_per_plot=(4, 3), log_scale=True
):
    """
    Plot histograms for each slice along the last axis of a 4D array.

    Parameters:
        arr (np.ndarray): Shape (nx, ny, nz, na)
        names (list of str): Optional titles per histogram.
        colors (list of RGB tuples): Optional colors for each histogram.
        bins (int): Number of histogram bins.
        bin_range (tuple): (min, max) range for histogram bins.
        max_cols (int): Max columns in subplot grid.
        figsize_per_plot (tuple): Size (width, height) per subplot.
        log_scale (bool): Whether to apply log scale to y-axis.
    """
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got {arr.shape}")

    na = arr.shape[-1]

    if names is None:
        names = [f"Index {i}" for i in range(na)]
    elif len(names) != na:
        raise ValueError("Length of `names` must match last dimension of array")

    if colors is None:
        colors = ['steelblue'] * na
    elif len(colors) != na:
        raise ValueError("Length of `colors` must match last dimension of array")

    # Normalize RGB values if needed
    colors = [tuple(c / 255 if isinstance(c, int) else c for c in color) for color in colors]

    ncols = min(na, max_cols)
    nrows = math.ceil(na / ncols)
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i in range(na):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        data = arr[..., i].ravel()

        ax.hist(data, bins=bins, range=bin_range, color=colors[i], edgecolor='black')
        ax.set_title(names[i])
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        if log_scale:
            ax.set_yscale('log')

    # Turn off unused subplots
    for j in range(na, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    plt.show()
