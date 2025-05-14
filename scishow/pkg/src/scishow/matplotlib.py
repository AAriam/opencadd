import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product
from typing import Sequence, Tuple


def plot_histograms_along_axes(
    arr: np.ndarray,
    axes: Tuple[int, ...],
    names: Sequence[str] | None = None,
    colors: Sequence[Tuple[float, float, float]] | None = None,
    bins: int = 50,
    bin_range: Tuple[float, float] | None = None,
    max_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (4, 3),
    log_scale: bool = True
) -> None:
    """
    Plot histograms for slices of an array along specified axes.

    Parameters
    ----------
    arr : np.ndarray
        Input array of arbitrary shape.
    axes : tuple of int
        Axes to iterate over; for each index combination along these axes,
        a histogram will be plotted.
    names : list of str, optional
        Titles per histogram. If not provided, auto-generated from indices.
    colors : list of RGB tuples, optional
        Colors for each histogram. If not provided, defaults used.
    bins : int
        Number of histogram bins.
    bin_range : tuple of float, optional
        (min, max) range for histogram bins.
    max_cols : int
        Max columns in subplot grid.
    figsize_per_plot : tuple of float
        Size (width, height) per subplot.
    log_scale : bool
        Whether to apply log scale to y-axis.
    """
    # Validate axes
    ndim = arr.ndim
    if any(ax < 0 or ax >= ndim for ax in axes):
        raise ValueError(f"Axes {axes} out of bounds for array with shape {arr.shape}")

    iter_shape = tuple(arr.shape[ax] for ax in axes)
    n_histograms = np.prod(iter_shape)

    # Prepare names
    if names is None:
        index_combinations = list(product(*[range(s) for s in iter_shape]))
        names = [f"indices {idx}" for idx in index_combinations]
    elif len(names) != n_histograms:
        raise ValueError(f"Expected {n_histograms} names, got {len(names)}")

    # Prepare colors
    if colors is None:
        colors = ['steelblue'] * n_histograms
    elif len(colors) != n_histograms:
        raise ValueError(f"Expected {n_histograms} colors, got {len(colors)}")

    # Normalize RGB if needed
    colors = [tuple(c / 255 if isinstance(c, int) and c > 1 else c for c in color) for color in colors]

    # Plot grid layout
    ncols = min(n_histograms, max_cols)
    nrows = math.ceil(n_histograms / ncols)
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Axes not being iterated (to keep flattened)
    remaining_axes = [ax for ax in range(ndim) if ax not in axes]

    for idx, index_tuple in enumerate(product(*[range(s) for s in iter_shape])):
        # Build slicing object
        slicer = [slice(None)] * ndim
        for ax_idx, ax in enumerate(axes):
            slicer[ax] = index_tuple[ax_idx]

        data_slice = arr[tuple(slicer)]

        # Flatten remaining axes
        data_flat = data_slice.reshape(-1)

        # Plotting
        r, c = divmod(idx, ncols)
        ax = axes_grid[r][c]

        ax.hist(data_flat, bins=bins, range=bin_range, color=colors[idx], edgecolor='black')
        ax.set_title(names[idx])
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        if log_scale:
            ax.set_yscale('log')

    # Turn off unused subplots
    for j in range(n_histograms, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes_grid[r][c])

    fig.tight_layout()
    plt.show()
