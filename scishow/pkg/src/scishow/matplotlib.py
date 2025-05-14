from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product

if TYPE_CHECKING:
    from typing import Sequence


class Plotter:
    """Modular plot composer for building complex matplotlib figures.

    Parameters
    ----------
    max_cols
        Maximum number of columns in subplot grid.
    figsize_per_plot
        Size (width, height) per subplot.
    """
    def __init__(self, max_cols: int = 4, figsize_per_plot: tuple[float, float] = (4, 3)):
        self._fig = None
        self._axes_grid = None
        self._nrows = None
        self._ncols = None
        self._max_cols = max_cols
        self._figsize_per_plot = figsize_per_plot
        self._next_plot_idx = 0  # Tracks next available subplot index
        return

    def figure(self) -> plt.Figure:
        """The underlying matplotlib Figure object."""
        return self._fig

    def axes(self) -> np.ndarray:
        """The underlying axes grid array."""
        return self._axes_grid

    def add_histograms(
        self,
        data: np.ndarray,
        axes: int | tuple[int, ...],
        names: Sequence[str] | None = None,
        colors: Sequence[tuple[float, float, float]] | None = None,
        bins: int = 50,
        bin_range: tuple[float | None, float | None] | None = None,
        log_scale: bool = True
    ) -> Plotter:
        """Add histograms for slices of arr along specified axes.

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
        ndim = data.ndim
        if isinstance(axes, int):
            axes = (axes,)
        if any(ax < 0 or ax >= ndim for ax in axes):
            raise ValueError(f"Axes {axes} out of bounds for shape {data.shape}")

        iter_shape = tuple(data.shape[ax] for ax in axes)
        n_histograms = np.prod(iter_shape)

        # Ensure grid is allocated
        self._ensure_grid(self._next_plot_idx + n_histograms)

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

        colors = [
            tuple(c / 255 if isinstance(c, int) and c > 1 else c for c in color)
            if isinstance(color, (tuple, list)) and len(color) == 3
            else color
            for color in colors
        ]

        for idx, index_tuple in enumerate(product(*[range(s) for s in iter_shape])):
            plot_idx = self._next_plot_idx + idx
            r, c = divmod(plot_idx, self._ncols)
            ax = self._axes_grid[r][c]

            # Slice data
            slicer = [slice(None)] * ndim
            for ax_idx, ax_dim in enumerate(axes):
                slicer[ax_dim] = index_tuple[ax_idx]
            data_slice = data[tuple(slicer)]
            data_flat = data_slice.reshape(-1)

            # Filter valid finite values
            data_flat = data_flat[np.isfinite(data_flat)]

            if data_flat.size == 0:
                # Skip this histogram or give a warning
                print(f"Warning: No finite data for histogram {names[idx]}, skipping.")
                continue

            # Determine bin range dynamically if needed
            if bin_range is not None:
                data_min, data_max = data_flat.min(), data_flat.max()
                actual_bin_range = (
                    bin_range[0] if bin_range[0] is not None else data_min,
                    bin_range[1] if bin_range[1] is not None else data_max,
                )
            else:
                actual_bin_range = None  # Let matplotlib decide

            # Plot histogram
            ax.hist(data_flat, bins=bins, range=actual_bin_range, color=colors[idx], edgecolor='black')
            ax.set_title(names[idx])
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            if log_scale:
                ax.set_yscale('log')

        self._next_plot_idx += n_histograms

        # Turn off unused subplots
        for idx in range(self._next_plot_idx, self._nrows * self._ncols):
            r, c = divmod(idx, self._ncols)
            self._fig.delaxes(self._axes_grid[r][c])
        return self

    def add_annotation(
        self,
        text: str,
        subplot_idx: int,
        xy: tuple[float, float],
        **kwargs
    ) -> Plotter:
        """Add annotation text to a specific subplot.

        Parameters
        ----------
        text
            Annotation text.
        subplot_idx
            Index of subplot in flattened grid.
        xy
            Coordinates to annotate.
        kwargs
            Additional arguments passed to ax.annotate().
        """
        if subplot_idx >= self._next_plot_idx:
            raise IndexError(f"Subplot index {subplot_idx} is out of bounds (current plots: {self._next_plot_idx})")
        r, c = divmod(subplot_idx, self._ncols)
        ax = self._axes_grid[r][c]
        ax.annotate(text, xy=xy, **kwargs)
        return self

    def finalize(self, tight_layout: bool = True) -> Plotter:
        """Finalize layout (e.g., tight_layout)."""
        if tight_layout:
            self._fig.tight_layout()
        return self

    def show(self) -> None:
        """Display the composed figure."""
        self.finalize()
        try:
            from IPython.display import display
            display(self._fig)
        except ImportError:
            self._fig.show()
        return

    def savefig(self, filepath: str, **kwargs) -> None:
        """Save figure to file."""
        self.finalize()
        self._fig.savefig(filepath, **kwargs)

    def _ensure_grid(self, total_plots: int) -> None:
        """Ensure subplot grid is created or expanded to fit total_plots."""
        ncols = min(total_plots, self._max_cols)
        nrows = math.ceil(total_plots / ncols)
        if self._fig is None:
            figsize = (self._figsize_per_plot[0] * ncols, self._figsize_per_plot[1] * nrows)
            self._fig, self._axes_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            self._nrows, self._ncols = nrows, ncols
            plt.close(self._fig)  # prevent Jupyter auto-display
        elif total_plots > self._nrows * self._ncols:
            raise ValueError("Dynamic grid resizing not yet implemented. Plan grid size in advance.")
