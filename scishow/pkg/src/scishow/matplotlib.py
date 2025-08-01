from __future__ import annotations

from typing import TYPE_CHECKING
import math
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display

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


class GridEditor2D:
    """2D grid editor for Jupyter.

    This widget allows interactive editing of a 2D boolean grid
    to generate a numpy array.

    Example
    -------
    >>> %matplotlib widget
    >>> import scishow
    >>> editor = scishow.matplotlib.GridEditor2D(rows=40, cols=60)
    >>> editor.grid
    array([[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False]])
    """

    def __init__(
        self,
        grid: np.ndarray | None = None,
        rows: int = 10,
        cols: int = 10,
    ):
        self._grid = grid or np.zeros((rows, cols), dtype=bool)

        self._rows = self._grid.shape[0]
        self._cols = self._grid.shape[1]

        self._dragging = False
        self._rect_selecting = False
        self._rect_start = None
        self._rectangle_patch = None
        self._edit_mode = "toggle"

        # GUI Controls
        self._mode_dropdown = widgets.ToggleButtons(
            options=["toggle", "on", "off"],
            value=self._edit_mode,
            description="Mode:"
        )
        self._mode_dropdown.observe(self._on_mode_change, names="value")

        self._btn_clear = widgets.Button(description="Clear All", button_style='danger')
        self._btn_fill = widgets.Button(description="Fill All", button_style='success')
        self._btn_clear.on_click(lambda b: self._set_all(False))
        self._btn_fill.on_click(lambda b: self._set_all(True))

        self._rectangle_toggle = widgets.Checkbox(value=False, description='Rectangle Drag')
        control_panel = VBox([
            self._mode_dropdown,
            self._rectangle_toggle,
            HBox([self._btn_fill, self._btn_clear])
        ])
        display(control_panel)

        # Plot setup
        self._fig, self._ax = plt.subplots(figsize=(min(10, cols * 0.2), min(10, rows * 0.2)))
        self._im = self._ax.imshow(self._grid, cmap='gray_r', interpolation='none', vmin=0, vmax=1)

        self._ax.set_xticks(self._sparse_ticks(cols))
        self._ax.set_yticks(self._sparse_ticks(rows))
        self._ax.set_xticklabels(self._sparse_ticks(cols))
        self._ax.set_yticklabels(self._sparse_ticks(rows))
        self._update_title()

        self._ax.format_coord = self._format_coord

        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        return

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def _sparse_ticks(self, size: int) -> list[int]:
        if size <= 20:
            return list(range(size))
        elif size <= 50:
            return list(range(0, size, 2))
        elif size <= 100:
            return list(range(0, size, 5))
        elif size <= 200:
            return list(range(0, size, 10))
        return list(range(0, size, 20))

    def _format_coord(self, x: float, y: float) -> str:
        row, col = int(y), int(x)
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return f"Grid index: [{row}, {col}]"
        return ""

    def _on_mode_change(self, change):
        self._edit_mode = change["new"]
        self._update_title()

    def _on_press(self, event):
        if event.inaxes != self._ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        self._dragging = True

        if self._rectangle_toggle.value:
            self._rect_start = (x, y)
            self._rect_selecting = True
            if self._rectangle_patch:
                self._rectangle_patch.remove()
            self._rectangle_patch = Rectangle((x, y), 0, 0, linewidth=1,
                                             edgecolor='blue', facecolor='blue',
                                             alpha=0.3)
            self._ax.add_patch(self._rectangle_patch)
        else:
            self._apply_at(y, x)

    def _on_motion(self, event):
        if event.inaxes != self._ax:
            return
        x, y = int(event.xdata), int(event.ydata)

        if self._rect_selecting and self._rect_start:
            x0, y0 = self._rect_start
            x1, y1 = x, y
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            self._rectangle_patch.set_xy((xmin, ymin))
            self._rectangle_patch.set_width(xmax - xmin + 1)
            self._rectangle_patch.set_height(ymax - ymin + 1)
            self._fig.canvas.draw_idle()
        elif self._dragging and not self._rectangle_toggle.value:
            self._apply_at(int(y), int(x))

    def _on_release(self, event):
        self._dragging = False
        if self._rect_selecting and self._rect_start:
            x0, y0 = self._rect_start
            x1, y1 = int(event.xdata), int(event.ydata)
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])

            self._apply_rectangle(ymin, ymax, xmin, xmax)

            self._rect_selecting = False
            self._rect_start = None
            if self._rectangle_patch:
                self._rectangle_patch.remove()
                self._rectangle_patch = None
            self._fig.canvas.draw_idle()

    def _apply_at(self, row: int, col: int):
        if 0 <= row < self._rows and 0 <= col < self._cols:
            match self._edit_mode:
                case 'on':
                    self._grid[row, col] = True
                case 'off':
                    self._grid[row, col] = False
                case 'toggle':
                    self._grid[row, col] = not self._grid[row, col]
            self._refresh()

    def _apply_rectangle(self, y0: int, y1: int, x0: int, x1: int):
        y0, y1 = max(0, y0), min(self._rows - 1, y1)
        x0, x1 = max(0, x0), min(self._cols - 1, x1)

        match self._edit_mode:
            case 'on':
                self._grid[y0:y1 + 1, x0:x1 + 1] = True
            case 'off':
                self._grid[y0:y1 + 1, x0:x1 + 1] = False
            case 'toggle':
                self._grid[y0:y1 + 1, x0:x1 + 1] ^= True
        self._refresh()

    def _set_all(self, value: bool):
        self._grid[:, :] = value
        self._refresh()

    def _refresh(self):
        self._im.set_data(self._grid)
        self._fig.canvas.draw_idle()

    def _update_title(self):
        self._ax.set_title(
            f"Mode: {self._edit_mode.upper()} (R = rectangle drag)"
        )
