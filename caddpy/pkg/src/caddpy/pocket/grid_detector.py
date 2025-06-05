
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Literal, Any
import operator
import asyncio
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import ipywidgets as widgets
from IPython.display import display
from scids.field import Field
from scids.grid import Grid
from scishow.nglview import NGLWidget

from caddpy.chemsys import ChemicalSystem
from caddpy import exception


class GridDetector:
    def __init__(self, receptor: ChemicalSystem, field: Field):
        self._receptor = receptor
        self._field = field
        self._grid_axis_indices = tuple(range(self.field.batch_ndim, self.field.tensor.ndim))

        self._mask_volume = np.logical_not(self.field.tensor)
        self._mask_ligsite: np.ndarray | None = None
        self._mask_custom: np.ndarray | None = None
        self._mask = self._mask_volume

        self._ligsite: LigSite | None = None
        self._gui = None
        return

    def set_mask_custom(self, mask: np.ndarray):
        mask = jnp.asarray(mask)
        if mask.shape != self.field.tensor.shape:
            raise exception.InputError(
                name="mask",
                message=f"Mask shape {mask.shape} does not match field shape {self.field.tensor.shape}."
            )
        self._mask_custom = mask
        self._set_mask()
        return self._mask_custom

    def set_mask_ligsite(
        self,
        count_lower: int | bool | None = None,
        count_upper: int | bool | None = None,
        dist_lower: float | bool | None = None,
        dist_upper: float | bool | None = None,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = "all",
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = "any",
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray = (1, 2, 3),
    ):
        if self._ligsite is None or not np.array_equiv(
            self._ligsite.direction,
            LigSite.calculate_direction_vectors(field=self.field, directions=directions),
        ):
            self._ligsite = LigSite(
                field=self.field,
                directions=directions,
            )
        self._mask_ligsite = self._ligsite.psp_mask(
            count_lower=count_lower,
            count_upper=count_upper,
            dist_lower=dist_lower,
            dist_upper=dist_upper,
            dist_lower_mode=dist_lower_mode,
            dist_upper_mode=dist_upper_mode,
        )
        self._set_mask()
        return self._mask_ligsite

    def set_mask_volume(
        self,
        closing_structure: np.ndarray | tuple[int, int] | None = None,
        closing_iterations: int = 1,
        closing_mask: np.ndarray | None = None,
        closing_border_value: Literal[0, 1] = 1,
        fill_structure: np.ndarray | None = None,
    ):
        if isinstance(closing_structure, tuple):
            structure_connectivity, structure_iterations = closing_structure
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
            closing_structure_initial = sp.ndimage.generate_binary_structure(
                rank=3, connectivity=structure_connectivity
            )
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
            closing_structure = sp.ndimage.iterate_structure(
                structure=closing_structure_initial, iterations=structure_iterations
            )
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_closing.html
        volume_closed = sp.ndimage.binary_closing(
            input=self.field.tensor,
            structure=closing_structure,
            iterations=closing_iterations,
            mask=closing_mask,
            border_value=closing_border_value,
            axes=self._grid_axis_indices,
        )
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
        volume_closed_and_filled = sp.ndimage.binary_fill_holes(
            input=volume_closed,
            structure=fill_structure,
            axes=self._grid_axis_indices,
        )
        self._mask_volume = jnp.logical_not(volume_closed_and_filled)
        self._set_mask()
        return self._mask_volume

    def unset_mask(self, *args: Literal["volume", "ligsite", "custom"]):
        args = set(args or ("volume", "ligsite", "custom"))
        if "volume" in args:
            self._mask_volume = np.logical_not(self.field.tensor)
        if "ligsite" in args:
            self._mask_ligsite = None
        if "custom" in args:
            self._mask_custom = None
        self._set_mask()
        return self._mask

    @property
    def mask(self) -> jax.Array:
        return self._mask

    @property
    def mask_volume(self) -> jax.Array:
        return self._mask_volume

    @property
    def mask_ligsite(self) -> jax.Array | None:
        return self._mask_ligsite

    @property
    def mask_custom(self) -> jax.Array | None:
        return self._mask_custom

    @property
    def ligsite(self) -> LigSite | None:
        return self._ligsite

    @property
    def field(self) -> Field:
        return self._field

    def _set_mask(self):
        masks = [self._mask_volume]
        if self._mask_ligsite is not None:
            masks.append(self._mask_ligsite)
        if self._mask_custom is not None:
            masks.append(self._mask_custom)
        self._mask = jnp.logical_and.reduce(jnp.array(masks))
        return


class GridDetectorGUI(GridDetector):
    def __init__(self, receptor: ChemicalSystem, field: Field):
        super().__init__(receptor=receptor, field=field)
        self._nglwidget = NGLWidget().display(gui=True)

        self._widget_psp_dirs = widgets.Dropdown(
            options={
                "Disabled": None,
                "1D": 1,
                "2D": 2,
                "3D": 3,
            },
            rows=4,
        )
        self._widget_psp_dirs.observe(self._on_value_change_psp_dirs, names="value")
        self._widget_psp_count = self._create_numeric_range_slider(
            dtype="int",
            disabled=True,
            observer=self._on_value_change_psp_count
        )
        self._widget_psp_dist = self._create_numeric_range_slider(
            dtype="float",
            disabled=True,
            observer=self._on_value_change_psp_dist
        )

        psp_control_panel = self._create_control_panel(
            header_name="PSP",
            controller_labels=("Directions", "Distance", "Count"),
            controllers=[
                self._widget_psp_dirs,
                self._widget_psp_dist,
                self._widget_psp_count,
            ],
        )

        main_panel = widgets.Accordion(
            children=[psp_control_panel],
        )
        main_panel.set_title(0, "LIGSITE")
        main_panel.selected_index = 0

        self._debug = widgets.Output()
        self._widgets = (self._debug, main_panel, self._nglwidget)

        return

    def display(self):
        """Display the GUI."""
        self._update_grid()
        display(*self._widgets)
        return

    def _update_grid(self):
        name = "grid"
        self._nglwidget.remove_component_by_name(name)
        self._nglwidget.add_spheres(
            coords=self.field.grid.coordinates[self.mask],
            name=name,
        )
        return

    def _on_value_change_psp_dirs(self, change: dict):
        value = change["new"]
        if not value:
            self._widget_psp_count.disabled = True
            self._widget_psp_dist.disabled = True
            self.unset_mask("ligsite")
            self._update_grid()
            return
        self.set_mask_ligsite(directions=tuple(range(1, value + 1)))
        self._widget_psp_count.disabled = False
        self._widget_psp_dist.disabled = False
        self._reset_slider_minmax(
            slider=self._widget_psp_count,
            minimum=self.ligsite.psp_count.min().item(),
            maximum=self.ligsite.psp_count.max().item(),
        )
        self._reset_slider_minmax(
            slider=self._widget_psp_dist,
            minimum=jnp.nanmin(self.ligsite.psp_distance).item(),
            maximum=jnp.nanmax(self.ligsite.psp_distance).item(),
        )
        return

    def _on_value_change_psp_dist(self, change: dict):
        old_lower, old_upper = change["old"]
        new_lower, new_upper = change["new"]
        self.set_mask_ligsite(
            dist_lower=new_lower if new_lower != old_lower else None,
            dist_upper=new_upper if new_upper != old_upper else None,
        )
        self._update_grid()
        return

    def _on_value_change_psp_count(self, change: dict):
        old_lower, old_upper = change["old"]
        new_lower, new_upper = change["new"]
        self.set_mask_ligsite(
            count_lower=new_lower if new_lower != old_lower else None,
            count_upper=new_upper if new_upper != old_upper else None,
        )
        self._update_grid()
        return

    @staticmethod
    def _create_toggle_buttons_source(
        labels: Sequence[str],
        tooltips: str | Sequence[str] = "",
        initial_values: bool | Sequence[bool] = False,
        button_style: Literal["success", "info", "warning", "danger", ""] = "danger",
        observer: Callable | None = None,
    ):
        if isinstance(tooltips, str):
            tooltips = [tooltips] * len(labels)
        if isinstance(initial_values, bool):
            initial_values = [initial_values] * len(labels)
        toggle_buttons = []
        for label, initial_value, tooltip in zip(labels, initial_values, tooltips, strict=False):
            toggle_button = widgets.ToggleButton(
                value=initial_value,
                description=label,
                tooltip=tooltip,
                layout=widgets.Layout(width="auto"),
                button_style=button_style,
            )
            toggle_buttons.append(toggle_button)
            if observer is not None:
                # Set all buttons to observe the same observer function:
                toggle_button.observe(observer, names="value")
        return toggle_buttons

    @staticmethod
    def _create_toggle_buttons_mode(
        labels: Sequence[str] | Sequence[tuple[str, Any]] = (
            ("Min", "min"),
            ("Avg", "avg"),
            ("Max", "max"),
            ("Sum", "sum"),
            ("All", "all"),
            ("Any", "any"),
            ("One", "one"),
        ),
        current_value: Any = "min",
        tooltips: Sequence[str] = (
            "Minimum value of all selected source fields.",
            "Average value of all selected source fields.",
            "Maximum value of all selected source fields.",
            "Sum of all selected source fields.",
            "All values of selected source fields.",
            "Any value of selected source fields.",
            "Only one value of selected source fields.",
        ),
        button_style: Literal["success", "info", "warning", "danger", ""] = "warning",
        disabled: bool = True,
        observer: Callable | None = None,
    ) -> widgets.ToggleButtons:
        toggle_buttons = widgets.ToggleButtons(
            options=labels,
            value=current_value,
            button_style=button_style,
            disabled=disabled,
            tooltips=tooltips,
            style=widgets.ToggleButtonsStyle(button_width="auto"),
        )
        if observer is not None:
            toggle_buttons.observe(observer, names="value")
        return toggle_buttons

    @staticmethod
    def _create_toggle_buttons_filter(
        observer: Callable | None = None,
    ):
        toggle_buttons = widgets.ToggleButtons(
            options=["Include", "Exclude"],
            value="Include",
            disabled=True,
            button_style="info",
            tooltips=[
                "Include the points filtered by current selection.",
                "Exclude the points filtered by current selection.",
            ],
            style=widgets.ToggleButtonsStyle(button_width="4em"),
        )
        if observer is not None:
            toggle_buttons.observe(observer, names="value")
        return toggle_buttons

    @staticmethod
    def _create_numeric_range_slider(
        dtype: Literal["int", "float"] = "float",
        minimum: int | float = 0,
        maximum: int | float = 0,
        value: tuple[int | float, int | float] | None = None,
        step: int | float | None = None,
        continuous_update: bool = False,
        readout_format: str | None = None,
        disabled: bool = False,
        observer: Callable | None = None,
    ):
        widget = widgets.IntRangeSlider if dtype == "int" else widgets.FloatRangeSlider
        if not step:
            step = 1 if dtype == "int" else 0.01
        if readout_format is None:
            readout_format = ".2f" if dtype == "float" else "d"
        range_slider = widget(
            value=value or (minimum, maximum),
            min=minimum,
            max=maximum,
            step=step,
            disabled=disabled,
            continuous_update=continuous_update,
            orientation="horizontal",
            readout=True,
            readout_format=readout_format,
            layout=widgets.Layout(display="flex", align_items="stretch", width="100%"),
        )
        if observer is not None:
            range_slider.observe(observer, names="value")
        return range_slider

    @staticmethod
    def _create_control_panel(
        controllers: Sequence[widgets.Widget],
        header_name: str,
        header_color: str = "lightblue",
        controller_labels: Sequence[str] = ("Source", "Mode", "Filter", "Range"),
    ):
        # Create the header, i.e. title of the widget
        # This is created as a button, because it offers more styling options than text,
        # but it doesn't have any functionality.
        header = widgets.Button(
            description=header_name,
            layout=widgets.Layout(width="auto"),
            style=widgets.ButtonStyle(button_color=header_color, font_weight="bold"),
        )
        # Create the left column holding labels of controllers:
        labels_column = widgets.VBox(
            [
                widgets.Label(
                    value=f"{label} :",
                    layout=widgets.Layout(margin="3px 10px 3px 0px", width="50px"),
                )
                for label in controller_labels
            ],
            layout=widgets.Layout(width="100px"),
        )
        # Create the right column holding the controllers
        controllers_column = widgets.VBox(
            controllers,
            layout=widgets.Layout(
                display="flex", flex_flow="column", align_items="stretch", width="100%"
            ),
        )

        # Horizontal container for two columns holding labels and controllers:
        controller_box = widgets.HBox([labels_column, controllers_column])

        # Assemble the control panel
        control_panel = widgets.Box(
            layout=widgets.Layout(
                display="flex", flex_flow="column", align_items="stretch", width="50%"
            ),
            children=[header, controller_box],
        )
        return control_panel

    @staticmethod
    def _reset_slider_minmax(
        slider: widgets.IntRangeSlider | widgets.FloatRangeSlider,
        minimum: int | float,
        maximum: int | float,
    ):
        """Reset the slider's min and max values."""
        if slider.min >= maximum:
            slider.min = minimum
            slider.max = maximum
        else:
            slider.max = maximum
            slider.min = minimum
        return



class LigSite:
    """LIGSITE binding pocket detector.

    References
    ----------
    - [LIGSITE: automatic and efficient detection of potential small molecule-binding sites in proteins](https://doi.org/10.1016/S1093-3263(98)00002-3)
    """

    def __init__(
        self,
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray = (1, 2, 3),
    ):
        # Validate inputs
        if field.field_ndim != 0:
            raise exception.InputError(
                name="field",
                message=f"Volume field must be scalar (0D), but is {field.field_ndim}D.",
            )
        if field.grid.dimension != 3:
            raise exception.InputError(
                name="field",
                message=f"Volume field must have a 3D grid, but is {field.grid.dimension}D.",
            )
        self._dir = self.calculate_direction_vectors(
            field=field,
            directions=directions,
        )

        # Calculate distance of each grid point to the nearest xeno grid point
        # in each half direction, in units of corresponding distance vectors
        ps_dist_int = field.nearest_target_distances(
            direction_vectors=self._dir,
            predicate=np.logical_xor,
        )
        self._ps_dist_int = jnp.asarray(ps_dist_int)
        dir_lengths = np.linalg.norm(
            self._dir[:, field.batch_ndim:] * field.grid.spacings,
            axis=-1
        )
        # Multiply by direction vector lengths, to get the real distances.
        ps_dists_float = ps_dist_int * dir_lengths
        # set distances that are 0 (meaning no neighbor was found in that direction) to NaN.
        ps_dists_float[ps_dists_float == 0] = np.nan
        self._ps_dist = jnp.asarray(ps_dists_float)
        # Add distances to neighbors in positive half-directions to distances to neighbors in
        # negative half-directions, in order to get the PSP lengths.
        ndir = self._dir.shape[0] // 2
        self._psp_dist = self._ps_dist[..., :ndir] + self._ps_dist[..., -1:-(ndir+1):-1]
        self._psp_count = jnp.count_nonzero(~jnp.isnan(self._psp_dist), axis=-1)

        self._psp_mask: dict[str, jnp.ndarray | None] = {
            "count_lower": None,
            "count_upper": None,
            "dist_lower": None,
            "dist_upper": None,
        }
        return

    def psp_mask(
        self,
        count_lower: int | bool | None = None,
        count_upper: int | bool | None = None,
        dist_lower: float | bool | None = None,
        dist_upper: float | bool | None = None,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = "all",
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = "any",
    ) -> jax.Array | None:
        def make_mask(
            arr: jnp.ndarray,
            threshold: int | float,
            side: Literal["lower", "upper"],
            mode: Literal["any", "all", "max", "min", "mean"]
        ):
            comparison_op = operator.le if side == "upper" else operator.ge
            if mode == "any":
                return jnp.any(comparison_op(arr, threshold), axis=-1)
            if mode == "all":
                return np.all(
                    jnp.logical_or(comparison_op(arr, threshold), jnp.isnan(arr)),
                    axis=-1
                )
            if mode in ("max", "min", "mean"):
                reduction_op = {"max": jnp.nanmax, "min": jnp.nanmin, "mean": jnp.nanmean}[mode]
                return comparison_op(reduction_op(arr, axis=-1), threshold)
            raise ValueError(f"Unknown mode: {mode}")

        if count_lower is not None:
            self._psp_mask["count_lower"] = None if count_lower is True else self.psp_count >= count_lower
        if count_upper is not None:
            self._psp_mask["count_upper"] = None if count_upper is True else self.psp_count <= count_upper
        if dist_lower is not None:
            self._psp_mask["dist_lower"] = None if dist_lower is True else make_mask(
                self.psp_distance, threshold=dist_lower, side="lower", mode=dist_lower_mode
            )
        if dist_upper is not None:
            self._psp_mask["dist_upper"] = None if dist_upper is True else make_mask(
                self.psp_distance, threshold=dist_upper, side="upper", mode=dist_upper_mode
            )
        active_masks = [mask for mask in self._psp_mask.values() if mask is not None]
        return jnp.logical_and.reduce(jnp.array(active_masks)) if active_masks else None

    @property
    def psp_count(self) -> jax.Array:
        """Number of protein-solvent-protein (PSP) events in each direction.

        For unoccupied grid points, this is equal to the number of solvent–protein–solvent (SPS) events.
        """
        return self._psp_count

    @property
    def psp_distance(self) -> jax.Array:
        """Protein–solvent–protein (PSP) distances in each direction, in units of grid spacings (e.g. Ångstrom).

        For unoccupied grid points, this is equal to solvent–protein–solvent (SPS) distances.
        """
        return self._psp_dist

    @property
    def ps_distance(self) -> jax.Array:
        """Distances to nearest xeno grid points in each direction, in units of grid spacings (e.g. Ångstrom).

        A distance of `numpy.nan` means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist

    @property
    def ps_distance_discrete(self) -> jax.Array:
        """Distances to nearest xeno grid points in each direction, in units of direction vectors.

        A distance of 0 means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist_int

    @property
    def direction(self) -> jax.Array:
        """Direction vectors for PSP events.

        This is a 2D array of shape `(26, (self.field.batch_ndim + 3))`
        containing 26 unit vectors pointing to the 26 neighbors of a grid point in a 3D grid.
        Each vector is padded with leading zeros to match the batch dimensions of volume.
        The vectors are ordered such that `self.directions[i] == -self.directions[-(i + 1)]`,
        i.e., the first 13 vectors are the opposite of the last 13 vectors in reverse order.
        """
        return self._dir

    @staticmethod
    def calculate_direction_vectors(
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray = (1, 2, 3),
    ) -> jax.Array:
        if isinstance(directions, int):
            directions = [directions]
        directions = np.asarray(directions)
        if not np.issubdtype(directions.dtype, np.integer):
            raise TypeError("Directions must be integers.")
        if directions.ndim == 1:
            if not np.all(np.isin(directions, [1, 2, 3])):
                raise ValueError("Directions must be 1, 2, or 3.")
            if len(set(directions)) != len(directions):
                raise ValueError("Directions must be unique.")
            dir_vectors = field.grid_direction_vectors(dimensions=directions)
            assert dir_vectors.ndim == 2, "Direction vectors should be 2-dimensional."
            assert dir_vectors.shape[1] == 3, "Direction vectors should be 3D."
        elif directions.ndim == 2:
            if directions.shape[1] != 3:
                raise ValueError("Directions must be 3D")
            dir_vectors = np.pad(
                directions,
                pad_width=((0, 0), (field.batch_ndim, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            raise ValueError("Directions must be 1D or 2D array-like.")
        if (ndirs := dir_vectors.shape[0]) % 2 != 0:
            raise ValueError(f"There should be an even number of direction vectors, but got {ndirs}.")
        if not np.all(dir_vectors[:ndirs] + dir_vectors[-1:-(ndirs+1):-1] == 0):
            raise ValueError("The first half of the direction vectors should be the negative of the second half.")
        return jnp.asarray(dir_vectors)


def from_chemsys(
    system: ChemicalSystem,
    grid: int | float | Sequence[int | float] | Grid = 0.5,
):
    return GridDetectorGUI(receptor=system, field=system.toxelate(grid=grid))
