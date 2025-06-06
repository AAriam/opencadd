"""Grid-based binding pocket detection."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
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
from scids.field import Field
from scids.grid import Grid
from scishow.nglview import NGLWidget
from scishow.ipywidgets import GUI
import scishow.ipywidgets as widgeter
import arrayer

from caddpy.chemsys import ChemicalSystem
from caddpy.typing import Array, JAXArray, Int, Num, Bool, Float, atypecheck
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
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray | None = None,
    ):
        if self._ligsite is None and directions is None:
            raise exception.InputError(
                name="directions",
                message="Directions must be specified when setting the mask for the first time.",
            )
        if self._ligsite is None or (
            directions is not None and not np.array_equiv(
                self._ligsite.direction,
                LigSite.calculate_direction_vectors(field=self.field, directions=directions),
            )
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


class GridDetectorGUI(GridDetector, GUI):

    _WIDGET_NAME = {
        "ngl": "ngl",
    }

    def __init__(self, receptor: ChemicalSystem, field: Field):

        def make_ngl() -> NGLWidget:
            nglwidget = NGLWidget().display(gui=True)
            nglwidget.add_trajectory(receptor, name="Receptor")
            nglwidget.component_0.add_surface(
                color="rgb(100,20,20)",
                opacity=0.5,
                surface_type="vws",
                scale_factor=10,
                smooth=10
            )
            self._gui__add_widget(self._WIDGET_NAME["ngl"], nglwidget)
            return nglwidget

        def make_ligsite():
            def make_top_panel():
                directions = widgeter.labeled_widget(
                    value="Directions:",
                    widget=self._gui__add_widget(
                        name=f"{name_prefix}psp_dirs",
                        widget=widgets.Dropdown(
                            options={"None": None, "1D": 1, "2D": 2, "3D": 3},
                            layout=widgets.Layout(width="flex-grow", min_width="20px"),
                        )
                    )
                )
                refresh = self._gui__add_widget(
                    name=f"{name_prefix}refresh",
                    widget=widgets.Button(
                        description="Refresh",
                        tooltip="Recalculate the LIGSITE mask with the current settings.",
                        button_style="success",
                        disabled=True,
                    )
                )
                auto_refresh = self._gui__add_widget(
                    name=f"{name_prefix}auto_refresh",
                    widget=widgets.ToggleButton(
                        description="Auto Refresh",
                        tooltip="Automatically recalculate the LIGSITE mask when the settings change.",
                        value=True,
                        button_style="success",
                        disabled=True,
                    )
                )
                reset = self._gui__add_widget(
                    name=f"{name_prefix}reset",
                    widget=widgets.Button(
                        description="Reset",
                        tooltip="Reset the LIGSITE mask to the default state.",
                        button_style="danger",
                        disabled=False,
                    )
                )
                return widgets.HBox(
                    [directions, status, refresh, auto_refresh, reset],
                    layout=widgets.Layout(width="100%", align_items="center")
                )

            def make_psp_panels():
                return widgets.HBox(
                    [make_psp_panel(panel_type) for panel_type in ("count", "dist")],
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="space-between",
                        flex_flow="row wrap",
                        margin="10px 0 10px 0"
                    )
                )

            def make_psp_panel(panel_type: Literal["count", "dist"]):
                title = widgets.HBox(
                    [widgets.HTML(f"<b>PSP {"Count" if panel_type == 'count' else 'Distance'}</b>")],
                    layout=widgets.Layout(justify_content="center", width="100%")
                )
                slider_class = widgets.IntRangeSlider if panel_type == "count" else widgets.FloatRangeSlider
                slider = self._gui__add_widget(
                    name=f"{name_prefix}psp_{panel_type}_slider",
                    widget=slider_class(
                        value=(0, 13) if panel_type == "count" else (0.0, 100.0),
                        min=0 if panel_type == "count" else 0.0,
                        max=13 if panel_type == "count" else 100.0,
                        step=1 if panel_type == "count" else 0.01,
                        disabled=True,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d" if panel_type == "count" else ".2f",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                dropdown_options = {"Enabled": True, "Disabled": False} if panel_type == "count" else {
                    "Any": "any",
                    "All": "all",
                    "Max": "max",
                    "Min": "min",
                    "Mean": "mean",
                    "Disabled": False
                }
                minmax_dropdowns = []
                for side in ("min", "max"):
                    dropdown = self._gui__add_widget(
                        name=f"{name_prefix}psp_{panel_type}_{side}",
                        widget=widgets.Dropdown(
                            options=dropdown_options,
                            value=True if panel_type == "count" else ("all" if side == "min" else "any"),
                            layout=widgets.Layout(width="100%"),
                            disabled=True,
                        )
                    )
                    minmax_dropdowns.append(
                        widgeter.labeled_widget(
                            value=f"{side.capitalize()}:",
                            widget=dropdown
                        )
                    )
                minmax_dropdowns.insert(
                    1,
                    widgets.Box(layout=widgets.Layout(flex="1 1 50px"))
                )
                minmax_settings = widgets.HBox(
                    minmax_dropdowns,
                    layout=widgets.Layout(width="100%", align_items="center")
                )
                return widgets.VBox(
                    [title, slider, minmax_settings],
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        padding="12px",
                        border="0.5px solid lightgray",
                        border_radius="10px",
                        min_width="0",
                        overflow="hidden",
                    )
                )

            name_prefix = "ligsite_"
            return widgets.VBox(
                [make_top_panel(), make_psp_panels()],
                layout=widgets.Layout(width="100%", overflow="hidden"),
            )

        def make_logger():
            self._gui__logger = widgets.Output()
            return widgets.Accordion(
                children=[self._gui__logger],
                titles=["Logs"],
                selected_index=None,
            )

        GridDetector.__init__(self, receptor=receptor, field=field)
        GUI.__init__(self, observer_method_name_template="_ovc_")

        status = self._gui__add_widget(
            name="status",
            widget=widgets.Label(
                value="",
                style={
                    "text_color": "red",
                    "background": "black",
                    "font_style": "italic",
                    "font_weight": "bold",
                },
                layout=widgets.Layout(
                    width="flex-grow",
                    min_width="fit-content",
                    display="none",
                    padding="0 10px 0 10px",
                ),
            )
        )
        top_panel = widgets.Tab(
            children=[make_ligsite()],
            titles=["LIGSITE"],
            selected_index=0,
        )

        self._gui__set_main_widget((top_panel, make_ngl(), make_logger()))
        return

    @property
    def nglwidget(self) -> NGLWidget:
        """The NGLWidget containing the protein structure and the pocket volume."""
        return self._gui__get_widget(self._WIDGET_NAME["ngl"])

    def _ovc__ligsite_psp_dirs(self, change: dict):
        with self._gui__logger:
            print(f"LIGSITE directions changed to {change['new']}.")
        # Temporarily disable controls during calculation to prevent user interaction.
        self._ligsite__toggle_controls(False)
        value = change["new"]
        if not value:
            # If no directions are selected, leave controls disabled and unset the mask.
            self.unset_mask("ligsite")
            return {}
        with self._show_status():
            # Calculate mask to get new PSP min/max values.
            self.set_mask_ligsite(directions=tuple(range(1, value + 1)))
            # Reset the sliders to the new min/max values.
            self._reset_slider_minmax(
                slider=self._gui__get_widget("ligsite_psp_count_slider"),
                minimum=self.ligsite.psp_count.min().item(),
                maximum=self.ligsite.psp_count.max().item(),
            )
            self._reset_slider_minmax(
                slider=self._gui__get_widget("ligsite_psp_dist_slider"),
                minimum=jnp.nanmin(self.ligsite.psp_distance).item(),
                maximum=jnp.nanmax(self.ligsite.psp_distance).item(),
            )
            auto_refresh = self._gui__get_widget("ligsite_auto_refresh").value
            if auto_refresh:
                # Recalculate mask with the current min/max values when auto-refresh is enabled.
                self._ligsite__recalculate_mask(pass_directions=False)
        # Reenable controls after calculation.
        self._ligsite__toggle_controls(True)
        return {} if auto_refresh else None

    def _ovc__ligsite_refresh(self, _: widgets.Button):
        with self._gui__logger:
            print("Refreshing LIGSITE mask with current settings.")
        directions = self._gui__get_widget("ligsite_psp_dirs")
        if directions.value:
            self._ligsite__recalculate_mask(pass_directions=True)
        else:
            self._ligsite__toggle_controls(False)
            self.unset_mask("ligsite")
        return {}

    def _ovc__ligsite_auto_refresh(self, change: dict):
        enabled = change["new"]
        with self._gui__logger:
            print(f"LIGSITE auto-refresh {'enabled' if enabled else 'disabled'}.")
        button = change["owner"]
        button.button_style = "success" if enabled else "danger"
        self._gui__toggle_widget_observer(
            observe=enabled,
            name_regex="^ligsite_psp_.*",
        )
        return

    def _ovc__ligsite_reset(self, _: widgets.Button):
        with self._gui__logger:
            print("Resetting LIGSITE mask to default state.")
        self._ligsite__toggle_controls(False)
        with self._show_status():
            self._gui__toggle_widget_observer(False)
            self._gui__get_widget("ligsite_psp_dirs").value = 3
            self.set_mask_ligsite(directions=tuple(range(1, 4)))
            count_slider = self._gui__get_widget("ligsite_psp_count_slider")
            count_slider_min = self.ligsite.psp_count.min().item()
            count_slider_max = self.ligsite.psp_count.max().item()
            dist_slider = self._gui__get_widget("ligsite_psp_dist_slider")
            dist_slider_min = jnp.nanmin(self.ligsite.psp_distance).item()
            dist_slider_max = jnp.nanmax(self.ligsite.psp_distance).item()
            self._reset_slider_minmax(
                slider=count_slider,
                minimum=count_slider_min,
                maximum=count_slider_max,
                disable_observe=False,
            )
            self._reset_slider_minmax(
                slider=dist_slider,
                minimum=dist_slider_min,
                maximum=dist_slider_max,
                disable_observe=False,
            )
            count_slider.lower = max(count_slider.lower, 5)
            dist_slider.upper = min(dist_slider.upper, 10.0)
            self._gui__get_widget("ligsite_psp_count_min").value = True
            self._gui__get_widget("ligsite_psp_count_max").value = False
            self._gui__get_widget("ligsite_psp_dist_min").value = "all"
            self._gui__get_widget("ligsite_psp_dist_max").value = "any"
            self._ligsite__recalculate_mask(pass_directions=False)
            self._gui__toggle_widget_observer(True)
        self._ligsite__toggle_controls(True)
        return {}

    def _ovc__ligsite_psp_count_slider(self, change: dict):
        with self._gui__logger:
            print(f"LIGSITE PSP count range changed to {change['new']}.")
        with self._show_status():
            self._ligsite__toggle_controls(False)
            old_lower, old_upper = change["old"]
            new_lower, new_upper = change["new"]
            min_enabled = self._gui__get_widget("ligsite_psp_count_min").value
            max_enabled = self._gui__get_widget("ligsite_psp_count_max").value
            self.set_mask_ligsite(
                count_lower=(new_lower if new_lower != old_lower else None) if min_enabled else True,
                count_upper=(new_upper if new_upper != old_upper else None) if max_enabled else True,
            )
        self._ligsite__toggle_controls(True)
        return {}

    def _ovc__ligsite_psp_count_min(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP count lower mode to {change['new']}.")
        with self._show_status():
            self._ligsite__toggle_controls(False)
            min_enabled = change["new"]
            lower = self._gui__get_widget("ligsite_psp_count_slider").lower
            self.set_mask_ligsite(count_lower=lower if min_enabled else True)
        self._ligsite__toggle_controls(True)
        return {}

    def _ovc__ligsite_psp_count_max(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP count upper mode to {change['new']}.")
        with self._show_status():
            self._ligsite__toggle_controls(False)
            max_enabled = change["new"]
            upper = self._gui__get_widget("ligsite_psp_count_slider").upper
            self.set_mask_ligsite(count_upper=upper if max_enabled else True)
        self._ligsite__toggle_controls(True)
        return {}

    def _ovc__ligsite_psp_dist_slider(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP distance slider to {change['new']}.")
        with self._show_status():
            self._ligsite__toggle_controls(False)
            old_lower, old_upper = change["old"]
            new_lower, new_upper = change["new"]
            min_type = self._gui__get_widget("ligsite_psp_dist_min").value
            max_type = self._gui__get_widget("ligsite_psp_dist_max").value
            self.set_mask_ligsite(
                dist_lower=(new_lower if new_lower != old_lower else None) if min_type else True,
                dist_upper=(new_upper if new_upper != old_upper else None) if max_type else True,
                dist_lower_mode=min_type,
                dist_upper_mode=max_type,
            )
        self._ligsite__toggle_controls(True)
        return {}

    def _ovc__ligsite_psp_dist_min(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP distance lower mode to {change['new']}.")
        with self._show_status():
            self._ligsite__toggle_controls(False)
            new_mode = change["new"]
            lower = self._gui__get_widget("ligsite_psp_dist_slider").lower
            self.set_mask_ligsite(
                dist_lower=lower if new_mode else True,
                dist_lower_mode=new_mode,
            )
        self._ligsite__toggle_controls(True)
        return {}

    def _ovc__ligsite_psp_dist_max(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP distance upper mode to {change['new']}.")
        with self._show_status():
            self._ligsite__toggle_controls(False)
            new_mode = change["new"]
            upper = self._gui__get_widget("ligsite_psp_dist_slider").upper
            self.set_mask_ligsite(
                dist_upper=upper if new_mode else True,
                dist_upper_mode=new_mode,
            )
        self._ligsite__toggle_controls(True)
        return {}

    def _gui__render(self):
        """Update the NGLWidget with the current mask.

        This method is automatically called by the `GUI` parent class when needed.
        """
        with self._show_status("Rendering..."):
            name = "Volume"
            ngl = self.nglwidget
            ngl.remove_component_by_name(name)
            ngl.add_spheres(
                coords=self.field.grid.coordinates[self.mask],
                name=name,
            )
        return

    def _ligsite__recalculate_mask(self, pass_directions: bool):
        psp_directions = self._gui__get_widget("ligsite_psp_dirs")
        psp_count_slider = self._gui__get_widget("ligsite_psp_count_slider")
        psp_dist_slider = self._gui__get_widget("ligsite_psp_dist_slider")
        psp_count_min = self._gui__get_widget("ligsite_psp_count_min")
        psp_count_max = self._gui__get_widget("ligsite_psp_count_max")
        psp_dist_min = self._gui__get_widget("ligsite_psp_dist_min")
        psp_dist_max = self._gui__get_widget("ligsite_psp_dist_max")
        self.set_mask_ligsite(
            count_lower=psp_count_slider.lower if psp_count_min.value else True,
            count_upper=psp_count_slider.upper if psp_count_max.value else True,
            dist_lower=psp_dist_slider.lower if psp_dist_min.value else True,
            dist_upper=psp_dist_slider.upper if psp_dist_max.value else True,
            dist_lower_mode=psp_dist_min.value,
            dist_upper_mode=psp_dist_max.value,
            directions=tuple(range(1, psp_directions.value + 1)) if pass_directions else None,
        )
        return

    def _ligsite__toggle_controls(self, status: bool):
        """Enable/Disable the LIGSITE mask and its controls."""
        disabled = not status
        for widget_name in (
            "refresh",
            "auto_refresh",
            "reset",
            "psp_dirs",
            "psp_count_slider",
            "psp_dist_slider",
            "psp_count_min",
            "psp_count_max",
            "psp_dist_min",
            "psp_dist_max",
        ):
            self._gui__get_widget(f"ligsite_{widget_name}").disabled = disabled
        return

    def _reset_slider_minmax(
        self,
        slider: widgets.IntRangeSlider | widgets.FloatRangeSlider,
        minimum: int | float,
        maximum: int | float,
        disable_observe: bool = True,
    ):
        """Reset the slider's min and max values."""
        if disable_observe:
            self._gui__toggle_widget_observer(False, widget=slider)
        if slider.min >= maximum:
            slider.min = minimum
            slider.max = maximum
        else:
            slider.max = maximum
            slider.min = minimum
        slider.value = (minimum, maximum)
        if disable_observe:
            self._gui__toggle_widget_observer(True, widget=slider)
        return slider

    @contextmanager
    def _show_status(self, status: str = "Calculating..."):
        """Show the status."""
        status_widget = self._gui__get_widget("status")
        status_widget.value = status
        status_widget.layout.display = ""
        try:
            yield
        finally:
            status_widget.layout.display = "none"
        return


class LigSite:
    """LIGSITE binding pocket detector.

    This class only implements the core functionality of LIGSITE;
    It calculates the protein-solvent-protein (PSP) events
    in the specified directions, and provides a method to generate masks
    based on the number and distance of these events.
    It is used in the `GridDetector` class,
    which implements the remaining functionality of LIGSITE,
    among other grid-based pocket detection methods.

    Parameters
    ----------
    field
        A voxel `Field` object where
        non-zero values represent the protein volume.
    directions
        Directions in which to calculate PSP events.
        This can be one of the following:
        - An integer array of shape `(n_directions, 3)`,
          where each row is a direction vector
          from one point to another in the 3D grid
          (e.g. `[1, 0, 0]` for the positive x-direction).
          All vectors must be linearly independent,
          and the smallest vector for each direction must be provided.
        - A single integer within the range `[1, 3]`,
          corresponding to 1D, 2D, or 3D directions, respectively.
          N-dimensional directions are those that have N non-zero components.
          For example, 1D directions are `[1, 0, 0]`, `[0, 1, 0]`, and `[0, 0, 1]`,
          corresponding to directions along the x, y, and z axes, respectively.
        - A non-repeating sequence of integers within the range `[1, 3]`,
          to combine 1D, 2D, and 3D directions.
          For example, `(1, 2)` will generate all 1D and 2D directions.

    References
    ----------
    - [LIGSITE: automatic and efficient detection of potential
      small molecule-binding sites in proteins](https://doi.org/10.1016/S1093-3263(98)00002-3)
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
    ) -> Bool[JAXArray, "*field.shape"] | None:
        """Generate a volume mask based on PSP events.

        This method generates a single boolean mask by combining up to four conditions
        with logical AND operations:
        - Minimum number of PSP events for each grid point, as defined by the `count_lower` parameter.
        - Maximum number of PSP events for each grid point, as defined by the `count_upper` parameter.
        - Minimum PSP distance for each grid point, as defined by the `dist_lower` and `dist_lower_mode` parameters.
        - Maximum PSP distance for each grid point, as defined by the `dist_upper` and `dist_upper_mode` parameters.

        To calculate more complex masks, you can directly access the individual properties
        `psp_count`, `psp_distance`, `ps_distance`, and `ps_distance_discrete` in this class,
        which are the underlying data used to generate the masks.

        To improve performance when generating multiple masks based on slight variations,
        the last submask calculated for each condition is cached, and will be reused in
        subsequent calls to this method if the corresponding parameter is set to `None` (default).
        Providing a value of `True` for any parameter will exclude that condition from the mask,
        and will also clear the corresponding cached submask.
        Providing another value will generate a new submask for that condition and update the cache.

        Since each grid point has multiple PSP distances (one for each direction),
        the distance masks are calculated by reducing the PSP distances along the direction axis,
        to obtain a single boolean value for each grid point. The reduction mode can be specified
        using the `dist_lower_mode` and `dist_upper_mode` parameters,
        for the lower and upper distance masks, respectively.
        The supported reduction modes are:

        - `any`: The mask is `True` for a grid point if
          any PSP distance for that point is below/above the threshold.
        - `all`: The mask is `True` for a grid point if
          all PSP distances for that point are below/above the threshold.
        - `max`: The mask is `True` for a grid point if
          the maximum PSP distance for that point is below/above the threshold.
        - `min`: The mask is `True` for a grid point if
          the minimum PSP distance for that point is below/above the threshold.
        - `mean`: The mask is `True` for a grid point if
          the mean PSP distance for that point is below/above the threshold.

        Note that all reduction modes ignore `NaN` values in the PSP distances
        (these are for directions where no PSP event was found).
        If you are interested in points that have no `NaN` values in the PSP distances
        (i.e., points that have PSP events in all directions),
        you can simply set the `count_lower` parameter to the number of directions,
        which will only include points that have PSP events in all directions.

        Parameters
        ----------
        count_lower
            Minimum number of PSP events for each grid point.
            Set to `True` to exclude this condition from the mask,
            or set to `None` (default) to reuse the last generated mask (if any).
        count_upper
            Maximum number of PSP events for each grid point.
            Set to `True` to exclude this condition from the mask,
            or set to `None` (default) to reuse the last generated mask (if any).
        dist_lower
            Minimum PSP distance for each grid point.
            Set to `True` to exclude this condition from the mask,
            or set to `None` (default) to reuse the last generated mask (if any).
        dist_upper
            Maximum PSP distance for each grid point.
            Set to `True` to exclude this condition from the mask,
            or set to `None` (default) to reuse the last generated mask (if any).
        dist_lower_mode
            Reduction mode for calculating the lower PSP distance mask.
        dist_upper_mode
            Reduction mode for calculating the upper PSP distance mask.

        Returns
        -------
        A boolean array of the same shape as the input field tensor,
        where `True` values indicate grid points that satisfy all specified conditions.
        If no conditions are specified,
        i.e., all parameters are `True` or are `None` with no previous cache,
        the method returns `None`.

        Note that the mask does not distinguish between occupied and unoccupied grid points,
        i.e., for unoccupied grid points, it calculates based on protein–solvent–protein (PSP) events,
        while for occupied grid points, it uses solvent–protein–solvent (SPS) events.
        To obtain a mask that is only `True` for unoccupied grid points (i.e., solvent/free space),
        you must simply combine this mask with a mask for unoccupied grid points using logical AND.
        """
        def make_mask(
            arr: jnp.ndarray,
            threshold: int | float,
            side: Literal["lower", "upper"],
            mode: Literal["any", "all", "max", "min", "mean"],
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
    def psp_count(self) -> Int[JAXArray, "*field.shape"]:
        """Number of protein-solvent-protein (PSP) events in each direction.

        For unoccupied grid points, this is equal to the number of solvent–protein–solvent (SPS) events.
        """
        return self._psp_count

    @property
    def psp_distance(self) -> Float[JAXArray, "*field.shape {self.direction.shape[0] / 2}"]:
        """Protein–solvent–protein (PSP) distances in each direction, in units of grid spacings (e.g. Ångstrom).

        For unoccupied grid points, this is equal to solvent–protein–solvent (SPS) distances.
        A distance of `numpy.nan` means that no PSP/SPS event was found in that direction.
        """
        return self._psp_dist

    @property
    def ps_distance(self) -> Float[JAXArray, "*field.shape {self.direction.shape[0]}"]:
        """Distances to nearest xeno grid points in each direction, in units of grid spacings (e.g. Ångstrom).

        A distance of `numpy.nan` means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist

    @property
    def ps_distance_discrete(self) -> Int[JAXArray, "*field.shape {self.direction.shape[0]}"]:
        """Distances to nearest xeno grid points in each direction, in units of direction vectors.

        A distance of 0 means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist_int

    @property
    def direction(self) -> Int[JAXArray, "{2 * n_directions} {field.batch_ndim + 3}"]:
        """Direction vectors used for PSP/SPS events calculation.

        This is a 2D array of shape `(2 * n_signed_directions, (field.batch_ndim + 3))`
        containing `2 * n_signed_directions` unit vectors pointing to the neighbors of a grid point in a 3D grid.
        Each vector is padded with leading zeros to match the batch dimensions of the input volume field.
        The vectors are ordered such that `self.directions[i] == -self.directions[-(i + 1)]`,
        i.e., the first `n_signed_directions` vectors are
        the opposite of the last `n_signed_directions` vectors in reverse order.
        These correspond to the positive and negative half-directions
        in which the PSP/SPS events are calculated.
        The order corresponds to the order of distances in the `psp_distance`,
        `ps_distance`, and `ps_distance_discrete` properties.
        """
        return self._dir

    @staticmethod
    def calculate_direction_vectors(
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray = (1, 2, 3),
    ) -> jax.Array:
        """Validate and calculate direction vectors for PSP events.

        This function does not need to be called by the user directly;
        it is called during the initialization,
        and used as a utility method in the `GridDetectorGUI` class
        to compare new directions with the existing ones.
        Parameters are the same as in the `LigSite` class constructor.
        """
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
            linearly_dependents = arrayer.matrix.linearly_dependent_pairs(directions)
            if linearly_dependents.size > 0:
                raise exception.InputError(
                    name="directions",
                    message="Following direction vector pairs "
                            f"are linearly dependent: {linearly_dependents.tolist()}."
                )
            full_directions = np.concatenate([directions, -directions[::-1]], axis=0)
            dir_vectors = np.pad(
                full_directions,
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
) -> GridDetectorGUI:
    """Create a grid-based pocket detector from a chemical system.

    Parameters
    ----------
    system
        A `ChemicalSystem` object containing the receptor structure.
    grid
        The grid spacing for the voxel grid.
        This can be a single value (e.g. `0.5` for 0.5 Ångstrom spacing),
        or a Grid object specifying the grid.
    """
    return GridDetectorGUI(receptor=system, field=system.toxelate(grid=grid))
