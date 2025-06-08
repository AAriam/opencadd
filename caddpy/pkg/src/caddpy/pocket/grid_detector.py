"""Grid-based binding pocket detection."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from enum import Enum

from IPython import display
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import ipywidgets as widgets
import scishow
from scids.field import Field
from scids.grid import Grid


from caddpy.chemsys import ChemicalSystem
from caddpy.pocket.ligsite import LigSite
from caddpy import exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from scishow.nglview import NGLWidget

class _Default:
    """Default values for the grid detector."""

    # Morphological Transformations

    # Closing
    MORPH_CLOSE = True
    MORPH_CLOSE_ITER = 1
    MORPH_CLOSE_BORDER = 1
    # Closing Structure
    MORPH_CLOSE_STRUCT_CONNECT = 1
    MORPH_CLOSE_STRUCT_ITER = 1

    # Hole Filling
    MORPH_FILL = True
    # Filling Structure
    MORPH_FILL_STRUCT_CONNECT = 1
    MORPH_FILL_STRUCT_ITER = 1


    # LIGSITE

    # PSP Count
    LIGSITE_COUNT = True
    LIGSITE_COUNT_LOWER = 4
    LIGSITE_COUNT_UPPER = 13

    # PSP Distance
    LIGSITE_DIST = True
    LIGSITE_DIST_LOWER = 1.2
    LIGSITE_DIST_UPPER = None
    LIGSITE_DIST_LOWER_MODE = "all"
    LIGSITE_DIST_UPPER_MODE = "any"


class GridDetector:
    def __init__(self, receptor: ChemicalSystem, field: Field):
        self._receptor = receptor
        self._field = field

        self._grid_axis_indices = tuple(range(self.field.batch_ndim, self.field.tensor.ndim))
        self._original_volume_receptor = self.field.tensor.astype(bool)
        self._original_volume_solvent = jnp.logical_not(self._original_volume_receptor)
        self._ligsite = LigSite(field=self.field, directions=(1, 2, 3))
        self._receptor_volume = self._original_volume_receptor
        self._mask_morphology = self._original_volume_solvent
        self._mask_ligsite: np.ndarray | None = None
        self._mask_custom: np.ndarray | None = None
        self._gui = None
        return

    def set_mask_morphology(
        self,
        close: bool = _Default.MORPH_CLOSE,
        fill: bool = _Default.MORPH_FILL,
        closing_structure: np.ndarray | tuple[int, int] = (
            _Default.MORPH_CLOSE_STRUCT_CONNECT,
            _Default.MORPH_CLOSE_STRUCT_ITER
        ),
        closing_iterations: int = _Default.MORPH_CLOSE_ITER,
        closing_mask: np.ndarray | None = None,
        closing_border_value: Literal[0, 1] = _Default.MORPH_CLOSE_BORDER,
        fill_structure: np.ndarray | tuple[int, int] = (
            _Default.MORPH_FILL_STRUCT_CONNECT,
            _Default.MORPH_FILL_STRUCT_ITER,
        )
    ):
        if close:
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
        else:
            volume_closed = self.field.tensor
        if fill:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
            if isinstance(fill_structure, tuple):
                structure_connectivity, structure_iterations = fill_structure
                fill_structure_initial = sp.ndimage.generate_binary_structure(
                    rank=3, connectivity=structure_connectivity
                )
                fill_structure = sp.ndimage.iterate_structure(
                    structure=fill_structure_initial, iterations=structure_iterations
                )
            volume_closed_and_filled = sp.ndimage.binary_fill_holes(
                input=volume_closed,
                structure=fill_structure,
                axes=self._grid_axis_indices,
            )
        else:
            volume_closed_and_filled = volume_closed
        self._receptor_volume = volume_closed_and_filled.astype(bool)
        self._mask_morphology = jnp.logical_not(volume_closed_and_filled)
        return self._mask_morphology

    def set_mask_ligsite(
        self,
        count_lower: int | None = _Default.LIGSITE_COUNT_LOWER,
        count_upper: int | None = _Default.LIGSITE_COUNT_UPPER,
        dist_lower: float | None = _Default.LIGSITE_DIST_LOWER,
        dist_upper: float | None = _Default.LIGSITE_DIST_UPPER,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = _Default.LIGSITE_DIST_LOWER_MODE,
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = _Default.LIGSITE_DIST_UPPER_MODE,
    ):
        self._mask_ligsite = self._ligsite.psp_mask(
            count_lower=count_lower,
            count_upper=count_upper,
            dist_lower=dist_lower,
            dist_upper=dist_upper,
            dist_lower_mode=dist_lower_mode,
            dist_upper_mode=dist_upper_mode,
        )
        return self._mask_ligsite

    def set_mask_custom(self, mask: np.ndarray):
        mask = jnp.asarray(mask)
        if mask.shape != self.field.tensor.shape:
            raise exception.InputError(
                name="mask",
                message=f"Mask shape {mask.shape} does not match field shape {self.field.tensor.shape}."
            )
        self._mask_custom = mask
        return self._mask_custom

    def unset_mask(self, *args: Literal["morphology", "ligsite", "custom"]) -> None:
        args = set(args or ("morphology", "ligsite", "custom"))
        if "morphology" in args:
            self._mask_morphology = np.logical_not(self.field.tensor)
        if "ligsite" in args:
            self._mask_ligsite = None
        if "custom" in args:
            self._mask_custom = None
        return

    @property
    def mask(self) -> jax.Array:
        masks = [self._mask_morphology]
        if self._mask_ligsite is not None:
            masks.append(self._mask_ligsite)
        if self._mask_custom is not None:
            masks.append(self._mask_custom)
        return jnp.logical_and.reduce(jnp.array(masks))

    @property
    def mask_morphology(self) -> jax.Array:
        return self._mask_morphology

    @property
    def mask_ligsite(self) -> jax.Array | None:
        return self._mask_ligsite

    @property
    def mask_custom(self) -> jax.Array | None:
        return self._mask_custom

    @property
    def receptor_volume(self) -> jax.Array:
        """The receptor volume tensor after morphological transformation.

        This is the inverse of `mask_morphology`.
        """
        return self._receptor_volume

    @property
    def receptor_volume_added(self) -> jax.Array:
        """The volume added to the receptor after morphological transformation."""
        return jnp.logical_and(self.receptor_volume, self._original_volume_solvent)

    @property
    def receptor_volume_removed(self) -> jax.Array:
        """The volume removed from the receptor after morphological transformation."""
        return jnp.logical_and(self.mask_morphology, self._original_volume_receptor)

    @property
    def ligsite(self) -> LigSite | None:
        return self._ligsite

    @property
    def field(self) -> Field:
        return self._field

    @property
    def receptor(self) -> ChemicalSystem:
        return self._receptor


class _WPrefix(Enum):
    MORPH = "morph_"
    LIGSITE = "ligsite_"


class _WName(Enum):
    """Widget names for the GUI."""

    # Morphological Transformations
    MORPH_REFRESH = f"_{_WPrefix.MORPH.value}refresh"
    MORPH_RESET = f"_{_WPrefix.MORPH.value}reset"

    # Morphological Closing
    MORPH_CLOSE = f"{_WPrefix.MORPH.value}close"
    MORPH_CLOSE_BORDER = f"{_WPrefix.MORPH.value}close_border"
    MORPH_CLOSE_ITER = f"{_WPrefix.MORPH.value}close_iter"
    MORPH_CLOSE_MASK = f"{_WPrefix.MORPH.value}close_mask"
    # Morphological Closing Structure
    MORPH_CLOSE_STRUCT_CONNECT = f"{_WPrefix.MORPH.value}close_struct_connect"
    MORPH_CLOSE_STRUCT_ITER = f"{_WPrefix.MORPH.value}close_struct_iter"
    MORPH_CLOSE_STRUCT_CUSTOM = f"{_WPrefix.MORPH.value}close_struct_custom"

    # Morphological Filling
    MORPH_FILL = f"{_WPrefix.MORPH.value}fill"
    # Morphological Filling Structure
    MORPH_FILL_STRUCT_CONNECT = f"{_WPrefix.MORPH.value}fill_struct_connect"
    MORPH_FILL_STRUCT_ITER = f"{_WPrefix.MORPH.value}fill_struct_iter"
    MORPH_FILL_STRUCT_CUSTOM = f"{_WPrefix.MORPH.value}fill_struct_custom"


    # LIGSITE
    LIGSITE_REFRESH = f"_{_WPrefix.LIGSITE.value}refresh"
    LIGSITE_RESET = f"_{_WPrefix.LIGSITE.value}reset"

    # LIGSITE PSP Count
    LIGSITE_COUNT = f"{_WPrefix.LIGSITE.value}count"
    LIGSITE_COUNT_RANGE = f"{_WPrefix.LIGSITE.value}count_range"

    # LIGSITE PSP Distance
    LIGSITE_DIST = f"{_WPrefix.LIGSITE.value}dist"
    LIGSITE_DIST_RANGE = f"{_WPrefix.LIGSITE.value}dist_range"
    LIGSITE_DIST_MIN = f"{_WPrefix.LIGSITE.value}dist_min"
    LIGSITE_DIST_MAX = f"{_WPrefix.LIGSITE.value}dist_max"


class GridDetectorGUI(scishow.widgets.GUI):

    _CSS_STYLE = """<style>
        .statusbar {
            color: #fff;
            text-align: left;
            font-weight: bold;
            border-radius: 3px;
            white-space: nowrap;
            padding: 0 8px;
            overflow: auto;
            flex: 1 1 auto;
        }
        .statusbar-idle {
            background-color: rgb(0 50 0);
        }
        .statusbar-running {
            background-color: rgb(50 0 0);
        }
        .button-bold {
            color: #fff;
            font-weight: bold;
        }
        .resetbutton {
            background-color: rgb(100 0 0) !important;
            color: white !important;
        }
        .togglebutton-on {
            background-color: rgb(0 100 0) !important;
            color: white !important;
        }
        .togglebutton-off {
            background-color: rgb(100 100 100) !important;
            color: white !important;
        }
    </style>"""

    def __init__(self, detector: GridDetector):

        def widget_status() -> widgets.Label:
            """Create a status widget to display the current status of the detector."""
            widget = widgets.Label(value="Idle")
            widget.add_class("statusbar")
            widget.add_class("statusbar-idle")
            return widget

        def widget_ngl() -> NGLWidget:
            nglwidget = scishow.nglview.NGLWidget().display(gui=True)
            nglwidget.add_trajectory(self.receptor, name="Receptor")
            nglwidget.add_spheres(
                coords=self.field.grid.coordinates[self.field.tensor.astype(bool)],
                name="Receptor Volume (Original)",
            )
            # nglwidget.component_0.add_surface(
            #     color="rgb(100,20,20)",
            #     opacity=0.5,
            #     surface_type="vws",
            #     scale_factor=10,
            #     smooth=10
            # )

            return nglwidget

        def tab_morph():
            def panel_close():
                def on_close_button_value_change(change: dict):
                    disabled = not change["new"]
                    for widget in (
                        structure_connectivity,
                        structure_iterations,
                        closing_iterations,
                        border_value,
                        custom_structure,
                        custom_mask,
                    ):
                        widget.disabled = disabled
                    return

                close = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE.value,
                    widget=scishow.widgets.toggle_button(
                        "Close",
                        value=_Default.MORPH_CLOSE,
                        disabled=False,
                        tooltip="Apply morphological closing to the protein volume.",
                    )
                )
                structure_connectivity = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE_STRUCT_CONNECT.value,
                    widget=widgets.Dropdown(
                        options=[1, 2, 3],
                        value=_Default.MORPH_CLOSE_STRUCT_CONNECT,
                        layout=widgets.Layout(width="100%"),
                        disabled=not _Default.MORPH_CLOSE,
                    )
                )
                structure_iterations = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE_STRUCT_ITER.value,
                    widget=widgets.IntSlider(
                        value=_Default.MORPH_CLOSE_STRUCT_ITER,
                        min=1,
                        max=100,
                        step=1,
                        disabled=not _Default.MORPH_CLOSE,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                closing_iterations = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE_ITER.value,
                    widget=widgets.IntSlider(
                        value=_Default.MORPH_CLOSE_ITER,
                        min=1,
                        max=100,
                        step=1,
                        disabled=not _Default.MORPH_CLOSE,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                border_value = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE_BORDER.value,
                    widget=widgets.Dropdown(
                        options=[0, 1],
                        value=_Default.MORPH_CLOSE_BORDER,
                        layout=widgets.Layout(width="100%"),
                        disabled=not _Default.MORPH_CLOSE,
                    )
                )
                custom_structure = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE_STRUCT_CUSTOM.value,
                    widget=widgets.Button(
                        description="Custom Structure",
                        icon="trash",
                        layout=widgets.Layout(display="none")
                    )
                )
                custom_structure.add_class("resetbutton")
                custom_mask = self._gui__add_widget(
                    name=_WName.MORPH_CLOSE_MASK.value,
                    widget=widgets.Button(
                        description="Custom Mask",
                        icon="trash",
                        layout=widgets.Layout(display="none")
                    )
                )
                custom_mask.add_class("resetbutton")
                close.observe(on_close_button_value_change, names="value")
                structure_connectivity_labeled = scishow.widgets.labeled_widget(
                    value="Structure Connectivity:",
                    widget=structure_connectivity
                )
                structure_iterations_labeled = scishow.widgets.labeled_widget(
                    value="Structure Iterations:",
                    widget=structure_iterations
                )
                closing_iterations_labeled = scishow.widgets.labeled_widget(
                    value="Closing Iterations:",
                    widget=closing_iterations
                )
                border_value_labeled = scishow.widgets.labeled_widget(
                    value="Border Value:",
                    widget=border_value
                )
                children = [
                    close,
                    structure_connectivity_labeled,
                    structure_iterations_labeled,
                    closing_iterations_labeled,
                    border_value_labeled,
                    custom_structure,
                    custom_mask
                ]
                return widgets.VBox(
                    children,
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        padding="12px",
                        border="0.5px solid lightgray",
                        border_radius="10px",
                        min_width="0",
                        overflow="hidden",
                    )
                )

            def panel_fill():
                def on_fill_button_value_change(change: dict):
                    disabled = not change["new"]
                    for widget in (
                        structure_connectivity,
                        structure_iterations,
                        custom_structure,
                    ):
                        widget.disabled = disabled
                    return

                fill = self._gui__add_widget(
                    name=_WName.MORPH_FILL.value,
                    widget=scishow.widgets.toggle_button(
                        "Fill Holes",
                        value=_Default.MORPH_FILL,
                        disabled=False,
                        tooltip="Fill holes in the protein volume after closing.",
                    )
                )
                structure_connectivity = self._gui__add_widget(
                    name=_WName.MORPH_FILL_STRUCT_CONNECT.value,
                    widget=widgets.Dropdown(
                        options=[1, 2, 3],
                        value=_Default.MORPH_FILL_STRUCT_CONNECT,
                        layout=widgets.Layout(width="100%"),
                        disabled=not _Default.MORPH_FILL,
                    )
                )
                structure_iterations = self._gui__add_widget(
                    name=_WName.MORPH_FILL_STRUCT_ITER.value,
                    widget=widgets.IntSlider(
                        value=_Default.MORPH_FILL_STRUCT_ITER,
                        min=1,
                        max=100,
                        step=1,
                        disabled=not _Default.MORPH_FILL,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                custom_structure = self._gui__add_widget(
                    name=_WName.MORPH_FILL_STRUCT_CUSTOM.value,
                    widget=widgets.Button(
                        description="Custom Structure",
                        icon="trash",
                        layout=widgets.Layout(display="none")
                    )
                )
                fill.observe(on_fill_button_value_change, names="value")
                structure_connectivity_labeled = scishow.widgets.labeled_widget(
                    value="Structure Connectivity:",
                    widget=structure_connectivity
                )
                structure_iterations_labeled = scishow.widgets.labeled_widget(
                    value="Structure Iterations:",
                    widget=structure_iterations
                )
                return widgets.VBox(
                    [fill, structure_connectivity_labeled, structure_iterations_labeled, custom_structure],
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        padding="12px",
                        border="0.5px solid lightgray",
                        border_radius="10px",
                        min_width="0",
                        overflow="hidden",
                    )
                )

            panels = widgets.HBox(
                    [panel_close(), panel_fill()],
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="space-between",
                        flex_flow="row wrap",
                        margin="10px 0 10px 0"
                    )
                )
            return widgets.VBox(
                [panel_top(_WPrefix.MORPH), panels],
                layout=widgets.Layout(width="100%", overflow="hidden"),
            )

        def tab_ligsite():
            def panel_count():
                def on_toggle(change: dict):
                    slider.disabled = not change["new"]
                    return
                toggle = self._gui__add_widget(
                    name=_WName.LIGSITE_COUNT,
                    widget=scishow.widgets.toggle_button(
                        "PSP Count",
                        value=_Default.LIGSITE_COUNT,
                        disabled=False,
                        tooltip="Apply PSP count mask to the protein volume.",
                    )
                )
                slider = self._gui__add_widget(
                    name=_WName.LIGSITE_COUNT_RANGE.value,
                    widget=widgets.IntRangeSlider(
                        value=(
                            max(_Default.LIGSITE_COUNT_LOWER, self.ligsite.psp_count_min),
                            min(_Default.LIGSITE_COUNT_UPPER, self.ligsite.psp_count_max)
                        ),
                        min=self.ligsite.psp_count_min,
                        max=self.ligsite.psp_count_max,
                        step=1,
                        disabled=not _Default.LIGSITE_COUNT,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                toggle.observe(on_toggle, names="value")
                return widgets.VBox(
                    [toggle, slider],
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        padding="12px",
                        border="0.5px solid lightgray",
                        border_radius="10px",
                        min_width="0",
                        overflow="hidden",
                    )
                )

            def panel_dist():
                def on_toggle(change: dict):
                    disabled = not change["new"]
                    for widget in (slider, *minmax_dropdowns):
                        widget.disabled = disabled
                    return
                toggle = self._gui__add_widget(
                    name=_WName.LIGSITE_DIST,
                    widget=scishow.widgets.toggle_button(
                        "PSP Distance",
                        value=_Default.LIGSITE_DIST,
                        disabled=False,
                        tooltip="Apply PSP distance mask to the protein volume.",
                    )
                )
                slider = self._gui__add_widget(
                    name=_WName.LIGSITE_DIST_RANGE.value,
                    widget=widgets.FloatRangeSlider(
                        value=(
                            max(_Default.LIGSITE_DIST_LOWER, self.ligsite.psp_distance_min) if _Default.LIGSITE_DIST_LOWER is not None else self.ligsite.psp_distance_min,
                            min(_Default.LIGSITE_DIST_UPPER, self.ligsite.psp_distance_max) if _Default.LIGSITE_DIST_UPPER is not None else self.ligsite.psp_distance_max,
                        ),
                        min=self.ligsite.psp_distance_min,
                        max= self.ligsite.psp_distance_max,
                        step=0.01,
                        disabled=not _Default.LIGSITE_DIST,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format=".2f",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                minmax_dropdowns = []
                minmax_dropdowns_labeled = []
                for side in ("min", "max"):
                    default_dist = _Default.LIGSITE_DIST_LOWER if side == "min" else _Default.LIGSITE_DIST_UPPER
                    default_mode = _Default.LIGSITE_DIST_LOWER_MODE if side == "min" else _Default.LIGSITE_DIST_UPPER_MODE
                    dropdown = self._gui__add_widget(
                        name=_WName[f"LIGSITE_DIST_{side.upper()}"].value,
                        widget=widgets.Dropdown(
                            options={
                                "Any": "any",
                                "All": "all",
                                "Max": "max",
                                "Min": "min",
                                "Mean": "mean",
                                "Off": None,
                            },
                            value=None if default_dist is None else default_mode,
                            layout=widgets.Layout(width="100%"),
                            disabled=not _Default.LIGSITE_DIST,
                        )
                    )
                    minmax_dropdowns.append(dropdown)
                    minmax_dropdowns_labeled.append(
                        scishow.widgets.labeled_widget(
                            value=f"{side.capitalize()}:",
                            widget=dropdown
                        )
                    )
                minmax_dropdowns_labeled.insert(
                    1,
                    widgets.Box(layout=widgets.Layout(flex="1 1 50px"))
                )
                minmax_settings = widgets.HBox(
                    minmax_dropdowns_labeled,
                    layout=widgets.Layout(width="100%", align_items="center")
                )
                toggle.observe(on_toggle, names="value")
                return widgets.VBox(
                    [toggle, slider, minmax_settings],
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        padding="12px",
                        border="0.5px solid lightgray",
                        border_radius="10px",
                        min_width="0",
                        overflow="hidden",
                    )
                )

            panels = widgets.HBox(
                    [panel_count(), panel_dist()],
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="space-between",
                        flex_flow="row wrap",
                        margin="10px 0 10px 0"
                    )
                )
            return widgets.VBox(
                [panel_top(_WPrefix.LIGSITE), panels],
                layout=widgets.Layout(width="100%", overflow="hidden"),
            )

        def panel_top(prefix: str):
            button_layout = widgets.Layout(min_width="70px", max_width="70px", flex="0 0 auto")
            refresh = self._gui__add_widget(
                name=_WName[f"{prefix.name}_REFRESH"].value,
                widget=scishow.widgets.toggle_button(
                    "Refresh",
                    value=True,
                    disabled=False,
                    tooltip="Automatically recalculate the mask when the settings change.",
                    layout=button_layout,
                    css_class="button-bold"
                )
            )
            reset = self._gui__add_widget(
                name=_WName[f"{prefix.name}_RESET"].value,
                widget=widgets.Button(
                    description="Reset",
                    tooltip="Reset the morphology mask to the default state.",
                    disabled=False,
                    layout=button_layout,
                )
            )
            reset.add_class("button-bold")
            reset.add_class("resetbutton")
            buttons_box = widgets.HBox(
                [refresh, reset],
                layout=widgets.Layout(justify_content="flex-end", flex="0 0 auto")
            )
            return widgets.HBox(
                [self._widg_status, buttons_box],
                layout=widgets.Layout(width="100%", align_items="center", justify_content="space-between")
            )

        super().__init__()
        self._detector = detector
        self._widg_status = widget_status()
        self._widg_ngl = widget_ngl()
        self._widg_log = widgets.Output()
        self._gui__set_main_widget(
            (
                display.HTML(self._CSS_STYLE),
                widgets.Tab(
                    children=[tab_morph(), tab_ligsite()],
                    titles=["Morphology", "LIGSITE"],
                    selected_index=0,
                ),
                self._widg_ngl,
                widgets.Accordion(
                    children=[self._widg_log],
                    titles=["Logs"],
                    selected_index=None,
                )
            )
        )
        self._custom_input = {
            _WName.MORPH_CLOSE_STRUCT_CUSTOM: None,
            _WName.MORPH_FILL_STRUCT_CUSTOM: None,
            _WName.MORPH_CLOSE_MASK: None,
        }
        return

    def set_mask_morphology(
        self,
        close: bool = _Default.MORPH_CLOSE,
        fill: bool = _Default.MORPH_FILL,
        closing_structure: np.ndarray | tuple[int, int] = (
            _Default.MORPH_CLOSE_STRUCT_CONNECT,
            _Default.MORPH_CLOSE_STRUCT_ITER
        ),
        closing_iterations: int = _Default.MORPH_CLOSE_ITER,
        closing_mask: np.ndarray | None = None,
        closing_border_value: Literal[0, 1] = _Default.MORPH_CLOSE_BORDER,
        fill_structure: np.ndarray | tuple[int, int] = (
            _Default.MORPH_FILL_STRUCT_CONNECT,
            _Default.MORPH_FILL_STRUCT_ITER,
        )
    ) -> None:
        with self._widg_log:
            print("Setting morphology mask.")
        with self._show_status():
            with self._gui__temporary_observation_toggle():
                self._gui__get_widget(_WName.MORPH_CLOSE.value).value = close
                self._gui__get_widget(_WName.MORPH_FILL.value).value = fill
                if close:
                    self._set_structuring_element(_WName.MORPH_CLOSE.name, closing_structure)
                    self._gui__get_widget(_WName.MORPH_CLOSE_ITER.value).value = closing_iterations
                    self._gui__get_widget(_WName.MORPH_CLOSE_BORDER.value).value = closing_border_value
                    # Set closing mask
                    self._custom_input[_WName.MORPH_CLOSE_MASK] = closing_mask
                    self._gui__get_widget(_WName.MORPH_CLOSE_MASK.value).layout.display = "none" if closing_mask is None else ""
                if fill:
                    self._set_structuring_element(_WName.MORPH_FILL.name, fill_structure)
            self._detector.set_mask_morphology(
                close=close,
                fill=fill,
                closing_structure=closing_structure,
                closing_iterations=closing_iterations,
                closing_mask=closing_mask,
                closing_border_value=closing_border_value,
                fill_structure=fill_structure,
            )
        self._gui__render(receptor_volume=True)
        return

    def set_mask_ligsite(
        self,
        count_lower: int | None = _Default.LIGSITE_COUNT_LOWER,
        count_upper: int | None = _Default.LIGSITE_COUNT_UPPER,
        dist_lower: float | None = _Default.LIGSITE_DIST_LOWER,
        dist_upper: float | None = _Default.LIGSITE_DIST_UPPER,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = _Default.LIGSITE_DIST_LOWER_MODE,
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = _Default.LIGSITE_DIST_UPPER_MODE,
    ) -> None:
        with self._widg_log:
            print("Setting LIGSITE mask.")
        with self._show_status():
            count_enabled = any(count is not None for count in (count_lower, count_upper))
            dist_enabled = any(dist is not None for dist in (dist_lower, dist_upper))
            with self._gui__temporary_observation_toggle():
                self._gui__get_widget(_WName.LIGSITE_COUNT.value).value = count_enabled
                self._gui__get_widget(_WName.LIGSITE_DIST.value).value = dist_enabled
                if count_enabled:
                    self._gui__get_widget(_WName.LIGSITE_COUNT_RANGE.value).value = (
                        max(count_lower, self.ligsite.psp_count_min) if count_lower is not None else self.ligsite.psp_count_min,
                        min(count_upper, self.ligsite.psp_count_max) if count_upper is not None else self.ligsite.psp_count_max,
                    )
                if dist_enabled:
                    self._gui__get_widget(_WName.LIGSITE_DIST_RANGE.value).value = (
                        max(dist_lower, self.ligsite.psp_distance_min) if dist_lower is not None else self.ligsite.psp_distance_min,
                        min(dist_upper, self.ligsite.psp_distance_max) if dist_upper is not None else self.ligsite.psp_distance_max,
                    )
                    self._gui__get_widget(_WName.LIGSITE_DIST_MIN.value).value = dist_lower_mode
                    self._gui__get_widget(_WName.LIGSITE_DIST_MAX.value).value = dist_upper_mode
            self._detector.set_mask_ligsite(
                count_lower=count_lower,
                count_upper=count_upper,
                dist_lower=dist_lower,
                dist_upper=dist_upper,
                dist_lower_mode=dist_lower_mode,
                dist_upper_mode=dist_upper_mode,
            )
        self._gui__render()
        return

    def set_mask_custom(self, mask: np.ndarray):
        mask = self._detector.set_mask_custom(mask=mask)
        return mask

    def unset_mask(self, *args: Literal["morphology", "ligsite", "custom"]) -> None:
        self._detector.unset_mask(*args)
        return

    @property
    def mask(self) -> jax.Array:
        return self._detector.mask

    @property
    def mask_morphology(self) -> jax.Array:
        return self._detector.mask_morphology

    @property
    def mask_ligsite(self) -> jax.Array | None:
        return self._detector.mask_ligsite

    @property
    def mask_custom(self) -> jax.Array | None:
        return self._detector.mask_custom

    @property
    def ligsite(self) -> LigSite | None:
        return self._detector.ligsite

    @property
    def field(self) -> Field:
        return self._detector.field

    @property
    def receptor(self) -> ChemicalSystem:
        return self._detector.receptor

    @property
    def nglwidget(self) -> NGLWidget:
        """The NGLWidget containing the protein structure and the pocket volume."""
        return self._widg_ngl

    def _ovc___morph_refresh(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"Morphology auto-refresh {'enabled' if enabled else 'disabled'}.")
        self._gui__toggle_widget_observation(
            observe=enabled,
            name_regex=f"^{_WPrefix.MORPH.value}.+",
        )
        if not enabled:
            return
        with self._show_status():
            self._morph__set_mask()
        return

    def _oc___morph_reset(self, _: widgets.Button):
        with self._widg_log:
            print("Morphology mask reset to default state.")
        with self._show_status(), self._gui__temporary_observation_toggle():
            self._gui__get_widget(_WName.MORPH_CLOSE.value).value = _Default.MORPH_CLOSE
            self._gui__get_widget(_WName.MORPH_FILL.value).value = _Default.MORPH_FILL
            self._gui__get_widget(_WName.MORPH_CLOSE_ITER.value).value = _Default.MORPH_CLOSE_ITER
            self._gui__get_widget(_WName.MORPH_CLOSE_BORDER.value).value = _Default.MORPH_CLOSE_BORDER
            self._gui__get_widget(_WName.MORPH_CLOSE_MASK.value).layout.display = "none"
            self._custom_input[_WName.MORPH_CLOSE_MASK] = None
            self._set_structuring_element(
                enum_prefix=_WName.MORPH_CLOSE.name,
                structure=(
                    _Default.MORPH_CLOSE_STRUCT_CONNECT,
                    _Default.MORPH_CLOSE_STRUCT_ITER
                )
            )
            self._set_structuring_element(
                enum_prefix=_WName.MORPH_FILL.name,
                structure=(
                    _Default.MORPH_FILL_STRUCT_CONNECT,
                    _Default.MORPH_FILL_STRUCT_ITER
                )
            )
            self._morph__set_mask()
        return

    def _ovc__morph_close(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"Morphological closing {'enabled' if enabled else 'disabled'}.")
        with self._show_status():
            self._morph__set_mask()
        return

    def _ovc__morph_close_border(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological closing border value set to {value}.")
        with self._show_status():
            self._morph__set_mask()
        return

    def _ovc__morph_close_iter(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological closing iterations set to {value}.")
        with self._show_status():
            self._morph__set_mask()
        return

    def _oc__morph_close_mask(self, _: widgets.Button):
        """Set a custom mask for morphological closing."""
        with self._widg_log:
            print("Deleting custom mask for morphological closing.")
        self._custom_input[_WName.MORPH_CLOSE_MASK] = None
        self._gui__get_widget(_WName.MORPH_CLOSE_MASK.value).layout.display = "none"
        with self._show_status():
            self._morph__set_mask()
        return

    def _ovc__morph_close_struct_connect(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological closing structure connectivity set to {value}.")
        with self._show_status():
            self._set_structuring_element(
                _WName.MORPH_CLOSE.name,
                structure=(value, self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_ITER.value).value)
            )
            self._morph__set_mask()
        return

    def _ovc__morph_close_struct_iter(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological closing structure iterations set to {value}.")
        with self._show_status():
            self._set_structuring_element(
                _WName.MORPH_CLOSE.name,
                structure=(
                    self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_CONNECT.value).value,
                    value
                )
            )
            self._morph__set_mask()
        return

    def _oc__morph_close_struct_custom(self, _: widgets.Button):
        """Set a custom structuring element for morphological closing."""
        with self._widg_log:
            print("Deleting custom structuring element for morphological closing.")
        self._custom_input[_WName.MORPH_CLOSE_STRUCT_CUSTOM] = None
        self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_CUSTOM.value).layout.display = "none"
        with self._show_status():
            self._morph__set_mask()
        return

    def _ovc__morph_fill(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"Morphological filling {'enabled' if enabled else 'disabled'}.")
        with self._show_status():
            self._morph__set_mask()
        return

    def _ovc__morph_fill_struct_connect(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological filling structure connectivity set to {value}.")
        with self._show_status():
            self._set_structuring_element(
                _WName.MORPH_FILL.name,
                structure=(value, self._gui__get_widget(_WName.MORPH_FILL_STRUCT_ITER.value).value)
            )
            self._morph__set_mask()
        return

    def _ovc__morph_fill_struct_iter(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological filling structure iterations set to {value}.")
        with self._show_status():
            self._set_structuring_element(
                _WName.MORPH_FILL.name,
                structure=(
                    self._gui__get_widget(_WName.MORPH_FILL_STRUCT_CONNECT.value).value,
                    value
                )
            )
            self._morph__set_mask()
        return

    def _oc__morph_fill_struct_custom(self, _: widgets.Button):
        """Set a custom structuring element for morphological filling."""
        with self._widg_log:
            print("Deleting custom structuring element for morphological filling.")
        self._custom_input[_WName.MORPH_FILL_STRUCT_CUSTOM] = None
        self._gui__get_widget(_WName.MORPH_FILL_STRUCT_CUSTOM.value).layout.display = "none"
        with self._show_status():
            self._morph__set_mask()
        return



    def _ovc__ligsite_refresh(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"LIGSITE auto-refresh {'enabled' if enabled else 'disabled'}.")
        self._gui__toggle_widget_observation(
            observe=enabled,
            name_regex=f"^{_WPrefix.LIGSITE.value}.+",
        )
        if not enabled:
            return
        with self._show_status():
            self._ligsite__set_mask()
        return

    def _oc__ligsite_reset(self, _: widgets.Button):
        with self._widg_log:
            print("LIGSITE mask reset to default state.")
        with self._show_status(), self._gui__temporary_observation_toggle():
            self._gui__get_widget(_WName.LIGSITE_COUNT.value).value = _Default.LIGSITE_COUNT
            self._gui__get_widget(_WName.LIGSITE_DIST.value).value = _Default.LIGSITE_DIST
            self._gui__get_widget(_WName.LIGSITE_COUNT_RANGE.value).value = (
                max(_Default.LIGSITE_COUNT_LOWER, self.ligsite.psp_count_min),
                min(_Default.LIGSITE_COUNT_UPPER, self.ligsite.psp_count_max)
            )
            self._gui__get_widget(_WName.LIGSITE_DIST_RANGE.value).value = (
                max(_Default.LIGSITE_DIST_LOWER, self.ligsite.psp_distance_min) if _Default.LIGSITE_DIST_LOWER is not None else self.ligsite.psp_distance_min,
                min(_Default.LIGSITE_DIST_UPPER, self.ligsite.psp_distance_max) if _Default.LIGSITE_DIST_UPPER is not None else self.ligsite.psp_distance_max,
            )
            for side in ("min", "max"):
                default_dist = _Default.LIGSITE_DIST_LOWER if side == "min" else _Default.LIGSITE_DIST_UPPER
                default_mode = _Default.LIGSITE_DIST_LOWER_MODE if side == "min" else _Default.LIGSITE_DIST_UPPER_MODE
                self._gui__get_widget(_WName[f"LIGSITE_DIST_{side.upper()}"].value).value = None if default_dist is None else default_mode
            self._ligsite__set_mask()
        return {}

    def _ovc__ligsite_count(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"PSP count mask {'enabled' if enabled else 'disabled'}.")
        with self._show_status():
            self._ligsite__set_mask()
        return

    def _ovc__ligsite_count_range(self, change: dict):
        with self._widg_log:
            print(f"LIGSITE PSP count range changed to {change['new']}.")
        with self._show_status():
            self._ligsite__set_mask()
        return

    def _ovc__ligsite_dist(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"PSP distance mask {'enabled' if enabled else 'disabled'}.")
        with self._show_status():
            self._ligsite__set_mask()
        return

    def _ovc__ligsite_dist_range(self, change: dict):
        with self._widg_log:
            print(f"LIGSITE PSP distance range changed to {change['new']}.")
        with self._show_status():
            self._ligsite__set_mask()
        return {}

    def _ovc__ligsite_dist_min(self, change: dict):
        with self._widg_log:
            print(f"LIGSITE PSP distance lower mode changed to {change['new']}.")
        with self._show_status():
            self._ligsite__set_mask()
        return

    def _ovc__ligsite_dist_max(self, change: dict):
        with self._widg_log:
            print(f"LIGSITE PSP distance upper mode changed to {change['new']}.")
        with self._show_status():
            self._ligsite__set_mask()
        return

    def _morph__set_mask(self):
        """Set the morphology mask based on the current GUI settings."""
        with self._widg_log:
            print("Recalculating morphology mask with current settings.")
        custom_closing_structure = self._custom_input[_WName.MORPH_CLOSE_STRUCT_CUSTOM]
        custom_fill_structure = self._custom_input[_WName.MORPH_FILL_STRUCT_CUSTOM]
        closing_structure = (
            self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_CONNECT.value).value,
            self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_ITER.value).value,
        ) if custom_closing_structure is None else custom_closing_structure
        fill_structure = (
            self._gui__get_widget(_WName.MORPH_FILL_STRUCT_CONNECT.value).value,
            self._gui__get_widget(_WName.MORPH_FILL_STRUCT_ITER.value).value,
        ) if custom_fill_structure is None else custom_fill_structure
        self._detector.set_mask_morphology(
            close=self._gui__get_widget(_WName.MORPH_CLOSE.value).value,
            fill=self._gui__get_widget(_WName.MORPH_FILL.value).value,
            closing_structure=closing_structure,
            closing_iterations=self._gui__get_widget(_WName.MORPH_CLOSE_ITER.value).value,
            closing_mask=self._custom_input[_WName.MORPH_CLOSE_MASK],
            closing_border_value=self._gui__get_widget(_WName.MORPH_CLOSE_BORDER.value).value,
            fill_structure=fill_structure,
        )
        self._gui__render(receptor_volume=True)
        return

    def _ligsite__set_mask(self):
        """Set the LIGSITE mask based on the current GUI settings."""
        with self._widg_log:
            print("Recalculating LIGSITE mask with current settings.")
        count_enabled = self._gui__get_widget(_WName.LIGSITE_COUNT.value).value
        dist_enabled = self._gui__get_widget(_WName.LIGSITE_DIST.value).value
        count_range = self._gui__get_widget(_WName.LIGSITE_PSP_COUNT_RANGE.value)
        dist_range = self._gui__get_widget(_WName.LIGSITE_PSP_DIST_RANGE.value)
        dist_min = self._gui__get_widget(_WName.LIGSITE_PSP_DIST_MIN.value)
        dist_max = self._gui__get_widget(_WName.LIGSITE_PSP_DIST_MAX.value)
        self._detector.set_mask_ligsite(
            count_lower=count_range.lower if count_enabled else None,
            count_upper=count_range.upper if count_enabled else None,
            dist_lower=dist_range.lower if dist_enabled and dist_min.value else None,
            dist_upper=dist_range.upper if dist_enabled and dist_max.value else None,
            dist_lower_mode=dist_min.value,
            dist_upper_mode=dist_max.value,
        )
        self._gui__render()
        return

    def _set_structuring_element(
        self,
        enum_prefix: str,
        structure: np.ndarray | tuple[int, int],
    ) -> None:
        enum_prefix = f"{enum_prefix.upper()}_STRUCT_"
        custom_structure_enum = _WName[f"{enum_prefix}CUSTOM"]
        w_connect = self._gui__get_widget(_WName[f"{enum_prefix}CONNECT"].value)
        w_iter = self._gui__get_widget(_WName[f"{enum_prefix}ITER"].value)
        w_custom = self._gui__get_widget(custom_structure_enum.value)
        if isinstance(structure, tuple):
            self._custom_input[custom_structure_enum] = None
            w_connect.value = structure[0]
            w_iter.value = structure[1]
            w_custom.layout.display = "none"
        else:
            self._custom_input[custom_structure_enum] = structure
            w_connect.disabled = True
            w_iter.disabled = True
            w_custom.layout.display = ""
        return

    def _gui__render(self, receptor_volume: bool = False):
        """Update the NGLWidget with the current mask.

        This method is automatically called by the `GUI` parent class when needed.
        """
        with self._show_status("Rendering..."):
            name = "Pocket Selection"
            ngl = self.nglwidget
            ngl.remove_component_by_name(name)
            ngl.add_spheres(
                coords=self.field.grid.coordinates[self.mask],
                name=name,
            )
            if receptor_volume:
                receptor_volume_added_name = "Receptor Volume (Added)"
                ngl.remove_component_by_name(receptor_volume_added_name)
                ngl.add_spheres(
                    name=receptor_volume_added_name,
                    coords=self.field.grid.coordinates[self._detector.receptor_volume_added],
                    colors=(0, 100, 0)
                )
                receptor_volume_removed_name = "Receptor Volume (Removed)"
                ngl.remove_component_by_name(receptor_volume_removed_name)
                ngl.add_spheres(
                    name=receptor_volume_removed_name,
                    coords=self.field.grid.coordinates[self._detector.receptor_volume_removed],
                    colors=(100, 0, 0)
                )
        return

    @contextmanager
    def _show_status(self, status: str = "Calculating..."):
        """Show the status."""
        current_status = self._widg_status.value
        is_idle = current_status == "Idle"
        self._widg_status.value = status
        if is_idle:
            self._widg_status.remove_class("statusbar-idle")
            self._widg_status.add_class("statusbar-running")
        try:
            yield
        finally:
            self._widg_status.value = current_status
            if is_idle:
                self._widg_status.remove_class("statusbar-running")
                self._widg_status.add_class("statusbar-idle")
        return


def from_chemsys(
    system: ChemicalSystem,
    *,
    field: Field | None = None,
    grid: int | float | Sequence[int | float] | Grid = 0.5,
    minimize_aabb: bool = True,
    gui: bool = False,
) -> GridDetector | GridDetectorGUI:
    """Create a grid-based pocket detector from a chemical system.

    Parameters
    ----------
    system
        A `ChemicalSystem` object containing the receptor structure.
    field
        An optional `Field` representing the receptor's voxel grid.
        If provided, it will be used directly
        and all other parameters below will be ignored.
        If not provided, the field will be generated from the receptor.
    grid
        The grid spacing for the voxel grid.
        This can be a single value (e.g. `0.5` for 0.5 Ångstrom spacing),
        or a Grid object specifying the grid.
    minimize_aabb
        Whether to minimize the axis-aligned bounding box (AABB) of the receptor
        before creating the voxel grid, in order to reduce the size of the grid.
    gui
        Whether to create a GUI for the grid detector.
    """
    if not field:
        if minimize_aabb:
            system = system.new(trajectory=system.trajectory.minimize_aabb())
        field = system.toxelate(grid=grid)
    detector = GridDetector(receptor=system, field=field)
    if not gui:
        return detector
    detector_gui = GridDetectorGUI(detector)
    detector_gui.display()
    return detector_gui
