"""Grid-based binding pocket detection."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

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

class GridDetector:
    def __init__(self, receptor: ChemicalSystem, field: Field):
        self._receptor = receptor
        self._field = field
        self._grid_axis_indices = tuple(range(self.field.batch_ndim, self.field.tensor.ndim))

        self._mask_morphology = np.logical_not(self.field.tensor)
        self._mask_ligsite: np.ndarray | None = None
        self._mask_custom: np.ndarray | None = None

        self._ligsite: LigSite | None = None
        self._gui = None
        return

    def set_mask_morphology(
        self,
        close: bool = True,
        fill: bool = True,
        closing_structure: np.ndarray | tuple[int, int] = (1, 1),
        closing_iterations: int = 1,
        closing_mask: np.ndarray | None = None,
        closing_border_value: Literal[0, 1] = 1,
        fill_structure: np.ndarray | tuple[int, int] = (1, 1),
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
        self._mask_morphology = jnp.logical_not(volume_closed_and_filled)
        return self._mask_morphology

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
    def ligsite(self) -> LigSite | None:
        return self._ligsite

    @property
    def field(self) -> Field:
        return self._field

    @property
    def receptor(self) -> ChemicalSystem:
        return self._receptor

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
        .togglebutton-on {
            background-color: rgb(0 100 0) !important;
            color: white !important;
        }
        .togglebutton-off {
            background-color: rgb(100 100 100) !important;
            color: white !important;
        }
    </style>"""

    def __init__(self, receptor: ChemicalSystem, field: Field):

        def make_ngl() -> NGLWidget:
            nglwidget = scishow.nglview.NGLWidget().display(gui=True)
            nglwidget.add_trajectory(receptor, name="Receptor")
            # nglwidget.component_0.add_surface(
            #     color="rgb(100,20,20)",
            #     opacity=0.5,
            #     surface_type="vws",
            #     scale_factor=10,
            #     smooth=10
            # )
            self._nglwidget = nglwidget
            return nglwidget

        def make_morphology_panel():

            def make_closing_panel():
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
                    name=f"{name_prefix}close",
                    widget=scishow.widgets.toggle_button(
                        "Close",
                        value=False,
                        disabled=False,
                        tooltip="Apply morphological closing to the protein volume.",
                    )
                )
                structure_connectivity = self._gui__add_widget(
                    name=f"{name_prefix}closing_structure_connectivity",
                    widget=widgets.Dropdown(
                        options=[1, 2, 3],
                        value=2,
                        layout=widgets.Layout(width="100%"),
                        disabled=True,
                    )
                )
                structure_iterations = self._gui__add_widget(
                        name=f"{name_prefix}closing_structure_iterations",
                        widget=widgets.IntSlider(
                            value=1,
                            min=1,
                            max=100,
                            step=1,
                            disabled=True,
                            continuous_update=False,
                            orientation="horizontal",
                            readout=True,
                            readout_format="d",
                            layout=widgets.Layout(width="100%"),
                        )
                    )
                closing_iterations = self._gui__add_widget(
                    name=f"{name_prefix}closing_iterations",
                    widget=widgets.IntSlider(
                        value=1,
                        min=1,
                        max=100,
                        step=1,
                        disabled=True,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                border_value = self._gui__add_widget(
                    name=f"{name_prefix}closing_border_value",
                    widget=widgets.Dropdown(
                        options=[0, 1],
                        value=1,
                        layout=widgets.Layout(width="100%"),
                        disabled=True,
                    )
                )
                custom_structure = self._gui__add_widget(
                    name=f"{name_prefix}closing_structure_custom",
                    widget=widgets.Button(
                        description="Custom Structure",
                        disabled=True,
                        icon="trash",
                        layout=widgets.Layout(display="none")
                    )
                )
                custom_mask = self._gui__add_widget(
                    name=f"{name_prefix}closing_mask",
                    widget=widgets.Button(
                        description="Custom Mask",
                        disabled=True,
                        icon="trash",
                        layout=widgets.Layout(display="none")
                    )
                )
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

            def make_fill_panel():
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
                    name=f"{name_prefix}fill",
                    widget=scishow.widgets.toggle_button(
                        "Fill Holes",
                        value=False,
                        disabled=False,
                        tooltip="Fill holes in the protein volume after closing.",
                    )
                )
                structure_connectivity = self._gui__add_widget(
                    name=f"{name_prefix}fill_structure_connectivity",
                    widget=widgets.Dropdown(
                        options=[1, 2, 3],
                        value=1,
                        layout=widgets.Layout(width="100%"),
                        disabled=True,
                    )
                )
                structure_iterations = self._gui__add_widget(
                        name=f"{name_prefix}fill_structure_iterations",
                        widget=widgets.IntSlider(
                            value=1,
                            min=1,
                            max=100,
                            step=1,
                            disabled=True,
                            continuous_update=False,
                            orientation="horizontal",
                            readout=True,
                            readout_format="d",
                            layout=widgets.Layout(width="100%"),
                        )
                    )
                custom_structure = self._gui__add_widget(
                    name=f"{name_prefix}closing_structure_custom",
                    widget=widgets.Button(
                        description="Custom Structure",
                        disabled=True,
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

            name_prefix = "morphology_"
            panels = widgets.HBox(
                    [make_closing_panel(), make_fill_panel()],
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="space-between",
                        flex_flow="row wrap",
                        margin="10px 0 10px 0"
                    )
                )
            return widgets.VBox(
                [make_top_panel(name_prefix), panels],
                layout=widgets.Layout(width="100%", overflow="hidden"),
            )

        def make_ligsite_panel():
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
                        scishow.widgets.labeled_widget(
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

            directions = scishow.widgets.labeled_widget(
                    value="Directions:",
                    widget=self._gui__add_widget(
                        name=f"{name_prefix}psp_dirs",
                        widget=widgets.Dropdown(
                            options={"None": None, "1D": 1, "2D": 2, "3D": 3},
                            layout=widgets.Layout(width="flex-grow", min_width="20px"),
                        )
                    )
                )

            return widgets.VBox(
                [make_top_panel(name_prefix), make_psp_panels()],
                layout=widgets.Layout(width="100%", overflow="hidden"),
            )

        def make_logger_panel():
            self._gui__logger = widgets.Output()
            return widgets.Accordion(
                children=[self._gui__logger],
                titles=["Logs"],
                selected_index=None,
            )

        def make_top_panel(name_prefix: str):
            button_layout = widgets.Layout(min_width="70px", max_width="70px", flex="0 0 auto")
            refresh = self._gui__add_widget(
                name=f"{name_prefix}refresh",
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
                name=f"{name_prefix}reset",
                widget=widgets.Button(
                    description="Reset",
                    tooltip="Reset the morphology mask to the default state.",
                    button_style="danger",
                    disabled=False,
                    layout=button_layout,
                )
            )
            reset.add_class("button-bold")
            buttons_box = widgets.HBox(
                [refresh, reset],
                layout=widgets.Layout(justify_content="flex-end", flex="0 0 auto")
            )
            return widgets.HBox(
                [self._status_widget, buttons_box],
                layout=widgets.Layout(width="100%", align_items="center", justify_content="space-between")
            )

        super().__init__()
        self._detector = GridDetector(receptor=receptor, field=field)

        self._status_widget = widgets.Label(value="Idle")
        self._status_widget.add_class("statusbar")
        self._status_widget.add_class("statusbar-idle")
        control_tabs = widgets.Tab(
            children=[make_morphology_panel(), make_ligsite_panel()],
            titles=["Morphology", "LIGSITE"],
            selected_index=0,
        )
        css_style = display.HTML(self._CSS_STYLE)
        self._gui__set_main_widget((css_style, control_tabs, make_ngl(), make_logger_panel()))
        self._morphology_closing_structure_custom = None
        self._morphology_closing_mask = None
        self._morphology_fill_structure_custom = None
        return

    def set_mask_morphology(
        self,
        close: bool = True,
        fill: bool = True,
        closing_structure: np.ndarray | tuple[int, int] = (1, 1),
        closing_iterations: int = 1,
        closing_mask: np.ndarray | None = None,
        closing_border_value: Literal[0, 1] = 1,
        fill_structure: np.ndarray | tuple[int, int] = (1, 1),
    ) -> None:

        with self._gui__logger:
            print("Setting morphology mask.")
        w_close_struct_connect = self._gui__get_widget("morphology_closing_structure_connectivity")
        w_close_struct_iter = self._gui__get_widget("morphology_closing_structure_iterations")
        w_close_iter = self._gui__get_widget("morphology_closing_iterations")
        w_close_struct_custom = self._gui__get_widget("morphology_closing_structure_custom")
        w_close_mask = self._gui__get_widget("morphology_closing_mask")
        w_fill_struct_connect = self._gui__get_widget("morphology_fill_structure_connectivity")
        w_fill_struct_iter = self._gui__get_widget("morphology_fill_structure_iterations")
        w_fill_struct_custom = self._gui__get_widget("morphology_fill_structure_custom")
        with self._show_status(), self._gui__temporary_observation_toggle():
            self._gui__get_widget("morphology_close").value = close
            self._gui__get_widget("morphology_fill").value = fill
            if close:
                w_close_iter.value = closing_iterations
                if isinstance(closing_structure, tuple):
                    self._morphology_closing_structure_custom = None
                    w_close_struct_connect.value = closing_structure[0]
                    w_close_struct_iter.value = closing_structure[1]
                    w_close_struct_custom.layout.display = "none"
                else:
                    self._morphology_closing_structure_custom = closing_structure
                    w_close_struct_connect.disabled = True
                    w_close_struct_iter.disabled = True
                    w_close_struct_custom.layout.display = ""
            # Set closing mask
            self._morphology_closing_mask = closing_mask
            w_close_mask.layout.display = "none" if closing_mask is None else ""
            if fill:
                if isinstance(fill_structure, tuple):
                    self._morphology_fill_structure_custom = None
                    w_fill_struct_connect.value = fill_structure[0]
                    w_fill_struct_iter.value = fill_structure[1]
                    w_fill_struct_custom.layout.display = "none"
                else:
                    self._morphology_fill_structure_custom = fill_structure
                    w_fill_struct_connect.disabled = True
                    w_fill_struct_iter.disabled = True
                    w_fill_struct_custom.layout.display = ""
            self._detector.set_mask_morphology(
                close=close,
                fill=fill,
                closing_structure=closing_structure,
                closing_iterations=closing_iterations,
                closing_mask=closing_mask,
                closing_border_value=closing_border_value,
                fill_structure=fill_structure,
            )
        self._gui__render()
        return

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
        mask = self._detector.set_mask_ligsite(
            count_lower=count_lower,
            count_upper=count_upper,
            dist_lower=dist_lower,
            dist_upper=dist_upper,
            dist_lower_mode=dist_lower_mode,
            dist_upper_mode=dist_upper_mode,
            directions=directions,
        )
        return mask

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
        return self._nglwidget

    def _ovc__morphology_refresh(self, change: dict):
        enabled = change["new"]
        with self._gui__logger:
            print(f"Morphology auto-refresh {'enabled' if enabled else 'disabled'}.")
        button = change["owner"]
        button.button_style = "success" if enabled else "danger"
        self._gui__toggle_widget_observation(
            observe=enabled,
            name_regex="^morphology_.+",
        )

        with self._gui__logger:
            print("Refreshing morphology mask with current settings.")
        with self._show_status():
            close = self._gui__get_widget("morphology_close").value
            fill = self._gui__get_widget("morphology_fill").value
            closing_structure = (
                self._gui__get_widget("morphology_closing_structure_connectivity").value,
                self._gui__get_widget("morphology_closing_structure_iterations").value,
            )
            closing_iterations = self._gui__get_widget("morphology_closing_iterations").value
            fill_structure = (
                self._gui__get_widget("morphology_fill_structure_connectivity").value,
                self._gui__get_widget("morphology_fill_structure_iterations").value,
            )
            with self._gui__temporary_toggle():
                self.set_mask_morphology(
                    close=close,
                    fill=fill,
                    closing_structure=closing_structure,
                    closing_iterations=closing_iterations,
                    fill_structure=fill_structure,
                )
        return {}

    def _oc__morphology_reset(self, _: widgets.Button):
        with self._gui__logger:
            print("Morphology mask reset to default state.")
        with self._show_status(), self._gui__temporary_toggle():
            self._gui__get_widget("morphology_close").value = True
            self._gui__get_widget("morphology_fill").value = True
            self._gui__get_widget("morphology_closing_structure_connectivity").value = 2
            self._gui__get_widget("morphology_closing_structure_iterations").value = 1
            self._gui__get_widget("morphology_closing_iterations").value = 1
            self._gui__get_widget("morphology_fill_structure_connectivity").value = 1
            self._gui__get_widget("morphology_fill_structure_iterations").value = 1
            self.set_mask_morphology(
                close=True,
                fill=True,
                closing_structure=(2, 1),
                closing_iterations=1,
            )
        self._gui__toggle_widget_availability(
            available=True,
            name_regex="^morphology_.+",
        )
        return {}

    def _oc__ligsite_auto_refresh(self, change: dict):
        enabled = change["new"]
        with self._gui__logger:
            print(f"LIGSITE auto-refresh {'enabled' if enabled else 'disabled'}.")
        button = change["owner"]
        button.button_style = "success" if enabled else "danger"
        self._gui__toggle_widget_observation(
            observe=enabled,
            name_regex="^ligsite_.*",
        )
        return

    def _oc__ligsite_refresh(self, _: widgets.Button):
        with self._gui__logger:
            print("Refreshing LIGSITE mask with current settings.")
        with self._show_status():
            directions = self._gui__get_widget("ligsite_psp_dirs")
            if directions.value:
                with self._gui__temporary_toggle():
                    self._ligsite__recalculate_mask(pass_directions=True)
            else:
                self._gui__toggle_widget_availability(
                    available=False,
                    name_regex="^ligsite_psp_(?!dirs$).+",
                )
                self.unset_mask("ligsite")
        return {}

    def _oc__ligsite_reset(self, _: widgets.Button):
        with self._gui__logger:
            print("Resetting LIGSITE mask to default state.")
        with self._show_status(), self._gui__temporary_toggle():
            self._gui__get_widget("ligsite_psp_dirs").value = 3
            self.set_mask_ligsite(directions=tuple(range(1, 4)))
            count_slider = self._gui__get_widget("ligsite_psp_count_slider")
            count_slider_min = self.ligsite.psp_count.min().item()
            count_slider_max = self.ligsite.psp_count.max().item()
            dist_slider = self._gui__get_widget("ligsite_psp_dist_slider")
            dist_slider_min = jnp.nanmin(self.ligsite.psp_distance).item()
            dist_slider_max = jnp.nanmax(self.ligsite.psp_distance).item()
            self._gui__reset_slider_minmax(
                slider=count_slider,
                minimum=count_slider_min,
                maximum=count_slider_max,
            )
            self._gui__reset_slider_minmax(
                slider=dist_slider,
                minimum=dist_slider_min,
                maximum=dist_slider_max,
            )
            count_slider.lower = max(count_slider.lower, 5)
            dist_slider.upper = min(dist_slider.upper, 10.0)
            self._gui__get_widget("ligsite_psp_count_min").value = True
            self._gui__get_widget("ligsite_psp_count_max").value = False
            self._gui__get_widget("ligsite_psp_dist_min").value = "all"
            self._gui__get_widget("ligsite_psp_dist_max").value = "any"
            self._ligsite__recalculate_mask(pass_directions=False)
        self._gui__toggle_widget_availability(
            available=True,
            name_regex="^ligsite_psp_.+",
        )
        return {}

    def _ovc__ligsite_psp_dirs(self, change: dict):
        with self._gui__logger:
            print(f"LIGSITE directions changed to {change['new']}.")
        with self._show_status():
            value = change["new"]
            if not value:
                # If no directions are selected, disable controls and unset the mask.
                self._gui__toggle_widget_availability(
                    available=False,
                    name_regex="^ligsite_psp_(?!dirs$).+",
                )
                self.unset_mask("ligsite")
                return {}
            with self._gui__temporary_toggle():
                    # Calculate mask to get new PSP min/max values.
                    self.set_mask_ligsite(directions=tuple(range(1, value + 1)))
                    # Reset the sliders to the new min/max values.
                    self._gui__reset_slider_minmax(
                        slider=self._gui__get_widget("ligsite_psp_count_slider"),
                        minimum=self.ligsite.psp_count.min().item(),
                        maximum=self.ligsite.psp_count.max().item(),
                    )
                    self._gui__reset_slider_minmax(
                        slider=self._gui__get_widget("ligsite_psp_dist_slider"),
                        minimum=jnp.nanmin(self.ligsite.psp_distance).item(),
                        maximum=jnp.nanmax(self.ligsite.psp_distance).item(),
                    )
                    auto_refresh = self._gui__get_widget("ligsite_auto_refresh").value
                    if auto_refresh:
                        # Recalculate mask with the current min/max values when auto-refresh is enabled.
                        self._ligsite__recalculate_mask(pass_directions=False)
            self._gui__toggle_widget_availability(
                available=True,
                name_regex="^ligsite_.+",
            )
        return {} if auto_refresh else None

    def _ovc__ligsite_psp_count_slider(self, change: dict):
        with self._gui__logger:
            print(f"LIGSITE PSP count range changed to {change['new']}.")
        with self._show_status(), self._gui__temporary_toggle():
            old_lower, old_upper = change["old"]
            new_lower, new_upper = change["new"]
            min_enabled = self._gui__get_widget("ligsite_psp_count_min").value
            max_enabled = self._gui__get_widget("ligsite_psp_count_max").value
            self.set_mask_ligsite(
                count_lower=(new_lower if new_lower != old_lower else None) if min_enabled else True,
                count_upper=(new_upper if new_upper != old_upper else None) if max_enabled else True,
            )
        return {}

    def _ovc__ligsite_psp_count_min(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP count lower mode to {change['new']}.")
        with self._show_status(), self._gui__temporary_toggle():
            min_enabled = change["new"]
            lower = self._gui__get_widget("ligsite_psp_count_slider").lower
            self.set_mask_ligsite(count_lower=lower if min_enabled else True)
        return {}

    def _ovc__ligsite_psp_count_max(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP count upper mode to {change['new']}.")
        with self._show_status(), self._gui__temporary_toggle():
            max_enabled = change["new"]
            upper = self._gui__get_widget("ligsite_psp_count_slider").upper
            self.set_mask_ligsite(count_upper=upper if max_enabled else True)
        return {}

    def _ovc__ligsite_psp_dist_slider(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP distance slider to {change['new']}.")
        with self._show_status(), self._gui__temporary_toggle():
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
        return {}

    def _ovc__ligsite_psp_dist_min(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP distance lower mode to {change['new']}.")
        with self._show_status(), self._gui__temporary_toggle():
            new_mode = change["new"]
            lower = self._gui__get_widget("ligsite_psp_dist_slider").lower
            self.set_mask_ligsite(
                dist_lower=lower if new_mode else True,
                dist_lower_mode=new_mode,
            )
        return {}

    def _ovc__ligsite_psp_dist_max(self, change: dict):
        with self._gui__logger:
            print(f"Changed LIGSITE PSP distance upper mode to {change['new']}.")
        with self._show_status(), self._gui__temporary_toggle():
            new_mode = change["new"]
            upper = self._gui__get_widget("ligsite_psp_dist_slider").upper
            self.set_mask_ligsite(
                dist_upper=upper if new_mode else True,
                dist_upper_mode=new_mode,
            )
        return {}

    def _morphology_set_mask(self):
        """Set the morphology mask based on the current GUI settings."""
        custom_closing_structure = self._morphology_closing_structure_custom
        custom_fill_structure = self._morphology_fill_structure_custom
        closing_structure = (
            self._gui__get_widget("morphology_closing_structure_connectivity").value,
            self._gui__get_widget("morphology_closing_structure_iterations").value,
        ) if custom_closing_structure is None else custom_closing_structure
        fill_structure = (
            self._gui__get_widget("morphology_fill_structure_connectivity").value,
            self._gui__get_widget("morphology_fill_structure_iterations").value,
        ) if custom_fill_structure is None else custom_fill_structure
        self._detector.set_mask_morphology(
                close=self._gui__get_widget("morphology_close").value,
                fill=self._gui__get_widget("morphology_fill").value,
                closing_structure=closing_structure,
                closing_iterations=self._gui__get_widget("morphology_closing_iterations").value,
                closing_mask=self._morphology_closing_mask,
                closing_border_value=self._gui__get_widget("morphology_closing_border_value").value,
                fill_structure=fill_structure,
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

    @contextmanager
    def _show_status(self, status: str = "Calculating..."):
        """Show the status."""
        self._status_widget.value = status
        self._status_widget.remove_class("statusbar-idle")
        self._status_widget.add_class("statusbar-running")
        try:
            yield
        finally:
            self._status_widget.value = "Idle"
            self._status_widget.remove_class("statusbar-running")
            self._status_widget.add_class("statusbar-idle")
        return


def from_chemsys(
    system: ChemicalSystem,
    *,
    field: Field | None = None,
    grid: int | float | Sequence[int | float] | Grid = 0.5,
    minimize_aabb: bool = True,
    gui: bool = False,
) -> GridDetectorGUI:
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
    if gui:
        detector = GridDetectorGUI(receptor=system, field=field)
        detector.display()
        return detector
    return GridDetector(receptor=system, field=field)
