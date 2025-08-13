from __future__ import annotations

from typing import TYPE_CHECKING
from enum import Enum
from contextlib import contextmanager

from IPython import display
import ipywidgets as widgets
import numpy as np

import scishow

from caddpy.pocket.detector import Detector
from caddpy.pocket.default import Default

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal, Callable, Any
    from scishow.nglview import NGLWidget
    from scids.field import Field
    from caddpy.typing import JAXArray
    from caddpy.chemsys import ChemicalSystem
    from caddpy.pocket.ligsite import LigSite
    from caddpy.pocket.pockets import Pockets


class DetectorGUI(scishow.widgets.GUI):

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

    def __init__(
        self,
        detector: Detector,
        ngl_name_receptor: str = "Receptor",
        ngl_name_receptor_volume_original: str = "Volume (Original)",
        ngl_name_receptor_volume_added: str = "Volume (Added)",
        ngl_name_receptor_volume_removed: str = "Volume (Removed)",
        ngl_name_mask: str = "Mask",
        ngl_name_pocket: str = "P "
    ):

        def widget_status() -> widgets.Label:
            """Create a status widget to display the current status of the detector."""
            widget = widgets.Label(value="Idle")
            widget.add_class("statusbar")
            widget.add_class("statusbar-idle")
            return widget

        def widget_ngl() -> NGLWidget:
            nglwidget = scishow.nglview.NGLWidget().display(gui=True)
            nglwidget.add_trajectory(self.receptor, name=ngl_name_receptor)
            nglwidget.add_volume(
                self.field.tensor.astype(bool).astype(np.uint8),
                name=ngl_name_receptor_volume_original,
                basis=self.field.grid.unit_vectors,
                origin=self.field.grid.lower_bounds,
                representation_params=scishow.nglview.SurfaceRepresentationParameters(
                    lazy=True,
                    opacity=0.7,
                    contour=False,
                    visible=False,
                    color=(100,100,100),
                    isolevel=0.5,
                    isolevel_type="value",
                )
            )
            nglwidget.component_0.add_surface(
                color="rgb(100,20,20)",
                opacity=0.5,
                surface_type="vws",
                scale_factor=0,
            )
            return nglwidget

        def tab_morph():
            panels = widgets.HBox(
                [
                    make_morphology_panel(
                        "closing",
                        enum_prefix="MORPH_CLOSE",
                        tooltip=f"Apply morphological closing to the protein volume."
                    ),
                    toggle_button(
                        name=_WName.MORPH_FILL,
                        text="Hole Filling",
                        value=Default.MORPH_FILL,
                        tooltip="Fill holes in the protein volume after morphological closing."
                    )
                ],
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

                slider = self._gui__add_widget(
                    name=_WName.LIGSITE_COUNT_RANGE.value,
                    widget=widgets.IntRangeSlider(
                        value=(
                            max(Default.LIGSITE_COUNT_LOWER, self.ligsite.psp_count_min),
                            min(Default.LIGSITE_COUNT_UPPER, self.ligsite.psp_count_max)
                        ),
                        min=self.ligsite.psp_count_min,
                        max=self.ligsite.psp_count_max,
                        step=1,
                        disabled=not Default.LIGSITE_COUNT,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        layout=widgets.Layout(width="100%"),
                    )
                )
                return widgets.VBox(
                    [
                        toggle_button(
                            name=_WName.LIGSITE_COUNT,
                            text="PSP Count",
                            value=Default.LIGSITE_COUNT,
                            tooltip="Apply PSP count mask to the protein volume.",
                            observer=on_toggle,
                        ),
                        slider
                    ],
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

                slider = self._gui__add_widget(
                    name=_WName.LIGSITE_DIST_RANGE.value,
                    widget=widgets.FloatRangeSlider(
                        value=(
                            max(Default.LIGSITE_DIST_LOWER, self.ligsite.psp_distance_min) if Default.LIGSITE_DIST_LOWER is not None else self.ligsite.psp_distance_min,
                            min(Default.LIGSITE_DIST_UPPER, self.ligsite.psp_distance_max) if Default.LIGSITE_DIST_UPPER is not None else self.ligsite.psp_distance_max,
                        ),
                        min=self.ligsite.psp_distance_min,
                        max= self.ligsite.psp_distance_max,
                        step=0.01,
                        disabled=not Default.LIGSITE_DIST,
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
                    default_dist = Default.LIGSITE_DIST_LOWER if side == "min" else Default.LIGSITE_DIST_UPPER
                    default_mode = Default.LIGSITE_DIST_LOWER_MODE if side == "min" else Default.LIGSITE_DIST_UPPER_MODE
                    dropdown = self._gui__add_widget(
                        name=_WName[f"LIGSITE_DIST_{side.upper()}"].value,
                        widget=widgets.Dropdown(
                            options={
                                "Any": "any",
                                "All": "all",
                                "Max": "max",
                                "Min": "min",
                                "Mean": "mean",
                                "Off": False,
                            },
                            value=False if default_dist is None else default_mode,
                            layout=widgets.Layout(width="100%"),
                            disabled=not Default.LIGSITE_DIST,
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
                return widgets.VBox(
                    [
                        toggle_button(
                            name=_WName.LIGSITE_DIST,
                            text="PSP Distance",
                            value=Default.LIGSITE_DIST,
                            tooltip="Apply PSP distance mask to the protein volume.",
                            observer=on_toggle,
                        ),
                        slider,
                        minmax_settings
                    ],
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

        def tab_extract():
            panels = widgets.HBox(
                [
                    make_morphology_panel(
                        typ="opening",
                        enum_prefix="EXTRACT_OPEN",
                        tooltip="Apply morphological opening to the pocket mask."
                    ),
                ],
                layout=widgets.Layout(
                    width="100%",
                    justify_content="space-between",
                    flex_flow="row wrap",
                    margin="10px 0 10px 0"
                )
            )
            return widgets.VBox(
                [panel_top(_WPrefix.EXTRACT), panels],
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

        def make_morphology_panel(
            typ: Literal["dilation", "erosion", "closing", "opening"],
            enum_prefix: str,
            tooltip: str,
        ):
            toggle_disabled = not Default[enum_prefix]
            iterations = self._gui__add_widget(
                name=_WName[f"{enum_prefix}_ITER"].value,
                widget=widgets.IntSlider(
                    value=Default[f"{enum_prefix}_ITER"],
                    min=1,
                    max=100,
                    step=1,
                    disabled=toggle_disabled,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                    layout=widgets.Layout(width="100%"),
                )
            )
            custom_mask = self._gui__add_widget(
                name=_WName[f"{enum_prefix}_MASK"].value,
                widget=widgets.Button(
                    description="Custom Mask",
                    icon="trash",
                    layout=widgets.Layout(display="none")
                )
            )
            custom_mask.add_class("resetbutton")
            iterations_labeled = scishow.widgets.labeled_widget(
                value="Iterations:",
                widget=iterations
            )
            return make_structure_panel(
                title=typ,
                enum_prefix=enum_prefix,
                on_toggle_widgets=(
                    iterations,
                    custom_mask,
                ),
                add_widgets=(
                    iterations_labeled,
                    custom_mask
                ),
                tooltip=tooltip,
            )

        def make_structure_panel(
            title: str,
            enum_prefix: str,
            on_toggle_widgets: Sequence[widgets.Widget] | None = None,
            add_widgets: Sequence[widgets.Widget] | None = None,
            tooltip: str = "",
        ):
            def on_toggle(change: dict):
                disabled = not change["new"]
                for widget in (
                    structure_radius,
                    custom_structure,
                    *(on_toggle_widgets or []),
                ):
                    widget.disabled = disabled
                return

            toggle_disabled = not Default[enum_prefix]
            structure_radius = self._gui__add_widget(
                name=_WName[f"{enum_prefix}_STRUCT_RADIUS"].value,
                widget=widgets.FloatSlider(
                    value=Default[f"{enum_prefix}_STRUCT_RADIUS"],
                    min=0,
                    max=5,
                    step=0.01,
                    disabled=toggle_disabled,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format=".2f",
                    layout=widgets.Layout(width="100%"),
                )
            )
            custom_structure = self._gui__add_widget(
                name=_WName[f"{enum_prefix}_STRUCT_CUSTOM"].value,
                widget=widgets.Button(
                    description="Custom Structure",
                    icon="trash",
                    layout=widgets.Layout(display="none")
                )
            )
            custom_structure.add_class("resetbutton")
            structure_radius_labeled = scishow.widgets.labeled_widget(
                value="Structure Radius:",
                widget=structure_radius
            )
            return widgets.VBox(
                [
                    toggle_button(
                        name=_WName[enum_prefix],
                        text=title.capitalize(),
                        value=not toggle_disabled,
                        tooltip=tooltip,
                        observer=on_toggle,
                    ),
                    custom_structure,
                    structure_radius_labeled,
                    *(add_widgets or [])
                ],
                layout=widgets.Layout(
                    flex="1 1 0%",
                    padding="12px",
                    border="0.5px solid lightgray",
                    border_radius="10px",
                    min_width="0",
                    overflow="hidden",
                )
            )

        def toggle_button(
            name: _WName | str,
            text: str,
            value: Default | bool,
            tooltip: str,
            observer: Callable[[dict[str, Any]], None] | None = None,
            disabled: bool = False
        ):
            button = self._gui__add_widget(
                name=name.value if isinstance(name, Enum) else name,
                widget=scishow.widgets.toggle_button(
                    text,
                    value=value.value if isinstance(value, Enum) else value,
                    disabled=disabled,
                    tooltip=tooltip,
                )
            )
            if observer:
                button.observe(observer, names="value")
            button_centered = widgets.HBox(
                [button],
                layout=widgets.Layout(justify_content="center", width="100%")
            )
            return button_centered

        super().__init__()
        self._detector = detector
        self._ngl_name_receptor_volume_added = ngl_name_receptor_volume_added
        self._ngl_name_receptor_volume_removed = ngl_name_receptor_volume_removed
        self._ngl_name_mask = ngl_name_mask
        self._ngl_name_pocket = ngl_name_pocket
        self._detector.set_mask_morphology()
        self._detector.set_mask_ligsite()
        self._pockets = self._detector.extract_pockets()
        self._widg_status = widget_status()
        self._widg_ngl = widget_ngl()
        self._widg_log = widgets.Output()
        self._gui__set_main_widget(
            (
                display.HTML(self._CSS_STYLE),
                widgets.Tab(
                    children=[tab_morph(), tab_ligsite(), tab_extract()],
                    titles=["Volume", "LIGSITE", "Extraction"],
                    selected_index=0,
                ),
                self._widg_ngl,
                widgets.Accordion(
                    children=[self._widg_log],
                    titles=["Logs"],
                    selected_index=None,
                )
            ),
            receptor_volume=True,
            pockets=True,
        )
        self._custom_input = {
            _WName.MORPH_CLOSE_STRUCT_CUSTOM: None,
            _WName.MORPH_CLOSE_MASK: None,
            _WName.EXTRACT_OPEN_STRUCT_CUSTOM: None,
            _WName.EXTRACT_OPEN_MASK: None,
        }
        self._ngl_current_pocket_names = []
        self._ngl_current_pocket_box_names = []
        return

    def extract_pockets(
        self,
        open: bool = Default.EXTRACT_OPEN,
        opening_structure: np.ndarray | float = Default.EXTRACT_OPEN_STRUCT_RADIUS,
        opening_iterations: int = Default.EXTRACT_OPEN_ITER,
        opening_mask: np.ndarray | None = None,
    ):
        with self._widg_log:
            print("Extracting pockets.")
        with self._show_status():
            with self._gui__temporary_observation_toggle():
                self._gui__get_widget(_WName.EXTRACT_OPEN.value).value = open
                self._set_structuring_element(_WName.EXTRACT_LABEL.name, label_structure)
                if open:
                    self._set_structuring_element(_WName.EXTRACT_OPEN.name, opening_structure)
                    self._gui__get_widget(_WName.EXTRACT_OPEN_ITER.value).value = opening_iterations
                    # Set opening mask
                    self._custom_input[_WName.EXTRACT_OPEN_MASK] = opening_mask
                    self._gui__get_widget(_WName.EXTRACT_OPEN_MASK.value).layout.display = "none" if opening_mask is None else ""
                labels, n_labels = self._detector.extract_pockets(
                    opening=open,
                    opening_structure=opening_structure,
                    opening_iterations=opening_iterations,
                    opening_mask=opening_mask,
                    label_structure=label_structure,
                )
        self._gui__render()
        return

    def set_mask_morphology(
        self,
        close: bool = Default.MORPH_CLOSE,
        fill: bool = Default.MORPH_FILL,
        closing_structure: np.ndarray | float = Default.MORPH_CLOSE_STRUCT_RADIUS,
        closing_iterations: int = Default.MORPH_CLOSE_ITER,
        closing_mask: np.ndarray | None = None,
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
                fill_structure=fill_structure,
            )
        self._gui__render(receptor_volume=True)
        return

    def set_mask_ligsite(
        self,
        count_lower: int | None = Default.LIGSITE_COUNT_LOWER,
        count_upper: int | None = Default.LIGSITE_COUNT_UPPER,
        dist_lower: float | None = Default.LIGSITE_DIST_LOWER,
        dist_upper: float | None = Default.LIGSITE_DIST_UPPER,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = Default.LIGSITE_DIST_LOWER_MODE,
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = Default.LIGSITE_DIST_UPPER_MODE,
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
    def pockets(self) -> Pockets:
        """The extracted pockets from the receptor."""
        return self._pockets

    @property
    def mask(self) -> JAXArray:
        return self._detector.mask

    @property
    def mask_morphology(self) -> JAXArray:
        return self._detector.mask_morphology

    @property
    def mask_ligsite(self) -> JAXArray | None:
        return self._detector.mask_ligsite

    @property
    def mask_custom(self) -> JAXArray | None:
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
            self._gui__get_widget(_WName.MORPH_CLOSE.value).value = Default.MORPH_CLOSE
            self._gui__get_widget(_WName.MORPH_FILL.value).value = Default.MORPH_FILL
            self._gui__get_widget(_WName.MORPH_CLOSE_ITER.value).value = Default.MORPH_CLOSE_ITER
            self._gui__get_widget(_WName.MORPH_CLOSE_MASK.value).layout.display = "none"
            self._custom_input[_WName.MORPH_CLOSE_MASK] = None
            self._set_structuring_element(
                enum_prefix=_WName.MORPH_CLOSE.name,
                structure=Default.MORPH_CLOSE_STRUCT_RADIUS,
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

    def _ovc__morph_close_struct_radius(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological closing structure radius set to {value}.")
        with self._show_status():
            self._set_structuring_element(
                _WName.MORPH_CLOSE.name,
                structure=value,
            )
            self._morph__set_mask()
        return

    def _oc__morph_close_struct_custom(self, _: widgets.Button):
        """Delete the custom structuring element for morphological closing."""
        with self._widg_log:
            print("Deleting custom structuring element for morphological closing.")
        self._custom_input[_WName.MORPH_CLOSE_STRUCT_CUSTOM] = None
        self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_CUSTOM.value).layout.display = "none"
        self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_RADIUS.value).disabled = False
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

    def _ovc___ligsite_refresh(self, change: dict):
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

    def _oc___ligsite_reset(self, _: widgets.Button):
        with self._widg_log:
            print("LIGSITE mask reset to default state.")
        with self._show_status(), self._gui__temporary_observation_toggle():
            self._gui__get_widget(_WName.LIGSITE_COUNT.value).value = Default.LIGSITE_COUNT
            self._gui__get_widget(_WName.LIGSITE_DIST.value).value = Default.LIGSITE_DIST
            self._gui__get_widget(_WName.LIGSITE_COUNT_RANGE.value).value = (
                max(Default.LIGSITE_COUNT_LOWER, self.ligsite.psp_count_min),
                min(Default.LIGSITE_COUNT_UPPER, self.ligsite.psp_count_max)
            )
            self._gui__get_widget(_WName.LIGSITE_DIST_RANGE.value).value = (
                max(Default.LIGSITE_DIST_LOWER, self.ligsite.psp_distance_min) if Default.LIGSITE_DIST_LOWER is not None else self.ligsite.psp_distance_min,
                min(Default.LIGSITE_DIST_UPPER, self.ligsite.psp_distance_max) if Default.LIGSITE_DIST_UPPER is not None else self.ligsite.psp_distance_max,
            )
            for side in ("min", "max"):
                default_dist = Default.LIGSITE_DIST_LOWER if side == "min" else Default.LIGSITE_DIST_UPPER
                default_mode = Default.LIGSITE_DIST_LOWER_MODE if side == "min" else Default.LIGSITE_DIST_UPPER_MODE
                self._gui__get_widget(_WName[f"LIGSITE_DIST_{side.upper()}"].value).value = False if default_dist is None else default_mode
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

    def _ovc___extract_refresh(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"Extraction auto-refresh {'enabled' if enabled else 'disabled'}.")
        self._gui__toggle_widget_observation(
            observe=enabled,
            name_regex=f"^{_WPrefix.EXTRACT.value}.+",
        )
        if not enabled:
            return
        with self._show_status():
            self._extract__set_mask()
        return

    def _oc___extract_reset(self, _: widgets.Button):
        with self._widg_log:
            print("Extraction mask reset to default state.")
        with self._show_status(), self._gui__temporary_observation_toggle():
            self._gui__get_widget(_WName.EXTRACT_OPEN.value).value = Default.EXTRACT_OPEN
            self._gui__get_widget(_WName.EXTRACT_LABEL.value).value = Default.EXTRACT_LABEL
            self._gui__get_widget(_WName.EXTRACT_OPEN_ITER.value).value = Default.EXTRACT_OPEN_ITER
            self._gui__get_widget(_WName.EXTRACT_OPEN_MASK.value).layout.display = "none"
            self._custom_input[_WName.EXTRACT_OPEN_MASK] = None
            self._set_structuring_element(
                enum_prefix=_WName.EXTRACT_OPEN.name,
                structure=(
                    Default.EXTRACT_OPEN_STRUCT_CONNECT,
                    Default.EXTRACT_OPEN_STRUCT_ITER
                )
            )
            self._set_structuring_element(
                enum_prefix=_WName.EXTRACT_LABEL.name,
                structure=(
                    Default.EXTRACT_LABEL_STRUCT_CONNECT,
                    Default.EXTRACT_LABEL_STRUCT_ITER
                )
            )
            self._extract__set_mask()
        return

    def _ovc__extract_open(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"Morphological opening {'enabled' if enabled else 'disabled'}.")
        with self._show_status():
            self._extract__set_mask()
        return

    def _ovc__extract_open_iter(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological opening iterations set to {value}.")
        with self._show_status():
            self._extract__set_mask()
        return

    def _oc__extract_open_mask(self, _: widgets.Button):
        with self._widg_log:
            print("Deleting custom mask for morphological opening.")
        self._custom_input[_WName.EXTRACT_OPEN_MASK] = None
        self._gui__get_widget(_WName.EXTRACT_OPEN_MASK.value).layout.display = "none"
        with self._show_status():
            self._extract__set_mask()
        return

    def _ovc__extract_open_struct_radius(self, change: dict):
        value = change["new"]
        with self._widg_log:
            print(f"Morphological opening structure radius set to {value}.")
        with self._show_status():
            self._set_structuring_element(
                _WName.EXTRACT_OPEN.name,
                structure=value,
            )
            self._extract__set_mask()
        return

    def _oc__extract_open_struct_custom(self, _: widgets.Button):
        with self._widg_log:
            print("Deleting custom structuring element for morphological opening.")
        self._custom_input[_WName.EXTRACT_OPEN_STRUCT_CUSTOM] = None
        self._gui__get_widget(_WName.EXTRACT_OPEN_STRUCT_CUSTOM.value).layout.display = "none"
        self._gui__get_widget(_WName.EXTRACT_OPEN_STRUCT_RADIUS.value).disabled = False
        with self._show_status():
            self._extract__set_mask()
        return

    def _ovc__extract_label(self, change: dict):
        enabled = change["new"]
        with self._widg_log:
            print(f"Labeling {'enabled' if enabled else 'disabled'}.")
        with self._show_status():
            self._extract__set_mask()
        return

    def _morph__set_mask(self):
        """Set the morphology mask based on the current GUI settings."""
        with self._widg_log:
            print("Recalculating morphology mask with current settings.")
        custom_closing_structure = self._custom_input[_WName.MORPH_CLOSE_STRUCT_CUSTOM]
        closing_structure = (
            self._gui__get_widget(_WName.MORPH_CLOSE_STRUCT_RADIUS.value).value
            if custom_closing_structure is None else custom_closing_structure
        )
        self._detector.set_mask_morphology(
            close=self._gui__get_widget(_WName.MORPH_CLOSE.value).value,
            fill=self._gui__get_widget(_WName.MORPH_FILL.value).value,
            closing_structure=closing_structure,
            closing_iterations=self._gui__get_widget(_WName.MORPH_CLOSE_ITER.value).value,
            closing_mask=self._custom_input[_WName.MORPH_CLOSE_MASK],
        )
        self._extract__set_mask(receptor_volume=True)
        return

    def _ligsite__set_mask(self):
        """Set the LIGSITE mask based on the current GUI settings."""
        with self._widg_log:
            print("Recalculating LIGSITE mask with current settings.")
        count_enabled = self._gui__get_widget(_WName.LIGSITE_COUNT.value).value
        dist_enabled = self._gui__get_widget(_WName.LIGSITE_DIST.value).value
        count_range = self._gui__get_widget(_WName.LIGSITE_COUNT_RANGE.value)
        dist_range = self._gui__get_widget(_WName.LIGSITE_DIST_RANGE.value)
        dist_min = self._gui__get_widget(_WName.LIGSITE_DIST_MIN.value)
        dist_max = self._gui__get_widget(_WName.LIGSITE_DIST_MAX.value)
        self._detector.set_mask_ligsite(
            count_lower=count_range.lower if count_enabled else None,
            count_upper=count_range.upper if count_enabled else None,
            dist_lower=dist_range.lower if dist_enabled and dist_min.value else None,
            dist_upper=dist_range.upper if dist_enabled and dist_max.value else None,
            dist_lower_mode=dist_min.value,
            dist_upper_mode=dist_max.value,
        )
        self._extract__set_mask()
        return

    def _extract__set_mask(self, receptor_volume: bool = False):
        """Extract pockets based on the current GUI settings."""
        refresh = self._gui__get_widget(_WName.EXTRACT_REFRESH.value).value
        if not refresh:
            self._gui__render(receptor_volume=receptor_volume)
            return
        with self._widg_log:
            print("Recalculating extraction mask with current settings.")
        custom_opening_structure = self._custom_input[_WName.EXTRACT_OPEN_STRUCT_CUSTOM]
        self._pockets = self._detector.extract_pockets(
            opening=self._gui__get_widget(_WName.EXTRACT_OPEN.value).value,
            opening_structure=(
                self._gui__get_widget(_WName.EXTRACT_OPEN_STRUCT_RADIUS.value).value
                if custom_opening_structure is None else custom_opening_structure
            ),
            opening_iterations=self._gui__get_widget(_WName.EXTRACT_OPEN_ITER.value).value,
            opening_mask=self._custom_input[_WName.EXTRACT_OPEN_MASK],
        )
        self._gui__render(receptor_volume=receptor_volume, mask=False, pockets=True)
        return

    def _set_structuring_element(
        self,
        enum_prefix: str,
        structure: np.ndarray | float,
    ) -> None:
        enum_prefix = f"{enum_prefix.upper()}_STRUCT_"
        custom_structure_enum = _WName[f"{enum_prefix}CUSTOM"]
        w_custom = self._gui__get_widget(custom_structure_enum.value)
        w_radius = self._gui__get_widget(_WName[f"{enum_prefix}RADIUS"].value)
        if isinstance(structure, int | float):
            self._custom_input[custom_structure_enum] = None
            w_radius.value = structure
            w_custom.layout.display = "none"
        else:
            self._custom_input[custom_structure_enum] = structure
            w_radius.disabled = True
            w_custom.layout.display = ""
        return

    def _gui__render(
        self,
        receptor_volume: bool = False,
        mask: bool = True,
        pockets: bool = False
    ):
        """Update the NGLWidget with the current mask.

        This method is automatically called by the `GUI` parent class when needed.
        """
        ngl = self.nglwidget
        with self._show_status("Rendering..."):
            if receptor_volume:
                ngl.remove_component_by_name(self._ngl_name_receptor_volume_added)
                ngl.remove_component_by_name(self._ngl_name_receptor_volume_removed)
                ngl.add_volume(
                    self._detector.receptor_volume_added.astype(np.uint8),
                    name=self._ngl_name_receptor_volume_added,
                    basis=self.field.grid.unit_vectors,
                    origin=self.field.grid.lower_bounds,
                    representation_params=scishow.nglview.SurfaceRepresentationParameters(
                        lazy=True,
                        opacity=1,
                        contour=False,
                        visible=False,
                        color=(0,200,0),
                        isolevel=0.5,
                        isolevel_type="value",
                    )
                )
                ngl.add_volume(
                    self._detector.receptor_volume_removed.astype(np.uint8),
                    name=self._ngl_name_receptor_volume_removed,
                    basis=self.field.grid.unit_vectors,
                    origin=self.field.grid.lower_bounds,
                    representation_params=scishow.nglview.SurfaceRepresentationParameters(
                        lazy=True,
                        opacity=1,
                        contour=False,
                        visible=False,
                        color=(200,0,0),
                        isolevel=0.5,
                        isolevel_type="value",
                    )
                )
            if mask:
                ngl.remove_component_by_name(self._ngl_name_mask)
                ngl.add_volume(
                    self.mask.astype(np.uint8),
                    name=self._ngl_name_mask,
                    basis=self.field.grid.unit_vectors,
                    origin=self.field.grid.lower_bounds,
                    representation_params=scishow.nglview.SurfaceRepresentationParameters(
                        lazy=True,
                        opacity=1,
                        contour=False,
                        visible=False,
                        color=(0,0,200),
                        isolevel=0.5,
                        isolevel_type="value",
                    )
                )
            if pockets:
                for pocket_name in self._ngl_current_pocket_names + self._ngl_current_pocket_box_names:
                    ngl.remove_component_by_name(pocket_name)
                self._ngl_current_pocket_names.clear()
                self._ngl_current_pocket_box_names.clear()
                for _, pocket in self._pockets.pockets.iterrows():
                    base_name = f"{self._ngl_name_pocket} {pocket.label}"
                    pocket_name = f"{base_name} ({round(pocket.volume)})"
                    box_name = f"{base_name} Box"
                    self._ngl_current_pocket_names.append(pocket_name)
                    self._ngl_current_pocket_box_names.append(box_name)
                    pocket.pocket.display(
                        nglwidget=ngl,
                        name=pocket_name,
                        box_name=box_name,
                        contour=True,
                        visible=True,
                        color=(0,0,200),
                    )


                # for pocket_idx in range(1, self._pockets.num_features + 1):
                #     pocket_size = self._pockets.num_points[pocket_idx]
                #     if pocket_size < 300:
                #         continue
                #     pocket_name = f"{self._ngl_name_pocket} {pocket_idx} ({pocket_size})"
                #     pocket_slice = self._pockets.slices[pocket_idx - 1]
                #     pocket_slice_dialated = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, self._pockets.pocket_labels.shape[axis_idx])) for axis_idx, s in enumerate(pocket_slice))
                #     pocket_lower_bound_idx = tuple(s.start for s in pocket_slice_dialated)
                #     pocket_lower_bound = self.field.grid.coordinates[pocket_lower_bound_idx]
                #     ngl.add_volume(
                #         (self._pockets.pocket_labels[pocket_slice_dialated] == pocket_idx).astype(np.uint8),
                #         name=pocket_name,
                #         basis=self.field.grid.unit_vectors,
                #         origin=pocket_lower_bound,
                #         representation_params=scishow.nglview.SurfaceRepresentationParameters(
                #             lazy=True,
                #             opacity=1,
                #             contour=False,
                #             visible=True,
                #             color=(0,0,200),
                #             isolevel=0.5,
                #             isolevel_type="value",
                #         )
                #     )
                #     self._ngl_current_pocket_names.append(pocket_name)
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


class _WPrefix(Enum):
    MORPH = "morph_"
    LIGSITE = "ligsite_"
    EXTRACT = "extract_"


class _WName(Enum):
    """Widget names for the GUI."""

    # Morphological Transformations
    MORPH_REFRESH = f"_{_WPrefix.MORPH.value}refresh"
    MORPH_RESET = f"_{_WPrefix.MORPH.value}reset"

    # Morphological Closing
    MORPH_CLOSE = f"{_WPrefix.MORPH.value}close"
    MORPH_CLOSE_ITER = f"{_WPrefix.MORPH.value}close_iter"
    MORPH_CLOSE_MASK = f"{_WPrefix.MORPH.value}close_mask"
    # Morphological Closing Structure
    MORPH_CLOSE_STRUCT_RADIUS = f"{_WPrefix.MORPH.value}close_struct_radius"
    MORPH_CLOSE_STRUCT_CUSTOM = f"{_WPrefix.MORPH.value}close_struct_custom"

    # Morphological Filling
    MORPH_FILL = f"{_WPrefix.MORPH.value}fill"

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


    # Extraction
    EXTRACT_REFRESH = f"_{_WPrefix.EXTRACT.value}refresh"
    EXTRACT_RESET = f"_{_WPrefix.EXTRACT.value}reset"

    # Morphological Opening
    EXTRACT_OPEN = f"{_WPrefix.EXTRACT.value}open"
    EXTRACT_OPEN_ITER = f"{_WPrefix.EXTRACT.value}open_iter"
    EXTRACT_OPEN_MASK = f"{_WPrefix.EXTRACT.value}open_mask"
    # Morphological Opening Structure
    EXTRACT_OPEN_STRUCT_RADIUS = f"{_WPrefix.EXTRACT.value}open_struct_radius"
    EXTRACT_OPEN_STRUCT_CUSTOM = f"{_WPrefix.EXTRACT.value}open_struct_custom"

    # Labeling
    EXTRACT_LABEL = f"{_WPrefix.EXTRACT.value}label"
