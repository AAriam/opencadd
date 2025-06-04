from __future__ import annotations

import asyncio
import operator
from time import time
from typing import TYPE_CHECKING

import ipywidgets as widgets
import numpy as np
from IPython.display import display

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    from caddpy.pocket.ligsite import LigSiteDetector
    from scishow.nglview import NGLWidget


class LigSiteWidget:
    def __init__(self, ligsite: LigSiteDetector, ngl_widget: NGLWidget):
        self._ligsite = ligsite
        self._nglwidget = ngl_widget

        self._psp_count = ligsite.psp_count
        self._psp_dist = ligsite.psp_distance

        self._mask_psp_count_lower = self._initialize_mask_array()
        self._mask_psp_count_upper = self._initialize_mask_array()

        self._mask_psp_dist_lower = self._initialize_mask_array()
        self._mask_psp_dist_upper = self._initialize_mask_array()

        self._mask = self._ligsite.volume_negative


        self._widget_psp_count = self._create_numeric_range_slider(
            dtype="int",
            minimum=self._psp_count.min(),
            maximum=self._psp_count.max(),
            observer=self._on_value_change_psp_count
        )
        self._widget_psp_dist = self._create_numeric_range_slider(
            dtype="float",
            minimum=np.nanmin(self._psp_dist),
            maximum=np.nanmax(self._psp_dist),
            observer=self._on_value_change_psp_dist
        )

        psp_control_panel = self._create_control_panel(
            header_name="PSP",
            controller_labels=("PSP Distance", "PSP Count"),
            controllers=[
                self._widget_psp_dist,
                self._widget_psp_count,
            ],

        )

        refinement_panel = widgets.HBox([psp_control_panel])
        main_panel = widgets.Accordion(
            children=[refinement_panel],
        )
        main_panel.set_title(0, "PSP")
        main_panel.selected_index = 0

        self._debug = widgets.Output()
        self._update_grid()
        display(self._debug, main_panel, self._nglwidget.display(gui=True))
        return

    @property
    def mask(self) -> np.ndarray:
        masks = [
            self._mask_psp_count_lower,
            self._mask_psp_count_upper,
            self._mask_psp_dist_lower,
            self._mask_psp_dist_upper,
            self._ligsite.volume_negative,
        ]
        np.logical_and.reduce(masks, out=self._mask)
        return self._mask


    def _update_grid(self):
        with self._debug:
            print("Updating grid...")
        name = "grid"
        self._nglwidget.remove_component_by_name(name)
        self._nglwidget.add_spheres(
            coords=self._ligsite.field.grid.coordinates[self.mask],
            name=name,
        )
        return

    def _initialize_mask_array(self):
        return np.ones(shape=self._ligsite.volume.shape, dtype=np.bool_)

    def _on_value_change_psp_dist(self, change: dict):
        with self._debug:
            print("PSP distance changed:", change)
        old_lower, old_upper = change["old"]
        new_lower, new_upper = change["new"]
        if new_lower != old_lower:
            self._mask_psp_dist_lower = np.any(self._psp_dist >= new_lower, axis=-1)
        if new_upper != old_upper:
            self._mask_psp_dist_upper = np.any(self._psp_dist <= new_upper, axis=-1)
        self._update_grid()
        return

    def _on_value_change_psp_count(self, change: dict):
        with self._debug:
            print("PSP count changed:", change)
        old_lower, old_upper = change["old"]
        new_lower, new_upper = change["new"]
        if new_lower != old_lower:
            self._mask_psp_count_lower = self._psp_count >= new_lower
        if new_upper != old_upper:
            self._mask_psp_count_upper = self._psp_count <= new_upper
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




class Widget:
    def __init__(self):

        self._widget_psp_count = self._create_numeric_range_slider(
            dtype="int",
            minimum=0,
            maximum=10,
            observer=self._on_value_change_psp_count
        )
        self._widget_psp_dist = self._create_numeric_range_slider(
            dtype="float",
            minimum=1,
            maximum=5,
            observer=self._on_value_change_psp_dist
        )

        psp_control_panel = self._create_control_panel(
            header_name="PSP",
            controller_labels=("PSP Distance", "PSP Count"),
            controllers=[
                self._widget_psp_dist,
                self._widget_psp_count,
            ],

        )

        refinement_panel = widgets.HBox([psp_control_panel])
        main_panel = widgets.Accordion(
            children=[refinement_panel],
        )

        main_panel.set_title(0, "PSP")
        main_panel.selected_index = 0

        self._debug = widgets.Output()
        self._update()
        display(self._debug, main_panel)
        return

    def _update(self):
        with self._debug:
            print("Updating...")
        return

    def _on_value_change_psp_dist(self, change: dict):
        with self._debug:
            print("PSP distance changed:", change)
        self._update()
        return

    def _on_value_change_psp_count(self, change: dict):
        with self._debug:
            print("PSP count changed:", change)
        self._update()
        return

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
