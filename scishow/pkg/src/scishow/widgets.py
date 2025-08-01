"""Functionalities for creating a GUI with [ipywidgets](https://ipywidgets.readthedocs.io/)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from IPython.display import display
import ipywidgets
from ipywidgets import IntRangeSlider, FloatRangeSlider, IntSlider, FloatSlider, HBox, VBox, Label, HTML, Box, Layout, Widget, Button
import re
import traitlets
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Sequence, Generator

class GUI:
    """Base class for creating a GUI with [ipywidgets](https://ipywidgets.readthedocs.io/).

    Parameters
    ----------
    auto_toggle_availability
        Whether to automatically toggle the availability of widgets
        during the handling of widget events.
        If set to `True`, the GUI will automatically disable all widgets
        during the handling of a widget event,
        and re-enable the previously enabled ones
        after the event handler has finished.
    auto_toggle_observation
        Whether to automatically toggle observation of widget events
        during the handling of widget events.
        If set to `True`, the GUI will automatically disable observation
        of all further widget events during the handling of a widget event,
        and re-enable it after the event handler has finished.
    observer_method_name_template
        Name template for observer methods that handle widget events in the subclass.
        This is used to dynamically find the corresponding observer method
        based on the widget's name and event type/name.
        It can be a [format string](https://docs.python.org/3/library/string.html#format-string-syntax)
        using the following variables:
        - `event_name`: Name of the traitlet event,
           as returned by the [`change` dictionary](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html#traitlet-events).
           If the event has no name (e.g., for `Button` widgets),
           this defaults to a single space character,
           which is subsequently removed after formatting.
        - `event_type`: Type of the traitlet event,
           as returned by the [`change` dictionary](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html#traitlet-events).
           For `Button` widgets, this is set to `"click"`.
        - `widget_name`: Name of the widget, as provided when adding it to the GUI
           using the `_gui__add_widget` method.

    Usage
    -----
    1. Create a subclass of `GUI`.
    2. In the subclass's `__init__` method:
        1. Initialize this class with a specific observer method prefix.
        2. Add interactive widgets with a unique name
           using the `_gui__add_widget` method.
        3. Call the `_gui__set_main_widget` method to
           set the main GUI widget that will be displayed.
    3. If needed, implement a `_gui__render` method to apply the current state of the GUI.
       This method is first called when the user calls the `display` method,
       and subsequently called whenever an interactive widget's value changes (see 4).
       The method can accept optional keyword arguments that will be passed
       from the observer methods.
    4. For each interactive widget, define observer method(s) to handle events as needed.
       The method names must follow the `observer_method_name_template` parameter.
       They must accept a single argument, which is a `ipywidgets.Button` instance
       for when the widget is a button, or a dictionary with a 'owner' key
       containing the widget instance for other widgets.
       Each observer method must either return `None`,
       or a dictionary of keyword arguments that will be passed to the `_gui__render` method.
       If a dictionary is returned (regardless of whether it is empty or not),
       the `_gui__render` method will be subsequently called
       with those keyword arguments to update the GUI.
    """
    def __init__(
        self,
        auto_toggle_availability: bool = False,
        auto_toggle_observation: bool = False,
        observer_method_name_template="_o{event_name[0]}{event_type[0]}__{widget_name}"
    ):
        self._gui__auto_toggle_observation = auto_toggle_observation
        self._gui__auto_toggle_availability = auto_toggle_availability
        self._gui__observer_method_name_template = observer_method_name_template
        self._gui__widget_name_to_widget: dict[str, Widget] = {}
        self._gui__widget_id_to_name: dict[int, str] = {}
        self._gui__main_widget = None
        self._gui__display_kwargs: dict[str, Any] = {}
        return

    def display(self) -> None:
        """Display the GUI in the current Jupyter notebook cell."""
        if self._gui__main_widget is None:
            raise RuntimeError("GUI has not been initialized.")
        self._gui__render(**self._gui__display_kwargs)
        if isinstance(self._gui__main_widget, Widget):
            display(self._gui__main_widget)
        else:
            display(*self._gui__main_widget)
        return

    def _gui__set_main_widget(
        self,
        widget: Widget | Sequence[Widget],
        **kwargs: Any,
    ) -> None:
        """Set the main GUI widget that will be displayed.

        This method should be called in the subclass's `__init__` method.
        """
        self._gui__main_widget = widget
        self._gui__display_kwargs = kwargs
        return

    def _gui__add_widget(
        self,
        name: str,
        widget: Widget,
        observe: bool = True,
        observe_name: str | traitlets.Sentinel | Sequence[str | traitlets.Sentinel] = "value",
        observe_type: str | traitlets.Sentinel = "change",
    ) -> Widget:
        """Add an interactive widget to the GUI.

        This method should be called in the subclass,
        normally in its `__init__` method.

        Parameters
        ----------
        name
            A unique name for the widget.
        widget
            An instance of an `ipywidgets.Widget` subclass
            (e.g., `Button`, `Dropdown`, etc.).
        observe
            Whether to observe widget events.
            If set to `False`, `observe_name` and `observe_type` are ignored.
        observe_name
            Name(s) of widget trait(s) to observe.
            Available options depend on the widget type.
            For example, a `Dropdown` widget has
            `comm`, `index`, `label`, `options`, and `value` traits,
            whereas a `Button` widget has only one `on_click` trait.
            Note that for `Button` widgets,
            the `observe_name` parameter is ignored,
            and the `on_click` event is always observed.
        observe_type
            Type of trait notification to observe.
            Available options depend on `observe_name`.
            Most traits only support the `change` type.

        References
        ----------
        - [Jupyter Widgets Documentation: Widget Events](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html)

        Returns
        -------
        The same input `widget` instance is returned for convenience.
        """
        if not isinstance(widget, Widget):
            raise TypeError(f"Expected a Widget instance, got {type(widget)}")
        widget_id = id(widget)
        self._gui__widget_name_to_widget[name] = widget
        self._gui__widget_id_to_name[widget_id] = name
        if observe:
            self._gui__toggle_widget_observation(
                widget=widget,
                observe=True,
                observe_name=observe_name,
                observe_type=observe_type,
            )
        return widget

    def _gui__get_widget(self, name: str) -> Widget:
        """Get a widget by its name.

        The widget must have been added using `_add_widget`.

        Parameters
        ----------
        name
            Unique name of the widget.
        """
        if name not in self._gui__widget_name_to_widget:
            raise KeyError(f"Widget '{name}' not found")
        return self._gui__widget_name_to_widget[name]

    def _gui__reset_slider_minmax(
        self,
        slider: IntSlider | FloatSlider | IntRangeSlider | FloatRangeSlider | str,
        minimum: int | float,
        maximum: int | float,
        value: int | float | tuple[int | None, int | None] | tuple[float | None, float | None] | None = None,
        disable_observe: bool = False,
    ):
        """Reset a numeric slider's minimum/maximum values.

        Parameters
        ----------
        slider
            Numeric slider widget to reset.
            This can be an instance of `IntSlider`, `FloatSlider`, `IntRangeSlider`, or `FloatRangeSlider`,
            or a string representing the slider widget's name.
        minimum
            Minimum value to set for the slider.
        maximum
            Maximum value to set for the slider.
        value
            Optional initial value to set for the slider.
            If set to `None`, the slider value will not be changed.
        disable_observe
            Whether to temporarily disable observing the slider widget's events
        """
        if isinstance(slider, str):
            slider = self._gui__get_widget(slider)
        if disable_observe:
            self._gui__toggle_widget_observation(False, widget=slider)
        if slider.min >= maximum:
            slider.min = minimum
            slider.max = maximum
        else:
            slider.max = maximum
            slider.min = minimum
        if value is not None:
            if isinstance(slider, IntSlider | FloatSlider):
                slider.value = value
            else:
                lower, upper = value
                if lower is not None:
                    slider.lower = lower
                if upper is not None:
                    slider.upper = upper
        if disable_observe:
            self._gui__toggle_widget_observation(True, widget=slider)
        return slider

    def _gui__toggle_widget_observation(
        self,
        observe: bool | None,
        *,
        observe_name: str | traitlets.Sentinel | Sequence[str | traitlets.Sentinel] = "value",
        observe_type: str | traitlets.Sentinel = "change",
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | re.Pattern | None = None,
    ) -> None:
        """Enable, disable, or toggle observing widget events.

        If none of the parameters `widget`, `name`, or `name_regex` is provided,
        this method will apply to all widgets registered in the GUI,
        otherwise it will only apply to the widgets specified by any of these parameters.

        This method is for example useful
        when making multiple changes to the GUI state at once,
        to avoid unnecessarily triggering other widget events.
        Note that this method must not be used to register observers for a widget;
        for that, use the `_gui__add_widget` method instead.

        Parameters
        ----------
        observe
            Whether to observe (True), unobserve (False),
            or toggle observation (None) of widget events.
            If toggling, this method will add `self._gui__widget_observer`
            as an observer if it is not already registered,
            or remove it if it is already registered.
        observe_name
            Name(s) of widget trait(s) to consider.
            Available options depend on the widget type.
            For example, a `Dropdown` widget has
            `comm`, `index`, `label`, `options`, and `value` traits,
            whereas a `Button` widget has only one `on_click` trait.
            Note that for `Button` widgets,
            the `observe_name` parameter is ignored,
            and the `on_click` event is always used.
        observe_type
            Type of trait notification to consider.
            Available options depend on `observe_name`.
            Most traits only support the `change` type.
        widget
            Optional widget or sequence of widgets to consider.
        name
            Optional name or sequence of names of widgets to consider.
        name_regex
            Optional regular expression to filter which widgets to consider.

        References
        ----------
        - [Jupyter Widgets Documentation: Widget Events](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html)
        """
        for w in self._gui__iterate_widgets(
            widget=widget,
            name=name,
            name_regex=name_regex
        ):
            if isinstance(w, Button):
                is_observed = self._gui__widget_observer in w._click_handlers.callbacks
                do_observe = not is_observed if observe is None else observe
                if is_observed != do_observe:
                    w.on_click(
                        self._gui__widget_observer,
                        remove=not do_observe
                    )
            else:
                is_observed = self._gui__widget_observer in w._trait_notifiers.get(observe_name, {}).get(observe_type, [])
                do_observe = not is_observed if observe is None else observe
                if is_observed != do_observe:
                    func = w.observe if do_observe else w.unobserve
                    func(
                        self._gui__widget_observer,
                        names=observe_name,
                        type=observe_type,
                    )
        return

    def _gui__toggle_widget_availability(
        self,
        available: bool | None,
        *,
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | re.Pattern | None = None,
    ) -> None:
        """Enable/disable widgets, or toggle their availability.

        If none of the parameters `widget`, `name`, or `name_regex` is provided,
        this method will apply to all widgets registered in the GUI,
        otherwise it will only apply to the widgets specified by any of these parameters.

        Parameters
        ----------
        available
            Whether to enable (True), disable (False),
            or toggle availability (None) of widgets.
            If toggling, this method will disable the widget
            if it is not already disabled,
            or enable it if it is already disabled.
        widget
            Optional widget or sequence of widgets to consider.
        name
            Optional name or sequence of names of widgets to consider.
        name_regex
            Optional regular expression to filter which widgets to consider.
        """
        for w in self._gui__iterate_widgets(
            widget=widget,
            name=name,
            name_regex=name_regex
        ):
            w.disabled = not w.disabled if available is None else not available
        return

    @contextmanager
    def _gui__temporary_toggle(
        self,
        *,
        observe: bool | None = False,
        available: bool | None = False,
        observe_name: str | traitlets.Sentinel | Sequence[str | traitlets.Sentinel] = "value",
        observe_type: str | traitlets.Sentinel = "change",
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | None = None,
        restore: bool = True,
    ) -> None:
        """Temporarily enable, disable, or toggle observing widget events and/or availability.

        This is a context manager wrapping the two other context managers
        `_gui__temporary_observation_toggle` and `_gui__temporary_availability_toggle`.
        All parameters are the same as for those methods combined.
        """
        with self._gui__temporary_observation_toggle(
            observe=observe,
            observe_name=observe_name,
            observe_type=observe_type,
            widget=widget,
            name=name,
            name_regex=name_regex
        ), self._gui__temporary_availability_toggle(
            available=available,
            widget=widget,
            name=name,
            name_regex=name_regex,
            restore=restore
        ):
            yield
        return

    @contextmanager
    def _gui__temporary_observation_toggle(
        self,
        observe: bool | None = False,
        *,
        observe_name: str | traitlets.Sentinel | Sequence[str | traitlets.Sentinel] = "value",
        observe_type: str | traitlets.Sentinel = "change",
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | None = None
    ) -> None:
        """Temporarily enable, disable, or toggle observing widget events.

        This is a context manager for the `_gui__toggle_widget_observation` method.
        All parameters are the same as for that method.
        """
        # Resolve the widgets once here for efficiency
        widgets = list(
            self._gui__iterate_widgets(
                widget=widget,
                name=name,
                name_regex=name_regex,
            )
        )
        self._gui__toggle_widget_observation(
            observe=observe,
            observe_name=observe_name,
            observe_type=observe_type,
            widget=widgets,
        )
        try:
            yield
        finally:
            self._gui__toggle_widget_observation(
                observe=None if observe is None else not observe,
                observe_name=observe_name,
                observe_type=observe_type,
                widget=widgets,
            )
        return

    @contextmanager
    def _gui__temporary_availability_toggle(
        self,
        available: bool | None = False,
        *,
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | None = None,
        restore: bool = True
    ) -> None:
        """Temporarily enable/disable widgets, or toggle their availability.

        This is a context manager for the `_gui__toggle_widget_availability` method.
        All parameters are the same as for that method, except for `restore`,
        which is only available here.

        Parameters
        ----------
        restore
            Whether to restore the original availability state of the widgets
            after the context manager exits.
            This is only relevant if `available` is set to `True` or `False`.
        """
        widgets_and_states = []
        for w in self._gui__iterate_widgets(
            widget=widget,
            name=name,
            name_regex=name_regex
        ):
            widgets_and_states.append((w, w.disabled))
            w.disabled = not w.disabled if available is None else not available
        try:
            yield
        finally:
            for w, original_state in widgets_and_states:
                w.disabled = original_state if restore else (
                    not w.disabled if available is None else available
                )
        return

    def _gui__iterate_widgets(
        self,
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | re.Pattern | None = None,
    ) -> Generator[Widget, None, None]:
        """Iterate over widgets in the GUI.

        This method yields each widget that matches any of the provided parameters.
        If no parameters are provided, it yields all widgets in the GUI.
        Note that the widgets must have been added using `_gui__add_widget`.

        Parameters
        ----------
        widget
            Optional widget or sequence of widgets to iterate over.
        name
            Optional name or sequence of names of widgets to iterate over.
        name_regex
            Optional regular expression to filter which widgets to iterate over.
        """
        widget_provided = widget is not None
        name_provided = name is not None
        name_regex_provided = name_regex is not None
        if widget_provided:
            if isinstance(widget, Widget):
                widget = [widget]
            for w in widget:
                if not isinstance(w, Widget):
                    raise TypeError(f"Expected a Widget instance, got {type(w)}")
                yield w
        if name_provided:
            if isinstance(name, str):
                name = [name]
            for w_name in name:
                w = self._gui__widget_name_to_widget.get(w_name)
                if not w:
                    raise KeyError(f"Widget '{w_name}' not found")
                yield w
        if name_regex_provided:
            if isinstance(name_regex, str):
                name_regex = re.compile(name_regex)
            for w_name, w in self._gui__widget_name_to_widget.items():
                if name_regex.match(w_name):
                    yield w
        if not (widget_provided or name_provided or name_regex_provided):
            # If no specific widgets or names are provided, apply to all widgets
            yield from self._gui__widget_name_to_widget.values()
        return

    def _gui__widget_observer(self, change: dict[str, Any] | Button) -> None:
        """Handle widget events for registered interactive widgets.

        This method is added as an observer to all widgets added with `_gui__add_widget`.
        It calls the corresponding observer method (if it exists) in the subclass,
        based on the widget's name and event name/type.
        """
        is_button = isinstance(change, Button)
        widget = change if is_button else change['owner']
        widget_id = id(widget)
        widget_name = self._gui__widget_id_to_name.get(widget_id)
        if not widget_name:
            # Widget is not registered using the `_gui__add_widget` method.
            return
        observer_method_name = self._gui__observer_method_name_template.format(
            event_name=" " if is_button else change.get("name", " "),
            event_type="click" if is_button else change["type"],
            widget_name=widget_name
        ).replace(" ", "")
        observer_method = getattr(self, observer_method_name, None)
        if observer_method:
            observation_context_manager = self._gui__temporary_observation_toggle(
                observe=False
            ) if self._gui__auto_toggle_observation else nullcontext()
            availability_context_manager = self._gui__temporary_availability_toggle(
                available=False
            ) if self._gui__auto_toggle_availability else nullcontext()
            with observation_context_manager, availability_context_manager:
                render_kwargs = observer_method(change)
            if render_kwargs is not None:
                self._gui__render(**render_kwargs)
        return

    def _gui__render(self, **kwargs) -> None:
        """Re-render (parts of) the GUI based on the current state of the widgets.

        If needed, this method should be implemented in the subclass.
        """
        return


def toggle_button(
    description: str,
    *,
    css_class_on: str = "togglebutton-on",
    css_class_off: str = "togglebutton-off",
    css_class: str | Sequence[str] | None = None,
    value: bool = False,
    **kwargs
) -> ipywidgets.ToggleButton:
    """Create a toggle button with custom CSS classes for on/off states.

    This function creates a toggle button with the given arguments and
    registers an observer to update its CSS class based on its value.
    The CSS classes must be defined separately and included in the notebook.
    For example:

    ```python
    from IPython.display import display, HTML
    display(
        HTML(
            \"""<style>
            .togglebutton-on {
                background-color: green;
                color: white;
            }
            .togglebutton-off {
                background-color: red;
                color: white;
            }
            </style>
            \"""
        )
    )
    ```

    Parameters
    ----------
    description
        Text to display on the button.
    css_class_on
        CSS class name to apply when the button is toggled on.
    css_class_off
        CSS class name to apply when the button is toggled off.
    value
        Initial value of the toggle button.
    **kwargs
        Additional keyword arguments to pass
        to the `ipywidgets.ToggleButton` constructor.
    """
    def update_css(change: dict[str, Any] = None):
        button.remove_class(toggle_css_class[not button.value])
        button.add_class(toggle_css_class[button.value])
        return

    toggle_css_class = {
        True: css_class_on,
        False: css_class_off
    }
    button = ipywidgets.ToggleButton(value=value, description=description, **kwargs)
    if css_class:
        if isinstance(css_class, str):
            css_class = [css_class]
        for name in css_class:
            button.add_class(name)
    button.observe(update_css, names="value", type="change")
    update_css()
    return button


def label(
    value: str,
    description: str | None = None,
    layout: dict[str, Any] | None = None,
):
    kwargs = locals()
    layout = {
        "min_width": "fit-content",
        "margin": "0 5px 0 0",
        "align_self": "center"
    } | (layout or {})
    return Label(
        value=value,
        layout=Layout(**layout),
        **{k: v for k, v in kwargs.items() if k not in ['value', 'layout'] and v is not None}
    )


def labeled_widget(
    value: str,
    widget: Widget,
    description: str | None = None,
    layout: dict[str, Any] | None = None,
) -> HBox:
    """Create a labeled widget with a label on the left.

    Parameters
    ----------
    value
        Text to display in the label.
    widget
        An instance of an `ipywidgets.Widget` subclass to be labeled.
    description
        Optional description text for the label.
    layout
        Optional layout dictionary to customize the label's appearance.
    """
    kwargs = locals()
    kwargs.pop('widget')
    widget_label = label(**kwargs)
    widget.layout.flex = "1 1 auto"
    return HBox(
        [widget_label, widget],
        layout=Layout(
            align_items='center',
            min_width='fit-content',
            width='100%',
        )
    )
