"""Functionalities for creating a GUI with [ipywidgets](https://ipywidgets.readthedocs.io/)."""

from contextlib import contextmanager
from typing import Any, Sequence
import re
import traitlets

from IPython.display import display
from ipywidgets import Dropdown, IntRangeSlider, HBox, VBox, Label, HTML, Box, Layout, Widget, Button


class GUI:
    """Base class for creating a GUI with [ipywidgets](https://ipywidgets.readthedocs.io/).

    Parameters
    ----------
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
        observer_method_name_template="_o{event_name[0]}{event_type[0]}__{widget_name}"
    ):
        self._gui__observer_method_name_template = observer_method_name_template
        self._gui__widget_name_to_widget: dict[str, Widget] = {}
        self._gui__widget_id_to_name: dict[int, str] = {}
        self._gui__main_widget = None
        return

    def display(self) -> None:
        """Display the GUI in the current Jupyter notebook cell."""
        if self._gui__main_widget is None:
            raise RuntimeError("GUI has not been initialized.")
        self._gui__render()
        if isinstance(self._gui__main_widget, Widget):
            display(self._gui__main_widget)
        else:
            display(*self._gui__main_widget)
        return

    def _gui__set_main_widget(self, widget: Widget | Sequence[Widget]) -> None:
        """Set the main GUI widget that will be displayed.

        This method should be called in the subclass's `__init__` method.
        """
        self._gui__main_widget = widget
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
            self._gui__toggle_widget_observer(
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

    def _gui__toggle_widget_observer(
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
        def toggle(w: Widget) -> None:
            if not isinstance(w, Widget):
                raise TypeError(f"Expected a Widget instance, got {type(w)}")
            if isinstance(w, Button):
                do_observe = (
                    self._gui__widget_observer not in w._click_handlers.callbacks
                    if observe is None else observe
                )
                w.on_click(
                    self._gui__widget_observer,
                    remove=not do_observe
                )
            else:
                do_observe = self._gui__widget_observer not in w._trait_notifiers.get(
                    observe_name, {}
                ).get(observe_type, []) if observe is None else observe
                func = w.observe if do_observe else w.unobserve
                func(
                    self._gui__widget_observer,
                    names=observe_name,
                    type=observe_type,
                )
            return

        widget_provided = widget is not None
        name_provided = name is not None
        name_regex_provided = name_regex is not None
        if widget_provided:
            if isinstance(widget, Widget):
                widget = [widget]
            for w in widget:
                toggle(w)
        if name_provided:
            if isinstance(name, str):
                name = [name]
            for w_name in name:
                w = self._gui__widget_name_to_widget.get(w_name)
                if not w:
                    raise KeyError(f"Widget '{w_name}' not found")
                toggle(w)
        if name_regex_provided:
            if isinstance(name_regex, str):
                name_regex = re.compile(name_regex)
            for w_name, w in self._gui__widget_name_to_widget.items():
                if name_regex.match(w_name):
                    toggle(w)
        if not (widget_provided or name_provided or name_regex_provided):
            # If no specific widgets or names are provided, apply to all widgets
            for w in self._gui__widget_name_to_widget.values():
                toggle(w)
        return

    @contextmanager
    def _gui__temporary_observation_toggle(
        self,
        observe: bool | None = False,
        observe_name: str | traitlets.Sentinel | Sequence[str | traitlets.Sentinel] = "value",
        observe_type: str | traitlets.Sentinel = "change",
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | None = None
    ) -> None:
        """Context manager to temporarily enable or disable observing value changes.

        This is useful when making multiple changes to the GUI state at once,
        to avoid triggering the observers unnecessarily.

        Parameters
        ----------
        observe
            Whether to observe value changes of interactive widgets.
        widget
            Optional widget or sequence of widgets to observe/unobserve.
            If provided, only these widgets will be affected,
            otherwise all widgets will be affected.
        name
            Optional name or sequence of names of widgets to observe/unobserve.
            If provided, only widgets with these names will be affected,
            otherwise all widgets will be affected.
        name_regex
            Optional regular expression to filter which widgets to observe/unobserve.
            If provided, only widgets whose names match the regex will be affected,
            otherwise all widgets will be affected.
        """
        self._gui__toggle_widget_observer(
            observe=observe,
            observe_name=observe_name,
            observe_type=observe_type,
            widget=widget,
            name=name,
            name_regex=name_regex
        )
        try:
            yield
        finally:
            self._gui__toggle_widget_observer(
                observe=not observe if observe is not None else None,
                observe_name=observe_name,
                observe_type=observe_type,
                widget=widget,
                name=name,
                name_regex=name_regex
            )
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
            event_name="click" if is_button else change["type"],
            event_type=" " if is_button else change.get("name", " "),
            widget_name=widget_name
        ).replace(" ", "")
        print(observer_method_name)
        observer_method = getattr(self, observer_method_name, None)
        if observer_method:
            render_kwargs = observer_method(change)
            if render_kwargs is not None:
                self._gui__render(**render_kwargs)
        return

    def _gui__render(self, **kwargs) -> None:
        """Re-render (parts of) the GUI based on the current state of the widgets.

        If needed, this method should be implemented in the subclass.
        """
        return


def label(
    value: str,
    description: str | None = None,
    layout: dict[str, Any] | None = None,
):
    kwargs = locals()
    layout = {
        "min_width": "fit-content",
        "margin": "0 5px 0 0"
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
    return HBox(
        [widget_label, widget],
        layout=Layout(
            align_items='center',
            min_width='fit-content',
            width='100%',
        )
    )
