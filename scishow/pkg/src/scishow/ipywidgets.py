"""Functionalities for creating a GUI with [ipywidgets](https://ipywidgets.readthedocs.io/)."""

from contextlib import contextmanager
from typing import Any, Sequence
import re

from IPython.display import display
from ipywidgets import Dropdown, IntRangeSlider, HBox, VBox, Label, HTML, Box, Layout, Widget, Button


class GUI:
    """Base class for creating a GUI with [ipywidgets](https://ipywidgets.readthedocs.io/).

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
    4. For each interactive widget, define an observer method that handles value changes.
       The method must be named `{observer_method_prefix}{widget_name}`,
       where `observer_method_prefix` is the prefix passed to the constructor and
       `widget_name` is the name used when adding the widget.
       It must accept a single argument, which is a `ipywidgets.Button` instance
       for when the widget is a button, or a dictionary with a 'owner' key
       containing the widget instance for other widgets.
       Each observer method must either return `None`,
       or a dictionary of keyword arguments that will be passed to the `_gui__render` method.
       If a dictionary is returned (regardless of whether it is empty or not),
       the `_gui__render` method will be subsequently called
       with those keyword arguments to update the GUI.
    """
    def __init__(self, observer_method_prefix='_ovc_'):
        self._gui__observer_method_prefix = observer_method_prefix
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

    def _gui__add_widget(self, name: str, widget: Widget, observe: bool = True) -> Widget:
        """Add an interactive widget to the GUI.

        This method should be called in the subclass's `__init__` method.

        Parameters
        ----------
        name
            A unique name for the widget.
        widget
            An instance of an `ipywidgets.Widget` subclass
            (e.g., `Button`, `Dropdown`, etc.).
        observe
            Whether to observe value changes of the widget.

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
            self._gui__set_widget_value_observer(widget, observe=True)
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

    def _gui__set_widget_value_observer(self, widget: Widget, observe: bool = True) -> None:
        """Set whether to observe value changes of a specific widget.

        Parameters
        ----------
        widget
            The widget instance to set the observer for.
        observe
            Whether to observe value changes of the widget.
        """
        if not isinstance(widget, Widget):
            raise TypeError(f"Expected a Widget instance, got {type(widget)}")
        if isinstance(widget, Button):
            widget.on_click(
                self._gui__on_widget_value_change,
                remove=not observe
            )
        elif observe:
            widget.observe(
                self._gui__on_widget_value_change,
                names='value',
            )
        else:
            widget.unobserve(
                self._gui__on_widget_value_change,
                names='value',
            )
        return

    def _gui__observe_value_changes(
        self,
        observe: bool,
        widget: Widget | Sequence[Widget] | None = None,
        name: str | Sequence[str] | None = None,
        name_regex: str | re.Pattern | None = None,
    ) -> None:
        """Enable or disable observing value changes of interactive widgets.

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
        widget_provided = widget is not None
        name_provided = name is not None
        name_regex_provided = name_regex is not None
        if widget_provided:
            if isinstance(widget, Widget):
                widget = [widget]
            for w in widget:
                self._gui__set_widget_value_observer(w, observe=observe)
        if name_provided:
            if isinstance(name, str):
                name = [name]
            for w_name in name:
                self._gui__set_widget_value_observer(
                    self._gui__widget_name_to_widget[w_name],
                    observe=observe
                )
        if name_regex_provided:
            if isinstance(name_regex, str):
                name_regex = re.compile(name_regex)
            for w_name, w in self._gui__widget_name_to_widget.items():
                if name_regex.match(w_name):
                    self._gui__set_widget_value_observer(w, observe=observe)
        if not (widget_provided or name_provided or name_regex_provided):
            # If no specific widgets or names are provided, apply to all widgets
            for w in self._gui__widget_name_to_widget.values():
                self._gui__set_widget_value_observer(w, observe=observe)
        return

    @contextmanager
    def _gui__temporary_observe(
        self,
        observe: bool,
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
        self._gui__observe_value_changes(observe, widget=widget, name=name, name_regex=name_regex)
        try:
            yield
        finally:
            self._gui__observe_value_changes(not observe, widget=widget, name=name, name_regex=name_regex)
        return

    def _gui__render(self, **kwargs) -> None:
        """Render the GUI based on the current state of the widgets.

        If needed, this method should be implemented in the subclass.
        """
        return

    def _gui__on_widget_value_change(self, change: dict[str, Any] | Button) -> None:
        """Handle value changes of interactive widgets.

        This method is added as an observer to all widgets added with `_add_widget`.
        It calls the corresponding observer method based on the widget's name,
        if it exists.
        """
        widget = change if isinstance(change, Button) else change['owner']
        widget_id = id(widget)
        widget_name = self._gui__widget_id_to_name.get(widget_id)
        if not widget_name:
            return
        observer_method_name = f"{self._gui__observer_method_prefix}{widget_name}"
        observer_method = getattr(self, observer_method_name, None)
        if observer_method:
            render_kwargs = observer_method(change)
            if render_kwargs is not None:
                self._gui__render(**render_kwargs)
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
