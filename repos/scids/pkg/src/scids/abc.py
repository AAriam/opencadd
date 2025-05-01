from abc import ABC, abstractmethod

import numpy.typing as npt


class Volume(ABC):
    """An n-dimensional volume form sampled at one or several instances.

    This is can be a length, area, volume, hyper-volume, etc.
    """

    @property
    @abstractmethod
    def size(self) -> npt.ArrayLike:
        """Size of the volume at each instance."""
        ...

    @property
    @abstractmethod
    def aabb_lower_bounds(self) -> npt.ArrayLike:
        """Lower bounds of the axis-aligned minimum bounding box."""
        ...

    @property
    @abstractmethod
    def aabb_upper_bounds(self) -> npt.ArrayLike:
        """Upper bounds of the axis-aligned minimum bounding box."""
        ...

    @property
    @abstractmethod
    def aabb_lengths(self) -> npt.ArrayLike:
        """Lengths of the axis-aligned minimum bounding box."""
        ...
