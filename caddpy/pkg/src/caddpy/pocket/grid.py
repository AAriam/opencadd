
import numpy as np
import scipy as sp
from scids.field import Field


class GridDetector:
    def __init__(self, field: Field):
        self._field = field
        self._volume = sp.ndimage.binary_fill_holes(
            input=self.field.tensor,
            axes=tuple(range(self.field.batch_ndim, self.field.tensor.ndim)),
        )
        return

    @property
    def volume(self) -> np.ndarray:
        """Binary volume of the macromolecule."""
        return np.array(self._volume)

    @property
    def field(self) -> Field:
        return self._field
