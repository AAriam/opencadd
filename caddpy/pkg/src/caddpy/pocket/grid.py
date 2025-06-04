
from collections.abc import Sequence
from typing import Literal, Any

import numpy as np
import scipy as sp
from scids.field import Field
import scicoda

from scids.field import Field
from scids.grid import Grid

from caddpy.chemsys import ChemicalSystem


class GridDetector:
    def __init__(self, field: Field):
        self._field = field
        # if closing_structure is None:
        #     hydrogen_vdw_diameter = scicoda.atom.van_der_waals_radii()[0] * 2
        #     hydrogen_vdw_diameter_grid_point_count = int(hydrogen_vdw_diameter // self.field.grid.spacings[0])
        #     closing_structure = np.ones(shape=(hydrogen_vdw_diameter_grid_point_count, ) * 3)
        # grid_axes = tuple(range(self.field.batch_ndim, self.field.tensor.ndim))
        # volume_closed = sp.ndimage.binary_closing(
        #     input=self.field.tensor,
        #     structure=closing_structure,
        #     iterations=1,
        #     border_value=1,
        #     axes=grid_axes,
        # )
        # volume_closed_and_filled = sp.ndimage.binary_fill_holes(
        #     input=self.field.tensor,
        #     axes=grid_axes,
        # )
        # self._volume = volume_closed_and_filled
        return

    # @property
    # def volume(self) -> np.ndarray:
    #     """Binary volume of the macromolecule."""
    #     return np.array(self._volume)

    # @property
    # def volume_negative(self) -> np.ndarray:
    #     """Binary volume of the empty space sorrounding the macromolecule."""
    #     return np.logical_not(self._volume)

    @property
    def field(self) -> Field:
        return self._field
