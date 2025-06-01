"""LIGSITE binding pocket detector.

References
----------
- [LIGSITE: automatic and efficient detection of potential small molecule-binding sites in proteins](https://doi.org/10.1016/S1093-3263(98)00002-3)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Any

import jax.numpy as jnp
import numpy as np

from scids.field import Field
from scids.grid import Grid

from caddpy.pocket.grid import GridDetector
from caddpy.chemsys import ChemicalSystem


class LigSiteDetector(GridDetector):
    """LIGSITE binding pocket detector."""
    def __init__(self, field: Field, closing_structure: np.ndarray | None = None):
        super().__init__(field=field, closing_structure=closing_structure)
        ndir = 13  # number of directions in a 3x3x3 unit cell
        self._dirs = self.field.grid_direction_vectors()
        assert self._dirs.ndim == 2, "Direction vectors should be 2-dimensional."
        assert self._dirs.shape[0] == ndir * 2, "There should be 26 direction vectors (13 directions, each with positive and negative half-direction)."
        assert np.all(self._dirs[:ndir] + self._dirs[-1:-(ndir+1):-1] == 0), "The first 13 direction vectors should be the negative of the last 13 direction vectors."
        dir_lengths = np.linalg.norm(
            self._dirs[:, self.field.batch_ndim:] * self.field.grid.spacings,
            axis=-1
        )
        # Calculate distance of each grid point to the nearest xeno grid point
        # in each half direction, in units of corresponding distance vectors
        self._ps_dists_int = self.field.nearest_target_distances(
            direction_vectors=self._dirs,
            predicate=np.logical_xor,
        )
        # Multiply by direction vector lengths, to get the real distances.
        ps_dists_float = self._ps_dists_int * dir_lengths
        # set distances that are 0 (meaning no neighbor was found in that direction) to NaN.
        ps_dists_float[ps_dists_float == 0] = np.nan
        self._ps_dists = ps_dists_float
        # Add distances to neighbors in positive half-directions to distances to neighbors in
        # negative half-directions, in order to get the PSP lengths.
        self._psp_dists = self._ps_dists[..., :ndir] + self._ps_dists[..., -1:-(ndir+1):-1]
        self._psp_counts = np.count_nonzero(~np.isnan(self._psp_dists), axis=-1)
        return

    @property
    def psp_count(self) -> np.ndarray:
        """Number of protein-solvent-protein (PSP) events in each direction.

        For unoccupied grid points, this is equal to the number of solvent–protein–solvent (SPS) events.
        """
        return np.array(self._psp_counts)

    @property
    def psp_distance(self) -> np.ndarray:
        """Protein–solvent–protein (PSP) distances in each direction, in units of grid spacings (e.g. Ångstrom).

        For unoccupied grid points, this is equal to solvent–protein–solvent (SPS) distances.
        """
        return np.array(self._psp_dists)

    @property
    def ps_distance(self) -> np.ndarray:
        """Distances to nearest xeno grid points in each direction, in units of grid spacings (e.g. Ångstrom).

        A distance of `numpy.nan` means that no xeno neighbor was found in that direction.
        """
        return np.array(self._ps_dists)

    @property
    def ps_distance_discrete(self) -> np.ndarray:
        """Distances to nearest xeno grid points in each direction, in units of direction vectors.

        A distance of 0 means that no xeno neighbor was found in that direction.
        """
        return np.array(self._ps_dists_int)

    @property
    def direction(self) -> np.ndarray:
        """Direction vectors for PSP events.

        This is a 2D array of shape `(26, (self.field.batch_ndim + 3))`
        containing 26 unit vectors pointing to the 26 neighbors of a grid point in a 3D grid.
        Each vector is padded with leading zeros to match the batch dimensions of volume.
        The vectors are ordered such that `self.directions[i] == -self.directions[-(i + 1)]`,
        i.e., the first 13 vectors are the opposite of the last 13 vectors in reverse order.
        """
        return np.array(self._dirs)

    def calculate_buriedness(
        self,
        psp_max_length: float = 20.0,
        psp_min_count: int = 2,
    ) -> np.ndarray:
        """
        Calculate whether each grid point is buried inside the target structure or not, based on
        counting the number of protein-solvent-protein (PSP) events for each point, and applying
        a cutoff.

        Parameters
        ----------
        vacancy : numpy.ndarray

        psp_distances : numpy.ndarray
        psp_max_length : float, Optional, default: 10.0
            Maximum acceptable distance for a PSP event, in Ångstrom (Å).
        psp_min_count : int, Optional, default: 4
            Minimum required number of PSP events for a grid point, in order to count as buried.
        """
        # Count PSP events that are shorter than the given cutoff.
        grid_psp_counts = jnp.count_nonzero(self._psp_distances <= psp_max_length, axis=-1)
        buriedness = grid_psp_counts >= psp_min_count
        site = jnp.logical_and(jnp.logical_not(self.toxel_vol.toxels), buriedness)
        return site


def from_chemsys(
    system: ChemicalSystem,
    grid: int | float | Sequence[int | float] | Grid = 0.5,
    instance_selection: Any = None,
    closing_structure: np.ndarray | None = None,
):
    field = system.toxelate(grid=grid, instance_selection=instance_selection)
    return LigSiteDetector(field=field, closing_structure=closing_structure)
