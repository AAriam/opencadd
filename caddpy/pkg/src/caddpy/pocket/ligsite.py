"""LIGSITE binding pocket detector.

References
----------
- [LIGSITE: automatic and efficient detection of potential small molecule-binding sites in proteins](https://doi.org/10.1016/S1093-3263(98)00002-3)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Any
import operator

import jax.numpy as jnp
import numpy as np
import scipy as sp

from scids.field import Field
from scids.grid import Grid

from caddpy.pocket.grid import GridDetector
from caddpy.pocket.ligsite_gui import LigSiteDetectorGUI
from caddpy.chemsys import ChemicalSystem


class LigSiteDetector(GridDetector):
    """LIGSITE binding pocket detector."""
    def __init__(self, field: Field):
        super().__init__(field=field)
        self._grid_axis_indices = tuple(range(self.field.batch_ndim, self.field.tensor.ndim))
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

        self._psp_masks: dict[str, np.ndarray | None] = {
            "count_lower": None,
            "count_upper": None,
            "dist_lower": None,
            "dist_upper": None,
        }
        self._psp_mask: np.ndarray | None = None
        self._volume_mask = np.logical_not(self.field.tensor)
        self._pocket_mask: np.ndarray | None = None
        self._gui = None
        return

    def pocket_mask(self, additional_mask: np.ndarray | None = None):
        masks = [self._volume_mask]
        if self._psp_mask is not None:
            masks.append(self._psp_mask)
        if additional_mask is not None:
            masks.append(additional_mask)
        self._pocket_mask = np.logical_and.reduce(masks)
        return

    def psp_mask(
        self,
        count_lower: int | bool | None = None,
        count_upper: int | bool | None = None,
        dist_lower: float | bool | None = None,
        dist_upper: float | bool | None = None,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = "all",
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = "any",
    ):
        def make_mask(
            arr: np.ndarray,
            threshold: int | float,
            side: Literal["lower", "upper"],
            mode: Literal["any", "all", "max", "min", "mean"]
        ):
            comparison_op = operator.le if side == "upper" else operator.ge
            reduction_op = {
                "any": np.any,
                "all": np.all,
                "max": np.max,
                "min": np.min,
                "mean": np.mean,
            }[mode]
            if mode in ("any", "all"):
                return reduction_op(comparison_op(arr, threshold), axis=-1)
            elif mode in ("max", "min", "mean"):
                return comparison_op(reduction_op(arr, axis=-1), threshold)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        if count_lower is not None:
            self._psp_masks["count_lower"] = None if count_lower is True else self.psp_count >= count_lower
        if count_upper is not None:
            self._psp_masks["count_upper"] = None if count_upper is True else self.psp_count <= count_upper
        if dist_lower is not None:
            self._psp_masks["dist_lower"] = None if dist_lower is True else make_mask(
                self.psp_distance, threshold=dist_lower, side="lower", mode=dist_lower_mode
            )
        if dist_upper is not None:
            self._psp_masks["dist_upper"] = None if dist_upper is True else make_mask(
                self.psp_distance, threshold=dist_upper, side="upper", mode=dist_upper_mode
            )
        active_masks = [mask for mask in self._psp_masks.values() if mask is not None]
        if not active_masks:
            self._psp_mask = None
        else:
            self._psp_mask = np.logical_and.reduce(active_masks, out=self._psp_mask)
        return

    def volume_mask(
        self,
        closing_structure: np.ndarray | tuple[int, int] | None = None,
        closing_iterations: int = 1,
        closing_mask: np.ndarray | None = None,
        closing_border_value: Literal[0, 1] = 1,
        fill_structure: np.ndarray | None = None,
    ):
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
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
        volume_closed_and_filled = sp.ndimage.binary_fill_holes(
            input=volume_closed,
            structure=fill_structure,
            axes=self._grid_axis_indices,
        )
        self._volume_mask = np.logical_not(volume_closed_and_filled)
        return

    @property
    def gui(self):
        if not self._gui:
            self._gui = LigSiteDetectorGUI(self)

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


def from_chemsys(
    system: ChemicalSystem,
    grid: int | float | Sequence[int | float] | Grid = 0.5,
    instance_selection: Any = None,
):
    field = system.toxelate(grid=grid, instance_selection=instance_selection)
    return LigSiteDetector(field=field)
