
from collections.abc import Sequence
from typing import Literal, Any
import operator

import numpy as np
import jax.numpy as jnp
import jax
import scipy as sp
from scids.field import Field
import scicoda

from scids.field import Field
from scids.grid import Grid
from scids import exception

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


class LigSite:
    def __init__(
        self,
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray | None = None,
    ):
        # Validate inputs
        if field.field_ndim != 0:
            raise exception.InputError(
                name="field",
                message=f"Volume field must be scalar (0D), but is {field.field_ndim}D.",
            )
        if field.grid.dimension != 3:
            raise exception.InputError(
                name="field",
                message=f"Volume field must have a 3D grid, but is {field.grid.dimension}D.",
            )
        self._dir = self.calculate_direction_vectors(
            field=field,
            directions=directions,
        )

        # Calculate distance of each grid point to the nearest xeno grid point
        # in each half direction, in units of corresponding distance vectors
        ps_dist_int = field.nearest_target_distances(
            direction_vectors=self._dir,
            predicate=np.logical_xor,
        )
        self._ps_dist_int = jnp.asarray(ps_dist_int)
        dir_lengths = np.linalg.norm(
            self._dir[:, field.batch_ndim:] * field.grid.spacings,
            axis=-1
        )
        # Multiply by direction vector lengths, to get the real distances.
        ps_dists_float = ps_dist_int * dir_lengths
        # set distances that are 0 (meaning no neighbor was found in that direction) to NaN.
        ps_dists_float[ps_dists_float == 0] = np.nan
        self._ps_dist = jnp.asarray(ps_dists_float)
        # Add distances to neighbors in positive half-directions to distances to neighbors in
        # negative half-directions, in order to get the PSP lengths.
        ndir = self._dir.shape[0] // 2
        self._psp_dist = self._ps_dist[..., :ndir] + self._ps_dist[..., -1:-(ndir+1):-1]
        self._psp_count = jnp.count_nonzero(~jnp.isnan(self._psp_dist), axis=-1)

        self._psp_mask: dict[str, jnp.ndarray | None] = {
            "count_lower": None,
            "count_upper": None,
            "dist_lower": None,
            "dist_upper": None,
        }
        return

    def psp_mask(
        self,
        count_lower: int | bool | None = None,
        count_upper: int | bool | None = None,
        dist_lower: float | bool | None = None,
        dist_upper: float | bool | None = None,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = "all",
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = "any",
    ) -> jax.Array:
        def make_mask(
            arr: jnp.ndarray,
            threshold: int | float,
            side: Literal["lower", "upper"],
            mode: Literal["any", "all", "max", "min", "mean"]
        ):
            comparison_op = operator.le if side == "upper" else operator.ge
            reduction_op = {
                "any": jnp.any,
                "all": jnp.all,
                "max": jnp.nanmax,
                "min": jnp.nanmin,
                "mean": jnp.nanmean,
            }[mode]
            if mode in ("any", "all"):
                return reduction_op(comparison_op(arr, threshold), axis=-1)
            elif mode in ("max", "min", "mean"):
                return comparison_op(reduction_op(arr, axis=-1), threshold)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        if count_lower is not None:
            self._psp_mask["count_lower"] = None if count_lower is True else self.psp_count >= count_lower
        if count_upper is not None:
            self._psp_mask["count_upper"] = None if count_upper is True else self.psp_count <= count_upper
        if dist_lower is not None:
            self._psp_mask["dist_lower"] = None if dist_lower is True else make_mask(
                self.psp_distance, threshold=dist_lower, side="lower", mode=dist_lower_mode
            )
        if dist_upper is not None:
            self._psp_mask["dist_upper"] = None if dist_upper is True else make_mask(
                self.psp_distance, threshold=dist_upper, side="upper", mode=dist_upper_mode
            )
        active_masks = [mask for mask in self._psp_mask.values() if mask is not None]
        return jnp.logical_and.reduce(active_masks, out=self._psp_mask) if active_masks else jnp.ones(shape=self._psp_count.shape, dtype=bool)

    @property
    def psp_count(self) -> jax.Array:
        """Number of protein-solvent-protein (PSP) events in each direction.

        For unoccupied grid points, this is equal to the number of solvent–protein–solvent (SPS) events.
        """
        return self._psp_count

    @property
    def psp_distance(self) -> jax.Array:
        """Protein–solvent–protein (PSP) distances in each direction, in units of grid spacings (e.g. Ångstrom).

        For unoccupied grid points, this is equal to solvent–protein–solvent (SPS) distances.
        """
        return self._psp_dist

    @property
    def ps_distance(self) -> jax.Array:
        """Distances to nearest xeno grid points in each direction, in units of grid spacings (e.g. Ångstrom).

        A distance of `numpy.nan` means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist

    @property
    def ps_distance_discrete(self) -> jax.Array:
        """Distances to nearest xeno grid points in each direction, in units of direction vectors.

        A distance of 0 means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist_int

    @property
    def direction(self) -> jax.Array:
        """Direction vectors for PSP events.

        This is a 2D array of shape `(26, (self.field.batch_ndim + 3))`
        containing 26 unit vectors pointing to the 26 neighbors of a grid point in a 3D grid.
        Each vector is padded with leading zeros to match the batch dimensions of volume.
        The vectors are ordered such that `self.directions[i] == -self.directions[-(i + 1)]`,
        i.e., the first 13 vectors are the opposite of the last 13 vectors in reverse order.
        """
        return self._dir

    @staticmethod
    def calculate_direction_vectors(
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray | None = None,
    ) -> jax.Array:
        if directions is None:
            directions = [1, 2, 3]
        if isinstance(directions, int):
            directions = [directions]
        directions = np.asarray(directions)
        if not np.issubdtype(directions.dtype, np.integer):
            raise TypeError("Directions must be integers.")
        if directions.ndim == 1:
            if not np.all(np.isin(directions, [1, 2, 3])):
                raise ValueError("Directions must be 1, 2, or 3.")
            if len(set(directions)) != len(directions):
                raise ValueError("Directions must be unique.")
            dir_vectors = field.grid_direction_vectors(dimensions=directions)
            assert dir_vectors.ndim == 2, "Direction vectors should be 2-dimensional."
            assert dir_vectors.shape[1] == 3, "Direction vectors should be 3D."
        elif directions.ndim == 2:
            if directions.shape[1] != 3:
                raise ValueError("Directions must be 3D")
            dir_vectors = np.pad(
                directions,
                pad_width=((0, 0), (field.batch_ndim, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            raise ValueError("Directions must be 1D or 2D array-like.")
        if (ndirs := dir_vectors.shape[0]) % 2 != 0:
            raise ValueError(f"There should be an even number of direction vectors, but got {ndirs}.")
        if not np.all(dir_vectors[:ndirs] + dir_vectors[-1:-(ndirs+1):-1] == 0):
            raise ValueError("The first half of the direction vectors should be the negative of the second half.")
        return jnp.asarray(dir_vectors)
