"""LIGSITE binding pocket detector.

References
----------
- [LIGSITE: automatic and efficient detection of potential
    small molecule-binding sites in proteins](https://doi.org/10.1016/S1093-3263(98)00002-3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import operator
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import arrayer
from numba import njit, prange, get_num_threads

from caddpy.typing import JAXArray, Int, Bool, Float
from caddpy import exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from scids.field import Field


class LigSite:
    """LIGSITE binding pocket detector.

    This class only implements the core functionality of LIGSITE;
    It calculates the protein-solvent-protein (PSP) events
    in the specified directions, and provides a method to generate masks
    based on the number and distance of these events.
    It is used in the `GridDetector` class,
    which implements the remaining functionality of LIGSITE,
    among other grid-based pocket detection methods.

    Parameters
    ----------
    field
        A voxel `Field` object where
        non-zero values represent the protein volume.
    directions
        Directions in which to calculate PSP events.
        This can be one of the following:
        - An integer array of shape `(n_directions, 3)`,
          where each row is a direction vector
          from one point to another in the 3D grid
          (e.g. `[1, 0, 0]` for the positive x-direction).
          All vectors must be linearly independent,
          and the smallest vector for each direction must be provided.
        - A single integer within the range `[1, 3]`,
          corresponding to 1D, 2D, or 3D directions, respectively.
          N-dimensional directions are those that have N non-zero components.
          For example, 1D directions are `[1, 0, 0]`, `[0, 1, 0]`, and `[0, 0, 1]`,
          corresponding to directions along the x, y, and z axes, respectively.
        - A non-repeating sequence of integers within the range `[1, 3]`,
          to combine 1D, 2D, and 3D directions.
          For example, the default value `(1, 2, 3)`
          will generate all 1D, 2D, and 3D directions,
          corresponding to all 26 directions from
          one grid point to each of its 26 neighbors in a 3x3x3 grid.
    """

    def __init__(
        self,
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray = (1, 2, 3),
        parallel: bool = True,
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
        ps_dist_int = self._calculate_ps_distances(
            tensor=np.asarray(field.tensor),
            dirs=np.asarray(self._dir),
            parallel=parallel,
        )
        self._ps_dist_int = jnp.asarray(ps_dist_int)
        dir_lengths = np.linalg.norm(
            self._dir * field.grid.spacings,
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

        self._psp_count_min = self._psp_count.min().item()
        self._psp_count_max = self._psp_count.max().item()
        self._psp_dist_min = jnp.nanmin(self._psp_dist).item()
        self._psp_dist_max = jnp.nanmax(self._psp_dist).item()
        self._psp_dist_nan = jnp.isnan(self._psp_dist)

        self._psp_mask: dict[str, jnp.ndarray | None] = {
            "count_lower": None,
            "count_upper": None,
            "dist_lower": None,
            "dist_upper": None,
        }
        self._last_mask_spec = {
            "count_lower": None,
            "count_upper": None,
            "dist_lower": None,
            "dist_upper": None,
        }
        return

    def psp_mask(
        self,
        count_lower: int | None = None,
        count_upper: int | None = None,
        dist_lower: float | None = None,
        dist_upper: float | None = None,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = "all",
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = "any",
    ) -> Bool[JAXArray, "*field.shape"] | None:
        """Generate a volume mask based on PSP events.

        This method generates a single boolean mask by combining up to four conditions
        with logical AND operations:
        - Minimum number of PSP events for each grid point, as defined by the `count_lower` parameter.
        - Maximum number of PSP events for each grid point, as defined by the `count_upper` parameter.
        - Minimum PSP distance for each grid point, as defined by the `dist_lower` and `dist_lower_mode` parameters.
        - Maximum PSP distance for each grid point, as defined by the `dist_upper` and `dist_upper_mode` parameters.

        To calculate more complex masks, you can directly access the individual properties
        `psp_count`, `psp_distance`, `ps_distance`, and `ps_distance_discrete` in this class,
        which are the underlying data used to generate the masks.

        To improve performance when generating multiple masks based on slight variations,
        the last submask calculated for each condition is cached, and will be reused in
        subsequent calls to this method if the corresponding argument is the same as the last one.
        Providing a different value will generate a new submask for that condition and update the cache.
        Providing a value of `None` for any parameter (default) will exclude that condition from the mask,
        and will also clear the corresponding cached submask.

        Since each grid point has multiple PSP distances (one for each direction),
        the distance masks are calculated by reducing the PSP distances along the direction axis,
        to obtain a single boolean value for each grid point. The reduction mode can be specified
        using the `dist_lower_mode` and `dist_upper_mode` parameters,
        for the lower and upper distance masks, respectively.
        The supported reduction modes are:

        - `any`: The mask is `True` for a grid point if
          any PSP distance for that point is below/above the threshold.
        - `all`: The mask is `True` for a grid point if
          all PSP distances for that point are below/above the threshold.
        - `max`: The mask is `True` for a grid point if
          the maximum PSP distance for that point is below/above the threshold.
        - `min`: The mask is `True` for a grid point if
          the minimum PSP distance for that point is below/above the threshold.
        - `mean`: The mask is `True` for a grid point if
          the mean PSP distance for that point is below/above the threshold.

        Note that all reduction modes ignore `NaN` values in the PSP distances
        (these are for directions where no PSP event was found).
        If you are interested in points that have no `NaN` values in the PSP distances
        (i.e., points that have PSP events in all directions),
        you can simply set the `count_lower` parameter to the number of directions,
        which will only include points that have PSP events in all directions.

        Parameters
        ----------
        count_lower
            Minimum number of PSP events for each grid point.
        count_upper
            Maximum number of PSP events for each grid point.
        dist_lower
            Minimum PSP distance for each grid point.
        dist_upper
            Maximum PSP distance for each grid point.
        dist_lower_mode
            Reduction mode for calculating the lower PSP distance mask.
        dist_upper_mode
            Reduction mode for calculating the upper PSP distance mask.

        Returns
        -------
        A boolean array of the same shape as the input field tensor,
        where `True` values indicate grid points that satisfy all specified conditions.
        If no conditions are specified,
        i.e., all parameters are `True` or are `None` with no previous cache,
        the method returns `None`.

        Note that the mask does not distinguish between occupied and unoccupied grid points,
        i.e., for unoccupied grid points, it calculates based on protein–solvent–protein (PSP) events,
        while for occupied grid points, it uses solvent–protein–solvent (SPS) events.
        To obtain a mask that is only `True` for unoccupied grid points (i.e., solvent/free space),
        you must simply combine this mask with a mask for unoccupied grid points using logical AND.
        """
        # PSP count
        for input_name, input_count, cap_count, test_op, mask_op in (
            ("count_lower", count_lower, self.psp_count_min, operator.le, operator.ge),
            ("count_upper", count_upper, self.psp_count_max, operator.ge, operator.le),
        ):
            if input_count is None or test_op(input_count, cap_count):
                self._psp_mask[input_name] = None
                self._last_mask_spec[input_name] = None
            elif input_count != self._last_mask_spec[input_name]:
                self._psp_mask[input_name] = mask_op(self.psp_count, input_count)
                self._last_mask_spec[input_name] = input_count

        # PSP distance
        for input_name, input_dist, input_mode, cap_dist, test_op, mask_op in (
            ("dist_lower", dist_lower, dist_lower_mode, self.psp_distance_min, operator.le, operator.ge),
            ("dist_upper", dist_upper, dist_upper_mode, self.psp_distance_max, operator.ge, operator.le),
        ):
            if input_dist is None or test_op(input_dist, cap_dist):
                self._psp_mask[input_name] = None
                self._last_mask_spec[input_name] = None
            elif (input_dist, input_mode) != self._last_mask_spec[input_name]:
                self._last_mask_spec[input_name] = (input_dist, input_mode)
                if input_mode == "any":
                    self._psp_mask[input_name] = jnp.any(mask_op(self.psp_distance, input_dist), axis=-1)
                elif input_mode == "all":
                    self._psp_mask[input_name] = np.all(
                        jnp.logical_or(mask_op(self.psp_distance, input_dist), self.psp_distance_nan),
                        axis=-1
                    )
                elif input_mode in ("max", "min", "mean"):
                    reduction_op = {"max": jnp.nanmax, "min": jnp.nanmin, "mean": jnp.nanmean}[input_mode]
                    self._psp_mask[input_name] = mask_op(reduction_op(self.psp_distance, axis=-1), input_dist)
                else:
                    raise ValueError(f"Unknown mode: {input_mode}")
        # Combine all masks
        active_masks = [mask for mask in self._psp_mask.values() if mask is not None]
        return jnp.logical_and.reduce(jnp.array(active_masks)) if active_masks else None

    @property
    def psp_count(self) -> Int[JAXArray, "*field.shape"]:
        """Number of protein-solvent-protein (PSP) events in each direction.

        For unoccupied grid points, this is equal to the number of solvent–protein–solvent (SPS) events.
        """
        return self._psp_count

    @property
    def psp_count_min(self) -> int:
        """Minimum number of PSP events in any direction."""
        return self._psp_count_min

    @property
    def psp_count_max(self) -> int:
        """Maximum number of PSP events in any direction."""
        return self._psp_count_max

    @property
    def psp_distance(self) -> Float[JAXArray, "*field.shape {self.direction.shape[0] / 2}"]:
        """Protein–solvent–protein (PSP) distances in each direction, in units of grid spacings (e.g. Ångstrom).

        For unoccupied grid points, this is equal to solvent–protein–solvent (SPS) distances.
        A distance of `numpy.nan` means that no PSP/SPS event was found in that direction.
        """
        return self._psp_dist

    @property
    def psp_distance_min(self) -> float:
        """Minimum PSP distance in any direction."""
        return self._psp_dist_min

    @property
    def psp_distance_max(self) -> float:
        """Maximum PSP distance in any direction."""
        return self._psp_dist_max

    @property
    def psp_distance_nan(self) -> Bool[JAXArray, "*field.shape {self.direction.shape[0] / 2}"]:
        """Boolean mask indicating which PSP distances are NaN."""
        return self._psp_dist_nan

    @property
    def ps_distance(self) -> Float[JAXArray, "*field.shape {self.direction.shape[0]}"]:
        """Distances to nearest xeno grid points in each direction, in units of grid spacings (e.g. Ångstrom).

        A distance of `numpy.nan` means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist

    @property
    def ps_distance_discrete(self) -> Int[JAXArray, "*field.shape {self.direction.shape[0]}"]:
        """Distances to nearest xeno grid points in each direction, in units of direction vectors.

        A distance of 0 means that no xeno neighbor was found in that direction.
        """
        return self._ps_dist_int

    @property
    def direction(self) -> Int[JAXArray, "{2 * n_directions} {field.batch_ndim + 3}"]:
        """Direction vectors used for PSP/SPS events calculation.

        This is a 2D array of shape `(2 * n_signed_directions, (field.batch_ndim + 3))`
        containing `2 * n_signed_directions` unit vectors pointing to the neighbors of a grid point in a 3D grid.
        Each vector is padded with leading zeros to match the batch dimensions of the input volume field.
        The vectors are ordered such that `self.direction[i] == -self.direction[-(i + 1)]`,
        i.e., the first `n_signed_directions` vectors are
        the opposite of the last `n_signed_directions` vectors in reverse order.
        These correspond to the positive and negative half-directions
        in which the PSP/SPS events are calculated.
        The order corresponds to the order of distances in the `psp_distance`,
        `ps_distance`, and `ps_distance_discrete` properties.
        """
        return self._dir

    @staticmethod
    def calculate_direction_vectors(
        field: Field,
        directions: Literal[1, 2, 3] | Sequence[Literal[1, 2, 3]] | np.ndarray = (1, 2, 3),
    ) -> jax.Array:
        """Validate and calculate direction vectors for PSP events.

        This function does not need to be called by the user directly;
        it is called during the initialization,
        and used as a utility method in the `GridDetectorGUI` class
        to compare new directions with the existing ones.
        Parameters are the same as in the `LigSite` class constructor.
        """
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
            dir_vectors = field.grid.direction_vectors(dimensions=directions)
            assert dir_vectors.ndim == 2, "Direction vectors should be 2-dimensional."
            assert dir_vectors.shape[1] == 3, "Direction vectors should be 3D."
        elif directions.ndim == 2:
            if directions.shape[1] != 3:
                raise ValueError("Directions must be 3D")
            linearly_dependents = arrayer.matrix.linearly_dependent_pairs(directions)
            if linearly_dependents.size > 0:
                raise exception.InputError(
                    name="directions",
                    message="Following direction vector pairs "
                            f"are linearly dependent: {linearly_dependents.tolist()}."
                )
            dir_vectors = np.concatenate([directions, -directions[::-1]], axis=0)
        else:
            raise ValueError("Directions must be 1D or 2D array-like.")
        if (ndirs := dir_vectors.shape[0]) % 2 != 0:
            raise ValueError(f"There should be an even number of direction vectors, but got {ndirs}.")
        if not np.all(dir_vectors[:ndirs] + dir_vectors[-1:-(ndirs+1):-1] == 0):
            raise ValueError("The first half of the direction vectors should be the negative of the second half.")
        return jnp.asarray(dir_vectors)

    @staticmethod
    def _calculate_ps_distances(tensor: np.ndarray, dirs: np.ndarray, parallel: bool = True) -> jax.Array:
        tensor = np.ascontiguousarray(tensor, dtype=np.bool_)
        ndirs = int(dirs.shape[0])

        # Batch (possibly empty) and spatial shapes
        nx, ny, nz = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
        batch_shape = tensor.shape[:-3]
        batch_size = int(np.prod(batch_shape))

        # Flatten batch dimensions
        tensor_reshaped = tensor.reshape(batch_size, nx, ny, nz)

        # Output buffer
        out = np.zeros((batch_size, nx, ny, nz, ndirs), dtype=np.uint32)

        if parallel and get_num_threads() <= 1:
            # If parallel is requested but only one thread is available, use serial kernel
            warnings.warn(
                "Parallel execution requested but only one thread is available. "
                "Falling back to serial execution.",
                UserWarning,
            )
            parallel = False
        kernel = _ps_distance_parallel if parallel else _ps_distance_serial
        kernel(tensor_reshaped, dirs, out)
        # Reshape back to original batch shape
        return out.reshape((*batch_shape, nx, ny, nz, ndirs))


@njit(cache=True, parallel=True)
def _ps_distance_parallel(
    tensor: np.ndarray,
    dirs: np.ndarray,
    out: np.ndarray,
) -> None:
    nbatch, nx, ny, nz = tensor.shape
    ndir = dirs.shape[0]
    nloop = nbatch * nx
    max_axis_size = max(nx, ny, nz)
    for iloop in prange(nloop):
        ibatch = iloop // nx
        ix = iloop % nx
        for iy in range(ny):
            for iz in range(nz):
                value = tensor[ibatch, ix, iy, iz]
                for idir in range(ndir):
                    dx = int(dirs[idir, 0])
                    dy = int(dirs[idir, 1])
                    dz = int(dirs[idir, 2])

                    m_x = _axis_bound(nx, ix, dx, max_axis_size)
                    m_y = _axis_bound(ny, iy, dy, max_axis_size)
                    m_z = _axis_bound(nz, iz, dz, max_axis_size)
                    max_steps = min(m_x, m_y, m_z)

                    x = ix
                    y = iy
                    z = iz
                    step = 0
                    for m in range(1, max_steps + 1):
                        x += dx
                        y += dy
                        z += dz
                        if value != tensor[ibatch, x, y, z]:
                            step = m
                            break
                    out[ibatch, ix, iy, iz, idir] = step
    return


@njit(cache=True, parallel=False)
def _ps_distance_serial(
    tensor: np.ndarray,
    dirs: np.ndarray,
    out: np.ndarray
) -> None:
    nbatch, nx, ny, nz = tensor.shape
    ndir = dirs.shape[0]
    max_axis_size = max(nx, ny, nz)
    for ibatch in range(nbatch):
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    value = tensor[ibatch, ix, iy, iz]
                    for idir in range(ndir):
                        dx = int(dirs[idir, 0])
                        dy = int(dirs[idir, 1])
                        dz = int(dirs[idir, 2])

                        m_x = _axis_bound(nx, ix, dx, max_axis_size)
                        m_y = _axis_bound(ny, iy, dy, max_axis_size)
                        m_z = _axis_bound(nz, iz, dz, max_axis_size)
                        max_steps = min(m_x, m_y, m_z)

                        x = ix
                        y = iy
                        z = iz
                        step = 0
                        for m in range(1, max_steps + 1):
                            x += dx
                            y += dy
                            z += dz
                            if value != tensor[ibatch, x, y, z]:
                                step = m
                                break
                        out[ibatch, ix, iy, iz, idir] = step
    return


@njit(inline='always')
def _axis_bound(size: int, index: int, delta: int, max_size: int) -> int:
    """Calculate the maximum in-bounds steps along one axis.

    Parameters
    ----------
    size
        Size of the axis, i.e., number of points along that axis.
    index
        Current index along the axis (0-based).
    delta
        Step size along the axis (positive or negative).
    max_size
        Maximum axis size among all axes; used to handle the case when `delta` is 0.
    """
    num = (size - 1 - index) * (delta > 0) + index * (delta < 0) + max_size * (delta == 0)
    return num // (abs(delta) if delta != 0 else 1)
