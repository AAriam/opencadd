from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.spatial

import scids
from scids import exception
import scids.util

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from scids.typing import ArrayLike


class DynamicPointCloud:
    """A discrete set of points in n-dimensional space.

    Thid represents a set of features observed for
    each individual in a sample/population at different instances.
    For example, instances may be consecutive timepoints,
    the individuals in the sample may be a set of particles,
    and the observed feature may be the position of particles in 3-dimensional space,
    in which case the point cloud represents a trajectory.

    Parameters
    ----------
    points
        A 3D array of shape (i, j, k),
        corresponding to `k` features observed for `j` individuals
        at `i` instances.
    """

    __slots__ = (
        "_points",
        "_points_2d",
        "_kdtree_combined",
        "_kdtrees_per_instance",
    )

    def __init__(self, points: ArrayLike):
        # Check for errors in `data`:
        points = jnp.asarray(points)
        if points.ndim != 3:
            raise ValueError(
                "Parameter `data` expects a 3D array, "
                f"but input argument had {points.ndim} dimensions."
            )
        self._points = points
        self._points_2d: jnp.ndarray = self._points.reshape(-1, self._points.shape[-1])
        self._kdtrees_per_instance: list[sp.spatial.KDTree] = None
        self._kdtree_combined: sp.spatial.KDTree = None
        return

    @property
    def points(self) -> jnp.ndarray:
        """Coordinates of the points in the point cloud."""
        return self._points

    @property
    def count_instances(self) -> int:
        return self._points.shape[0]

    @property
    def point_count_per_instance(self) -> int:
        return self._points.shape[1]

    @property
    def point_count_total(self) -> int:
        return np.prod(self._points.shape[:2])

    @property
    def dimension_points(self) -> int:
        return self._points.shape[2]

    @property
    def kdtree(self) -> sp.spatial.KDTree:
        """Combined KDTree of all points in the point cloud."""
        if self._kdtree_combined is None:
            self._kdtree_combined = sp.spatial.KDTree(self._points_2d)
        return self._kdtree_combined

    @property
    def kdtrees(self) -> list[sp.spatial.KDTree]:
        """KDTree of each instance in the point cloud."""
        if self._kdtrees_per_instance is None:
            self._kdtrees_per_instance = [sp.spatial.KDTree(data=points) for points in self._points]
        return self._kdtrees_per_instance

    def aabb(self, per_instance: bool = True) -> scids.volume.RectangularCuboid:
        """Axis-aligned minimum bounding box ([AABB](https://en.wikipedia.org/wiki/Minimum_bounding_box#Axis-aligned_minimum_bounding_box)) of the point cloud.

        Also known as the [axis-aligned minimum bounding rectangle](https://en.wikipedia.org/wiki/Minimum_bounding_rectangle),
        this is the smallest box that contains all points in the point cloud,
        with its sides aligned with the coordinate axes.

        Parameters
        ----------
        per_instance
            Whether to calculate one AABB per instance,
            or one for all instances superposed.

        References
        ----------
        - [Bounding volume - Wikipedia](https://en.wikipedia.org/wiki/Bounding_volume)

        See Also
        --------
        - `DynamicPointCloud.minimize_aabb`: Minimize the AABB volume of the point cloud.
        """
        if per_instance:
            mins = jnp.min(self._points, axis=1)
            maxes = jnp.max(self._points, axis=1)
        else:
            mins = jnp.expand_dims(jnp.min(self._points, axis=(0, 1)), axis=0)
            maxes = jnp.expand_dims(jnp.max(self._points, axis=(0, 1)), axis=0)
        return scids.volume.RectangularCuboid(lower_bounds=mins, upper_bounds=maxes)

    def toxelate(
        self,
        grid: float | Sequence[float] | scids.grid.Grid,
        point_radii: float | npt.ArrayLike,
        padding: float | npt.ArrayLike = 0,
    ) -> scids.volume.ToxelVolume:
        """Create a Toxel volume from the point cloud.

        Parameters
        ----------
        grid
            Grid to use for the Toxel volume.
            If a `Grid` object is provided, it is used directly.
            Otherwise, a float is interpreted as the spacing
            between grid points in all dimensions,
            and a sequence is interpreted as the spacing in each dimension.
            In both cases, the grid is created to cover the entire
            axis-aligned minimum bounding box of the combined instances.
        point_radii
            Radius of the points in the point cloud.
            If a float is provided, it is interpreted as the radius of all points.
            If an array is provided, it must have the same length
            as the number of points in the point cloud.
        padding

        """
        if isinstance(grid, scids.grid.Grid):
            grid = grid
        else:
            # Get the bounding box of all instances superposed.
            total_bounding_box = self.aabb(per_instance=False)
            # Create a grid the size of the total bounding box, with given resolution
            grid = scids.grid.from_bounds_spacing(
                lower_bounds=total_bounding_box.lower_bounds[0] - padding,
                upper_bounds=total_bounding_box.upper_bounds[0] + padding,
                spacings=grid,
            )
        # If `point_radii` is a scalar (i.e. int or float), it means all points have the same
        # radius, and thus we only need to query for the first nearest neighbor of each point:
        radii_type = type(point_radii)
        if any(np.issubdtype(radii_type, scalar_type) for scalar_type in (np.integer, np.floating)):
            # Radius must be positive:
            if point_radii <= 0:
                raise exception.InputError(
                    name="point_radii",
                    message=f"A positive real number is expected, but got {point_radii}."
                )

            dists, indices = self.nearest_neighbors(
                points=grid.coordinates,
                count=1,
                per_instance=True,
                distance_upper_bound=point_radii,
            )
            # Each toxel on the grid is occupied when the nearest point in self is within
            # `point_radii`, i.e. when it is not `np.inf`, since `self.nearest_neighbors` returns
            # `np.inf` for points where the nearest distance is larger than the upper bound.
            toxel_tensor = dists != np.inf  # True when toxel is occupied
            return scids.field.from_tensor(
                grid=grid,
                tensor=np.squeeze(toxel_tensor, axis=-1),
            )
        # If `point_radii` is an array of values, then we cannot rely only on the distances to
        # first nearest neighbors, since it is possible that the first k nearest neighbors have
        # small radii and do not overlap with the toxel, while the (k+1)-th neighbor has a large
        # enough radius to overlap.
        radii_array = np.asarray(point_radii)
        max_radius = radii_array.max()
        min_radius = radii_array.min()
        toxel_tensor = np.zeros(shape=(self.count_instances, grid.shape, 1), dtype=np.bool_)
        ind_self, ind_gird, dists = self.distance_matrix_sparse(
            points=grid, max_distance=max_radius
        )
        filter_definitely_occupied = dists <= min_radius
        grid_inds_occupied = np.unravel_index(
            ind_gird[1][filter_definitely_occupied], shape=grid.shape
        )
        toxel_tensor[(ind_self[0][filter_definitely_occupied], *grid_inds_occupied)] = True
        filter_maybe_occupied = jnp.logical_not(filter_definitely_occupied)
        dists_from_surface = (
            dists[filter_maybe_occupied] - radii_array[ind_self[1][filter_maybe_occupied]]
        )
        occupied = dists_from_surface <= 0
        grid_inds_occupied2 = np.unravel_index(
            ind_gird[1][filter_maybe_occupied][occupied], shape=grid.shape
        )
        toxel_tensor[(ind_self[0][filter_maybe_occupied][occupied], *grid_inds_occupied2)] = True
        return scids.field.from_tensor(
            grid=grid,
            tensor=toxel_tensor,
        )

    def distance_matrix_sparse(
        self,
        points: DynamicPointCloud,
        max_distance: float,
        p_norm: float = 2,
        output_type: Literal[
            "nd_unraveled", "dok_matrix", "coo_matrix", "dict", "ndarray"
        ] = "nd_unraveled",
    ):
        dist_matrix = self.kdtree.sparse_distance_matrix(
            other=points.kdtree,
            max_distance=max_distance,
            p=p_norm,
            output_type="ndarray" if output_type == "nd_unraveled" else output_type,
        )
        if output_type != "nd_unraveled":
            return dist_matrix
        indices_self = np.unravel_index(
            dist_matrix["i"], shape=(self.count_instances, self.point_count_per_instance)
        )
        indices_other = np.unravel_index(
            dist_matrix["j"], shape=(points.count_instances, points.point_count_per_instance)
        )
        return indices_self, indices_other, dist_matrix["v"]

    def find_point_pairs_within_radius(
        self,
    ):
        pass

    def count_neighbors_within_radius(self, points):
        pass

    def minimize_aabb(
        self,
        instance_slice: slice = slice(None),
        centered: bool = True,
        algorithm: Literal["pca", "hull", "auto"] = "auto",
    ) -> DynamicPointCloud:
        """Minimize the [axis-aligned minimum bounding box](https://en.wikipedia.org/wiki/Minimum_bounding_box#Axis-aligned_minimum_bounding_box) volume of the point cloud.

        This is done by rotating the point cloud around its center of mass.

        Parameters
        ----------
        instance_slice
            Slice of instances to consider.
            By default, all instances are considered,
            i.e., the AABB is calculated for all instances superposed.
        centered
            Whether to keep the center of mass of the point cloud at the origin.
            To find the optimal rotation, this method needs to first translate
            the point cloud's center of mass to the origin.
            If this argument is set to False, the point cloud is translated back
            to its original position after rotation,
            otherwise the point cloud is left centered at the origin.
        algorithm
            Algorithm to use for finding the rotation.
            - "pca": Principal Component Analysis (PCA).
              This works for any number of dimensions.
              However, it is a is not guaranteed to find the optimal rotation,
              but it is usually a good approximation.
            - "hull": Convex hull-based brute-force search.

        Notes
        -----
        This method uses principal component analysis (PCA) to find the rotation.
        It is not guaranteed to find the optimal rotation,
        but it is a good approximation
        (cf. [On the bounding boxes obtained by principal component analysis](https://www.researchgate.net/publication/235758825_On_the_bounding_boxes_obtained_by_principal_component_analysis)).
        For other algorithms, see:
        - [Minimum bounding box algorithms - Wikipedia](https://en.wikipedia.org/wiki/Minimum_bounding_box_algorithms)
        - https://perso.uclouvain.be/chia-tche.chang/resources/CGM11_paper.pdf
        - https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
        - https://math.stackexchange.com/questions/2342844/how-to-find-the-rotation-which-minimizes-the-volume-of-the-bounding-box
        """
        if algorithm == "hull" and self.dimension_points != 3:
            raise exception.InputError(
                name="algorithm",
                message="The 'hull' algorithm is only available for 3D point clouds."
            )
        if algorithm == "auto":
            algorithm = "hull" if self.dimension_points == 3 else "pca"

        # if algorithm == "pca":
        #     scids.tensor.pca(
        #         points=
        #     )

        points = self.points[instance_slice].reshape(-1, self.dimension_points)
        center = jnp.mean(points, axis=0)
        points_centered = points - center
        u, s, vt = jnp.linalg.svd(points_centered, full_matrices=False)
        # Flip eigenvectors' sign to enforce deterministic output:
        #  Adjusts the columns of u and the rows of v such that the loadings in the
        #  columns in u that are largest in absolute value are always positive.
        max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
        signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
        u *= signs
        vt *= signs[:, jnp.newaxis]
        principal_components = vt
        variance = s ** 2 / points.shape[0]
        variance_normalized = variance / variance.sum()

        all_points_centered = self._points_2d - center
        all_points_rotated = all_points_centered @ principal_components.T
        if not centered:
            all_points_rotated += center
        return DynamicPointCloud(all_points_rotated.reshape(self._points.shape))

    def nearest_neighbors(
        self,
        points: npt.ArrayLike,
        count: int | Sequence[int] = 1,
        per_instance: bool = True,
        instance_slice: slice = slice(None),
        error_tolerance: float = 0,
        p_norm: float = 2,
        distance_upper_bound: float = np.inf,
        distance_dtype: npt.DTypeLike = np.single,
    ):
        """Find the nearest points in self to a given set of points.

        For each point in `points`, find the distances to, and indices of,
        a given number of nearest points in self.

        Parameters
        ----------
        points : numpy.ndarray, shape: (d1, d2, ..., d{m-1}, self.dimension_points)
            Coordinates of n points (n = d1 * d2 * ... * d{m-1}),
            for which the nearest points in self must be found.
        count : int | Sequence[int], optional, default: 1
            Either the number of nearest neighbors (as an integer), or a sequence of
            the k-th nearest neighbors to find.
        per_instance
            Whether to calculate the nearest neighbors for each instance separately,
            or for all instances combined.
        instance_slice
            Slice of instances to consider.
            By default, all instances are considered.
            This only has an effect when `per_instance` is True.
        error_tolerance : float, optional, default: 0
            Tolerance for error in finding the nearest atoms. The k-th nearest atom will be
            within (1 + eps) times the distance to the real k-th nearest atom.
        p_norm : float, range: [1, inf), optional, default: 2
            The Minkowski p-norm to use, e.g.:
                * 1: Manhattan distance, i.e. sum-of-absolute-values distance.
                * 2: Euclidean distance.
                * inf: Maximum-coordinate-difference distance.
        distance_upper_bound : float, range: [0, inf), optional, default: inf
            Prune the search tree to return only neighbors within this range.
        distance_dtype : numpy.dtype, optional, default: np.single
            Data type to use for the distances.

        Returns
        -------
        distances, indices : Tuple[ndarray, ndarray], shape: (d_1, d_2, ..., d_{m-1}, k)
            Distances to, and indices of the k nearest points in self, to each point in
            `points`. Both returned arrays match the dimensions of `points` along all
            axes but last; the last axis has k elements, corresponding to the k nearest neighbors.
        """
        if np.issubdtype(type(count), np.integer):
            if count < 1:
                raise ValueError(
                    "Parameter `num_neaerst_neighbors` expects positive (nonzero) integers. "
                    f"Input argument was: {count}"
                )
            k_neighbors = tuple(range(1, count + 1))
        else:
            num_neaerst_neighbors_array = np.asarray(count)
            if not np.issubdtype(num_neaerst_neighbors_array.dtype, np.integer):
                raise ValueError("Parameter `num_neaerst_neighbors` expects integers.")
            if num_neaerst_neighbors_array.ndim != 1:
                raise ValueError("Parameter `num_neaerst_neighbors` expects 1-dimensional arrays.")
            k_neighbors = tuple(num_neaerst_neighbors_array)

        count_neighbors = len(k_neighbors)
        points_array = jnp.asarray(points)

        if per_instance:
            kdtrees = self.kdtrees[instance_slice]
            count_instances = len(kdtrees)
            shape_distances = (count_instances, *points_array.shape[:-1], count_neighbors)
            shape_indices = shape_distances + (2,)
            distances = np.empty(shape=shape_distances, dtype=distance_dtype)
            indices = np.empty(
                shape=shape_indices,
                dtype=scids.util.smallest_np_integer_dtype_for_range(
                    min_val=0, max_val=self.point_count_per_instance
                ),
            )
            for idx_instance, kdtree in enumerate(kdtrees):
                indices[idx_instance, ..., 0] = idx_instance
                # Scipy KDTree.query:
                #  returns distances and indices; distances
                distances[idx_instance], indices[idx_instance, ..., 1] = kdtree.query(
                    x=points_array,
                    k=k_neighbors,
                    eps=error_tolerance,
                    p=p_norm,
                    distance_upper_bound=distance_upper_bound,
                    workers=-1,
                )
            return distances, indices
        raise NotImplementedError

    def count_neighbors(self):
        pass

    def cluster__common_nearest_neighbor(
        self,
        radius_neighborhood: float,
        min_samples: int,
        metric: Literal = "euclidean",
        metric_params=None,
        leaf_size=30,
        p_norm: int = 2,
    ):
        """

        Parameters
        ----------
        radius_neighborhood
        min_samples
        metric
        metric_params
        leaf_size
        p_norm

        Returns
        -------

        References
        ----------
        Documentation on scikit-learn-extra, with examples:
        * https://scikit-learn-extra.readthedocs.io/en/stable/modules/cluster.html#common-nearest-neighbors-clustering
        * https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/plot_commonnn.html#sphx-glr-auto-examples-plot-commonnn-py
        * https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/cluster/plot_commonnn_data_sets.html
        Source code of the scikit-learn-extra implementation:
        * https://github.com/scikit-learn-contrib/scikit-learn-extra/tree/main/sklearn_extra/cluster
        GitHub and documentation of the independent package:
        * https://github.com/bkellerlab/CommonNNClustering
        * https://bkellerlab.github.io/CommonNNClustering
        Publications:
        * https://doi.org/10.1063/1.3301140
        * https://doi.org/10.3390/a11020019
        * https://doi.org/10.1063/1.4965440
        """

    def rmsd(self, points, weights=None):
        if weights is not None:
            weights_arr = jnp.asarray(weights)
            if weights_arr.ndim == 1:
                if weights_arr.size != self.point_count:
                    raise ValueError
                weights_arr = jnp.expand_dims(weights_arr, axis=-1)
            elif weights_arr.ndim == 2 and (
                weights_arr.shape[0] not in (1, self.point_count)
                or weights_arr.shape[1] not in (1, self.dimension)
                or weights_arr.shape == (1, 1)
            ):
                raise ValueError
            else:
                raise ValueError
        points_arr = jnp.asarray(points)
        if points_arr.ndim == 2 and points_arr.shape == self._points.shape:
            return oc.spacetime.vectorized


def from_array(points: ArrayLike) -> DynamicPointCloud:
    """Create a dynamic point cloud from an array of coordinates.

    Parameters
    ----------
    points
        An array of points in n-dimensional space.
    """
    return DynamicPointCloud(points)
