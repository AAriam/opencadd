"""Point cloud dataset."""

import functools

import arrayer
import bbo
import jax.numpy as jnp
import numpy as np
import scipy as sp

from commonnn import cluster as cnn_cluster
import scids
from scids import dataset, exception
from scids.volume import AxisAlignedRectangularCuboid
from scids.grid import Grid
from scids.field import Field
from scids.typing import (
    atypecheck, Array, JAXArray, Num, Shaped, Is,
    Annotated, PositiveInt, PositiveFloat, PositiveInts1D, NonNegativeFloat, PathLike
)
from scids.protocol import CNNClusteringConfig

from collections.abc import Sequence
from typing import Literal, Any, Self
from numpy.typing import ArrayLike, DTypeLike


class PointCloud(dataset.DataSet):
    """A discrete set of data points in n-dimensional space.

    This represents a set of features observed for
    each sample in a population at one or several instances.
    For example, instances may be consecutive timepoints,
    the samples may be a set of particles,
    and the observed feature may be the position of particles in space,
    in which case the point cloud represents a trajectory.

    Parameters
    ----------
    points
        Point cloud(s) as a real or complex-valued array of
        shape `(*batch_shape, point_count, point_dim)`,
        where `*batch_shape` is zero or more batch axes,
        holding different instances of a point cloud
        with `point_count` samples each with `point_dim` features.
    batch
        Information about batch axes.
        If provided, this must be a sequence
        with the same length as the number of batch axes.
        Each element of the sequence can be:
        - A string representing the label of the axis.
        - A 2-tuple, where the first element is a string
          representing the label of the axis,
          and the second element is a sequence of strings
          representing the labels for each instance along that axis.
    """

    @atypecheck
    def __init__(
        self,
        points: Num[JAXArray, "*batch_shape point_count point_dim"],
        batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
    ):
        super().__init__(data=points, batch=batch or points.ndim - 2)
        self._points_2d: jnp.ndarray = None
        self._kdtrees_per_instance: list[sp.spatial.KDTree] = None
        self._kdtree_combined: sp.spatial.KDTree = None
        return

    @property
    def points(self) -> Num[JAXArray, "*self_batch_shape {self.point_count} {self.point_dim}"]:
        """Coordinates of all points in all point clouds.

        This array has the same shape as the input data
        """
        return self._data

    @property
    def points_2d(self) -> Num[JAXArray, "{self.point_count_total} {self.point_dim}"]:
        """Coordinates of all points in all point clouds.

        This array contains the same data as `self.points`,
        but reshaped to a 2D array,
        i.e., all batch dimensions are collapsed.
        When the point cloud has no batch dimensions,
        this is equivalent to `self.points`.
        """
        if self._points_2d is None:
            self._points_2d = self._data.reshape(-1, self._data.shape[-1])
        return self._points_2d

    @property
    def point_count(self) -> int:
        """Number of points in each point cloud."""
        return self._data.shape[-2]

    @property
    def point_count_total(self) -> int:
        """Total number of points in all point clouds."""
        return np.prod(self._data.shape[:-1])

    @property
    def point_dim(self) -> int:
        """Dimension of the points in the point cloud."""
        return self._data.shape[-1]

    @property
    def kdtree(self) -> sp.spatial.KDTree:
        """Combined KDTree of all points in all point clouds."""
        if self._kdtree_combined is None:
            self._kdtree_combined = sp.spatial.KDTree(self.points_2d)
        return self._kdtree_combined

    @property
    def kdtrees(self) -> Num[Array, "*self_batch_shape"] | None:
        """KDTree of each point cloud.

        This is an array of `KDTree` objects, one for each point cloud.
        The shape of the array is the same as the batch dimensions of the point cloud.
        If the point cloud has no batch dimensions,
        this is `None`.
        """
        if self.batch_ndim == 0:
            return None
        if self._kdtrees_per_instance is None:
            self._kdtrees_per_instance = np.empty(shape=self.batch_shape, dtype=object)
            for idx in np.ndindex(*self.batch_shape):
                self._kdtrees_per_instance[idx] = sp.spatial.KDTree(data=self._data[idx])
        return self._kdtrees_per_instance.copy()

    def aabb(self, per_instance: bool = True) -> AxisAlignedRectangularCuboid:
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
        axis = -2 if per_instance else tuple(range(self.batch_ndim + 1))
        mins = jnp.min(self._data, axis=axis)
        maxes = jnp.max(self._data, axis=axis)
        return scids.volume.AxisAlignedRectangularCuboid(lower_bounds=mins, upper_bounds=maxes)

    def toxelate(
        self,
        grid: float | Sequence[float] | Grid,
        point_radii: float | ArrayLike,
        padding: float | None = None,
        instance_selection: Any = None,
        error_tolerance: NonNegativeFloat = 0,
    ) -> Field:
        """Create a voxel representation of the point cloud.

        Parameters
        ----------
        grid
            Grid to use for the voxel volume.
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
            Padding to add to the grid bounds.
            If `None`, the padding is set to the maximum radius
            of the points in the point cloud plus one spacing,
            guaranteeing that all outermost toxels are unoccupied.
        instance_selection
            Any array indexing object to select a subset of instances.
            By default, all instances are considered.
            This only has an effect when `self.batch_ndim > 0`.
        error_tolerance
            Error tolerance for the distance calculations.
            This is only used when all points have the same radius,
            otherwise it is ignored.

        Returns
        -------
        An integer field representing the volume voxels,
        where 0 indicates an unoccupied voxel.
        Occupied voxels are represented by positive integers
        that correspond to the 1-based index of the point
        in the point cloud that occupies the voxel,
        i.e., the first point in the point cloud is represented by 1.
        """
        single_radius = any(np.issubdtype(type(point_radii), scalar_type) for scalar_type in (np.integer, np.floating))
        if single_radius:
            if point_radii <= 0:
                raise exception.InputError(
                    name="point_radii",
                    message=f"A positive real number is expected, but got {point_radii}."
                )
        else:
            point_radii = np.asarray(point_radii)
            if point_radii.ndim != 1 or point_radii.shape[0] != self.point_count:
                raise exception.InputError(
                    name="point_radii",
                    message=f"An array of shape ({self.point_count},) is expected, "
                            f"but got {point_radii.shape}."
                )
            if np.any(point_radii <= 0):
                raise exception.InputError(
                    name="point_radii",
                    message=f"All radii must be positive, but got {point_radii}."
                )
        max_radius = point_radii if single_radius else point_radii.max()
        if not isinstance(grid, Grid):
            if padding is None:
                spacing = grid if isinstance(grid, (int, float)) else np.asarray(grid).max()
                padding = max_radius + spacing  # Guarantee that all outermost toxels are unoccupied
            # Get the bounding box of all instances superposed.
            total_bounding_box = self.aabb(per_instance=False)
            # Create a grid the size of the total bounding box, with given resolution
            grid = scids.grid.from_bounds_spacing(
                lower=total_bounding_box.lower_bounds - padding,
                upper=total_bounding_box.upper_bounds + padding,
                spacing=grid,
            )
        # If `point_radii` is a scalar (i.e. int or float), it means all points have the same
        # radius, and thus we only need to query for the first nearest neighbor of each point:
        if single_radius:
            dists, indices = self.nearest_neighbors(
                points=grid.coordinates,
                count=1,
                per_instance=True,
                instance_selection=instance_selection,
                error_tolerance=error_tolerance,
                distance_upper_bound=point_radii,
            )
            # Each toxel on the grid is occupied when the nearest point in self is within
            # `point_radii`, i.e. when it is not `np.inf`, since `self.nearest_neighbors` returns
            # `np.inf` for points where the nearest distance is larger than the upper bound.
            return scids.field.from_tensor(
                grid=grid,
                tensor=np.where(dists != np.inf, indices + 1, 0),  # +1 to avoid 0 index
                batch=self.batch_ndim,
            )
        # If `point_radii` is an array of values, then we cannot rely only on the distances to
        # first nearest neighbors, since it is possible that the first k nearest neighbors have
        # small radii and do not overlap with the toxel, while the (k+1)-th neighbor has a large
        # enough radius to overlap.
        toxel_tensor = np.zeros(shape=(*self.batch_shape, *grid.shape), dtype=np.uint64)
        dists, (*ind_batch, ind_point), ind_grid = self.distance_matrix_sparse(
            points=grid.coordinates, max_distance=max_radius
        )
        definitely_occupied = dists <= point_radii.min()
        toxel_tensor[
            tuple(axis_ind[definitely_occupied] for ind in (ind_batch, ind_grid) for axis_ind in ind)
        ] = ind_point[definitely_occupied] + 1  # +1 to avoid 0 index
        maybe_occupied = jnp.logical_not(definitely_occupied)
        dists_from_surface = (
            dists[maybe_occupied] - point_radii[ind_point[maybe_occupied]]
        )
        occupied = dists_from_surface <= 0
        toxel_tensor[
            tuple(axis_ind[maybe_occupied][occupied] for ind in (ind_batch, ind_grid) for axis_ind in ind)
        ] = ind_point[maybe_occupied][occupied] + 1  # +1 to avoid 0 index
        return scids.field.from_tensor(
            tensor=toxel_tensor,
            grid=grid,
            batch=self.batch_ndim,
        )

    def distance_matrix_sparse(
        self,
        points: Self | sp.spatial.KDTree | Num[Array, "*batch_shape {self.point_dim}"],
        max_distance: PositiveFloat,
        p_norm: PositiveInt = 2,
        output_type: Literal[
            "nd_unraveled", "dok_matrix", "coo_matrix", "dict", "ndarray"
        ] = "nd_unraveled",
    ) -> tuple[
        np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]
    ] | sp.sparse._dok.dok_matrix | sp.sparse._coo.coo_matrix | dict | np.ndarray:
        """Calculate a sparse distance matrix between the points in self and the points in `points`.

        Parameters
        ----------
        points
            Points to calculate the distance to.
            This can be a `PointCloud` object, a `scipy.spatial.KDTree` object,
            or an array of shape `(*batch_shape, self.point_dim)`
            where `*batch_shape` is zero or more batch axes.
        max_distance
            Maximum distance to consider.
            Distances larger than this are
            not included in the output.
        p_norm
            The Minkowski p-norm to use, e.g.:
            - 1: Manhattan distance, i.e. sum-of-absolute-values distance.
            - 2: Euclidean distance.
            - inf: Maximum-coordinate-difference distance.
        output_type
            Type of output to return.
            - "nd_unraveled": A 3-tuple of arrays:
              1. Distances between the points in self and the points in `points`.
              2. Indices of the points in self.
              3. Indices of the points in `points`.
              Both indices are in the form of a tuple of arrays,
              where each array contains the indices of the points
              in the corresponding dimension.
              These can be directly used to index into
              the points in self and `points`.
            - "dok_matrix": A sparse matrix in DOK format.
            - "coo_matrix": A sparse matrix in COO format.
            - "dict": A dictionary where keys are 2-tuples
              of indices in the form (i, j),
              and values are the distances between the points.
              Note that the indices are not unraveled here,
              i.e. they correspond to the indices of the corresponding
              2D arrays of points.
            - "ndarray": A NumPy record array with fields "i", "j", and "v",
              where "i" and "j" are the indices of the points in self and `points`,
              and "v" is the distance between them.
              Again, the indices are not unraveled here.

        See Also
        --------
        - [`scipy.spatial.KDTree.sparse_distance_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.sparse_distance_matrix.html):
          The underlying function used to calculate the distance matrix.
        """
        if isinstance(points, PointCloud):
            points_shape = points.points.shape
            kdtree = points.kdtree
        elif isinstance(points, sp.spatial.KDTree):
            points_shape = points.data.shape
            kdtree = points
        else:
            points = np.asarray(points)
            points_shape = points.shape
            points = points.reshape(-1, points_shape[-1])
            kdtree = sp.spatial.KDTree(points)
        if points_shape[-1] != self.point_dim:
            raise exception.InputError(
                name="points",
                message=f"Points must have the same number of elements along the last axis as self, "
                        f"but got {points_shape[-1]} instead of {self.point_dim}."
            )
        dist_matrix = self.kdtree.sparse_distance_matrix(
            other=kdtree,
            max_distance=max_distance,
            p=p_norm,
            output_type="ndarray" if output_type == "nd_unraveled" else output_type,
        )
        if output_type != "nd_unraveled":
            return dist_matrix
        indices_self = np.unravel_index(dist_matrix["i"], shape=self.points.shape[:-1])
        indices_other = np.unravel_index(dist_matrix["j"], shape=points_shape[:-1])
        return dist_matrix["v"], indices_self, indices_other

    def _distance_matrix_full(
        self,
        points: Self | sp.spatial.KDTree | Num[Array, "*batch_shape {self.point_dim}"],
        p_norm: float = 2,
        threshold: PositiveInt = 1e7,
        instance_selection: Any = None,
    ):
        """Calculate the full distance matrix between the points in self and the points in `points`.

        Parameters
        ----------
        points
            Points to calculate the distance to.
            This can be a `PointCloud` object, a `scipy.spatial.KDTree` object,
            or an array of shape `(*batch_shape, self.point_dim)`
            where `*batch_shape` is zero or more batch axes.
        p_norm
            The Minkowski p-norm to use, e.g.:
            - 1: Manhattan distance, i.e. sum-of-absolute-values distance.
            - 2: Euclidean distance.
            - inf: Maximum-coordinate-difference distance.
        threshold
            Maximum number of points to calculate the distance to.
            If the total number of point coordinates is larger than this,
            the distance matrix is calculated in chunks.
            This is useful to avoid memory issues when calculating
            the distance matrix for large point clouds.
        instance_selection
            Any array indexing object to select a subset of instances.
            By default, all instances are considered.
            This only has an effect when `self.batch_ndim > 0`.
        Returns
        -------
        For `points` with shape `(*batch_shape, self.point_dim)`,
        distances are returned as an array of shape `(*instance_axes, *batch_shape)`,
        where `*instance_axes` are the batch dimensions of self
        as specified by `instance_selection`,
        or `self.batch_shape` if `instance_selection` is None.
        """
        if isinstance(points, PointCloud):
            points = points.points_2d
            points_shape = points.points.shape
        elif isinstance(points, sp.spatial.KDTree):
            points = points.data
            points_shape = points.data.shape
        else:
            points = np.asarray(points)
            points_shape = points.shape
            points = points.reshape(-1, points_shape[-1])
        if points_shape[-1] != self.point_dim:
            raise exception.InputError(
                name="points",
                message=f"Points must have the same number of elements along the last axis as self, "
                        f"but got {points_shape[-1]} instead of {self.point_dim}."
            )
        if instance_selection is None:
            self_points = self.points_2d
            self_points_shape = self.points.shape
        else:
            if self.batch_ndim == 0:
                raise exception.InputError(
                    name="instance_selection",
                    message="Parameter is not applicable when there are no batch dimensions."
                )
            self_points = self.points[instance_selection]
            self_points_shape = self_points.shape
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html#scipy.spatial.distance_matrix
        dists = sp.spatial.distance_matrix(
            x=self_points,
            y=points,
            p=p_norm,
            threshold=threshold,
        )
        return dists.reshape(*self_points_shape[:-1], *points_shape[:-1])

    def minimize_aabb(
        self,
        instance_selection: Any = None,
        mode: Literal["per_instance", "one_for_all", "one_for_slice"] = "per_instance",
        algorithm: Literal["pca", "hull", "best"] = "best",
    ) -> "PointCloud":
        """Minimize the [axis-aligned minimum bounding box](https://en.wikipedia.org/wiki/Minimum_bounding_box#Axis-aligned_minimum_bounding_box) volume of the point cloud.

        This is done by rotating the point cloud
        so that its minimum-volume oriented bounding box
        is aligned with the coordinate axes.

        Parameters
        ----------
        instance_selection
            Slice of instances to consider.
            By default, all instances are considered.
            This is only applicable when the point cloud
            has shape `(n_instances, n_samples, n_features)`.
        mode
            Mode of application (only applicable when the point cloud
            has shape `(n_instances, n_samples, n_features)`.):
            - "per_instance": Minimize the bounding box for each instance separately.
            - "one_for_slice": Minimize the bounding box for all instances superposed,
              and apply the same rotation to all selected instances.
            - "one_for_all": Minimize the bounding box for all instances superposed,
              and apply the same rotation to all instances (not just the selected ones).
        algorithm
            Algorithm to use for finding the rotation.
            - "pca": Principal Component Analysis (PCA).
              This works for any number of dimensions.
              However, it is a is not guaranteed to find the optimal rotation,
              but it is usually a good approximation.
            - "hull": Convex hull-based brute-force search.
              This is guaranteed to find the optimal rotation for 2D points,
              but is an approximation for higher dimensions.
            - "best": For 2D points, this is the same as "hull".
              For 3D points, this runs both "pca" and "hull",
              and returns the one with the smallest volume.
        """
        if mode not in ("per_instance", "one_for_all", "one_for_slice"):
            raise exception.InputError(
                name="mode",
                message="The `mode` parameter must be one of 'per_instance', 'one_for_all', or 'one_for_slice'."
            )
        if self.batch_ndim == 0:
            if instance_selection is not None:
                raise exception.InputError(
                    name="instance",
                    message="The `instance` parameter is not applicable when there are no prefix dimensions."
                )
            bbout = bbo.run(points=self.points, method=algorithm)
            return PointCloud(points=bbout.points)
        if instance_selection is None:
            instance_selection = slice(None)
        instances = self._select_instances(instance_selection)
        if mode == "per_instance":
            bbo_output = bbo.run(points=instances, method=algorithm)
            new_points = self._data.at[instance_selection].set(bbo_output.points)
            return PointCloud(new_points)
        combined_points = instances.reshape(-1, self.point_dim)
        bbo_output = bbo.run(points=combined_points, method=algorithm)
        if mode == "one_for_all":
            new_points = self._data @ bbo_output.rotation
            return PointCloud(new_points)
        new_points = self._data.at[instance_selection].set(bbo_output.points.reshape(instances.shape))
        return PointCloud(new_points)

    @atypecheck
    def nearest_neighbors(
        self,
        points: Num[Array, "*batch_shape {self.point_dim}"],
        count: PositiveInt | PositiveInts1D = 1,
        per_instance: bool = True,
        instance_selection: Any = None,
        error_tolerance: NonNegativeFloat = 0,
        p_norm: PositiveInt = 2,
        distance_upper_bound: NonNegativeFloat = np.inf,
        distance_dtype: DTypeLike = np.float64,
    ) -> tuple[Num[Array, "..."], Num[Array, "..."]]:
        """Find the nearest points in self to a given set of points.

        For each point in `points`, this finds the distances to, and indices of,
        a given number of nearest points in self.

        Parameters
        ----------
        points
            Coordinates of points for which the nearest points in self must be found.
            The last axis must have the same size as `self.point_dim`.
            Other than that, the shape of `points` can be arbitrary,
            any number of leading batch dimensions are allowed.
        count
            Either the number of nearest neighbors (as an integer),
            or a sequence of the k-th (k >= 1) nearest neighbors to find.
        per_instance
            Whether to calculate the nearest neighbors for each instance separately,
            or for all instances combined.
        instance_selection
            Any array indexing object to select a subset of instances.
            By default, all instances are considered.
            This only has an effect when `per_instance` is True.
        error_tolerance
            Tolerance for error in finding the nearest atoms.
            The k-th nearest atom will be within (1 + eps) times
            the distance to the real k-th nearest atom.
        p_norm
            The Minkowski p-norm to use, e.g.:
            - 1: Manhattan distance, i.e. sum-of-absolute-values distance.
            - 2: Euclidean distance.
            - inf: Maximum-coordinate-difference distance.
        distance_upper_bound
            Prune the search tree to return only neighbors within this range.
        distance_dtype
            Data type to use for the distances.

        Returns
        -------
        A 2-tuple of arrays:
        1. Distances to the k nearest points in self, for each point in `points`.
        2. Indices of the k nearest points in self, for each point in `points`.

        For `points` with shape `(*batch_shape, self.point_dim)`
        both arrays have shape `(*instance_axes, *batch_shape, k)`,
        where `*instance_axes` are the batch dimensions of self
        as specified by `instance_selection`,
        or `self.batch_shape` if `instance_selection` is None.
        If `count == 1`, the last dimension (i.e., `k`) is omitted.
        """
        if self.batch_ndim == 0 or not per_instance:
            distances, self_indices = self.kdtree.query(
                x=points,
                k=count,
                eps=error_tolerance,
                p=p_norm,
                distance_upper_bound=distance_upper_bound,
                workers=-1,
            )
            return distances, self_indices
        if isinstance(count, int):
            k = () if count == 1 else (count,)
        else:
            k = (len(count),)
        kdtrees = self.kdtrees if instance_selection is None else self.kdtrees[instance_selection]
        output_shape = (*kdtrees.shape, *points.shape[:-1], *k)
        distances = np.empty(shape=output_shape, dtype=distance_dtype)
        indices = np.empty(
            shape=output_shape,
            dtype=arrayer.dtype.smallest_integer(minimum=0, maximum=self.point_count + 1),  # +1 since `self.toxelate` adds 1 to indices to avoid 0 index in toxel tensor
        )
        for instance_idx in np.ndindex(kdtrees.shape):
            distances[instance_idx], indices[instance_idx] = kdtrees[instance_idx].query(
                x=points,
                k=count,
                eps=error_tolerance,
                p=p_norm,
                distance_upper_bound=distance_upper_bound,
                workers=-1,
            )
        return distances, indices

    def cluster_cnn(
        self,
        max_distance: PositiveFloat | Sequence[PositiveFloat],
        min_neighbors: PositiveInt | Sequence[PositiveInt],
        min_members: PositiveInt = 1,
        max_members: PositiveInt | None = None,
        instance_selection: Any = (),
    ) -> Num[Array, "*batch_selection_shape {self.point_count}"]:
        """Cluster points using the Common Nearest Neighbors (CNN) algorithm.

        If multiple instances are selected,
        the clustering is performed separately for each instance.

        Parameters
        ----------
        max_distance
            Maximum distance between two points to consider them as neighbors.
        min_neighbors
            Minimum number of common neighbors between two points
            that belong to the same cluster.
        min_members
            Minimum number of members in a cluster.
            Cluster with fewer members than this are discarded.
        max_members
            Maximum number of members in a cluster.
            If specified, clusters with more members than this
            are reclustered into smaller clusters.
            For this, either one or both of `max_distance` and `min_neighbors`
            must be a sequence of values, where the i-th value
            corresponds to the i-th clustering step.
            In each step, clusters from the last step
            with more members than `max_members`
            are reclustered until either all clusters
            have maximum `max_members` members,
            or the end of the sequence is reached.
            If only one of `max_distance` or `min_neighbors`
            is a sequence, the other one is assumed to be constant
            for all clustering steps.
            If both are sequences,
            they must have the same length,
            and the i-th value of `max_distance` and `min_neighbors`
            is used for the i-th clustering step.
        instance_selection
            Any array indexing object to select a subset of instances.
            By default, all instances are considered.

        Returns
        -------
        Cluster Label for each point as an integer array of shape `(*batch_selection_shape, self.point_count)`.

        References
        ----------
        Documentation on scikit-learn-extra, with examples:
        - https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/plot_commonnn.html#sphx-glr-auto-examples-plot-commonnn-py
        - https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/cluster/plot_commonnn_data_sets.html
        - https://scikit-learn-extra.readthedocs.io/en/stable/modules/cluster.html#common-nearest-neighbors-clustering

        Source code of the scikit-learn-extra implementation:
        - https://github.com/scikit-learn-contrib/scikit-learn-extra/tree/main/sklearn_extra/cluster

        GitHub and documentation of the independent package:
        - https://github.com/bkellerlab/CommonNNClustering
        - https://bkellerlab.github.io/CommonNNClustering

        Publications:
        - https://doi.org/10.1063/1.3301140
        - https://doi.org/10.3390/a11020019
        - https://doi.org/10.1063/1.4965440
        """
        def cluster_with_max_members(points: Array) -> Array:
            final_clusters: list[Array] = []
            # each task is (indices_to_cluster, which_param_index)
            tasks: list[tuple[Array, int]] = [(np.arange(n_points, dtype=int), 0)]
            while tasks:
                indices, idx = tasks.pop(0)
                labels = cluster(
                    points=points[indices],
                    max_dist=clustering_inputs.max_distance[idx],
                    min_neigh=clustering_inputs.min_neighbors[idx]
                )
                for label in np.unique(labels):
                    if label == 0:
                        # noise stays noise
                        continue
                    mask = labels == label
                    member_idxs = indices[mask]
                    if member_idxs.size <= clustering_inputs.max_members:
                        final_clusters.append(member_idxs)
                    else:
                        next_idx = idx + 1
                        if next_idx >= len(clustering_inputs.max_distance):
                            raise exception.InputError(
                                name="max_members",
                                message=f"Cluster with {member_idxs.size} members is larger than "
                                        f"`max_members` ({clustering_inputs.max_members}), "
                                        f"but no more clustering steps are available."
                            )
                        tasks.append((member_idxs, next_idx))
            # sort final clusters by size descending and assign final labels
            final_clusters.sort(key=lambda arr: arr.size, reverse=True)
            final_labels = np.zeros(n_points, dtype=int)
            for new_label, member_idxs in enumerate(final_clusters, start=1):
                final_labels[member_idxs] = new_label
            return final_labels

        def cluster(points, max_dist, min_neigh):
            clustering = cnn_cluster.Clustering(points)
            clustering.fit(
                sort_by_size=True,
                member_cutoff=clustering_inputs.min_members,
                radius_cutoff=max_dist,
                similarity_cutoff=min_neigh,
                v=False
            )
            return clustering.labels

        clustering_inputs = CNNClusteringConfig(
            **{
                k: v for k, v in locals().items()
                if k in ("max_distance", "min_neighbors", "min_members", "max_members")
            }
        )
        n_points = self.point_count
        instances = self._select_instances(instance_selection)
        if instances.ndim == 2:
            instances = instances.reshape(1, *instances.shape)
            single_instance = True
        else:
            single_instance = False
        all_labels = np.empty(shape=instances.shape[:-1], dtype=np.int64)
        clustering_func = functools.partial(
            cluster,
            max_dist=clustering_inputs.max_distance[0],
            min_neigh=clustering_inputs.min_neighbors[0]
        ) if clustering_inputs.max_members is None else cluster_with_max_members
        for instance_index in np.ndindex(instances.shape[:-2]):
            all_labels[instance_index] = clustering_func(instances[instance_index])
        if single_instance:
            all_labels = all_labels.reshape(-1)
        return all_labels

    def to_npz(self, filepath: PathLike | None = None, compress: bool = False) -> dict[str, Any]:
        """Save the point cloud to a .npz file."""
        return super()._to_npz(filepath=filepath, data_key="points", compress=compress)

    def _select_instances(self, instance_selection: Any, param_name: str = "instance_selection"):
        instances = self._data[instance_selection]
        if instances.ndim < 2:
            raise exception.InputError(
                name=param_name,
                message=f"Selection must yield at least a 2D array, "
                        f"but got {instances.ndim}D array with shape {instances.shape}."
            )
        if instances.shape[-2:] != self.points.shape[-2:]:
            raise exception.InputError(
                name=param_name,
                message=f"Selection must yield an array with the same shape as self along the last two axes, "
                        f"but got {instances.shape[-2:]} instead of {self.points.shape[-2:]}."
            )
        return instances

    def _rmsd(self, points, weights=None):
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
        if points_arr.ndim == 2 and points_arr.shape == self._data.shape:
            return oc.spacetime.vectorized


def from_array(
    points: ArrayLike,
    dtype: DTypeLike,
    batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
) -> PointCloud:
    """Create a point cloud from an array of values.

    Parameters
    ----------
    points
        Point cloud(s) as a real or complex-valued array of
        shape `(*batch_shape, point_count, point_dim)`,
        where `*batch_shape` is zero or more batch axes,
        holding different instances of a point cloud
        with `point_count` samples each with `point_dim` features.
    batch
        Information about batch axes.
        If provided, this must be a sequence
        with the same length as the number of batch axes.
        Each element of the sequence can be:
        - A string representing the label of the axis.
        - A 2-tuple, where the first element is a string
          representing the label of the axis,
          and the second element is a sequence of strings
          representing the labels for each instance along that axis.
    """
    return PointCloud(points=jnp.asarray(points, dtype=dtype), batch=batch)


def from_npz(filepath: PathLike) -> PointCloud:
    """Load a point cloud from a .npz file.

    Parameters
    ----------
    filepath
        Path to the .npz file containing the point cloud data.
    """
    data = dataset.from_npz(filepath=filepath, data_key="points", return_dict=True)
    return PointCloud(points=data["points"], batch=data["batch"])
