"""Clustering functions for pharmacophore features.

This module provides functions that return clustering functions
suitable for use with the `Pharmacophore.cluster` method.
"""
from typing import Sequence, Literal, Callable, Any, TypeAlias

import numpy as np
import scids
from sklearn.cluster import AgglomerativeClustering

from t2fpharm.typing import PositiveInt, PositiveFloat


ClusteringFunction: TypeAlias = Callable[[np.ndarray, np.ndarray | None], np.ndarray | Sequence[int]]
"""A compatible clustering function.

Parameters
----------
input_values
    First positional argument, corresponding to the clustering input values
    as a 2D NumPy array of floats.
    Depending on the clustering method,
    this is either an array of shape `(n_features, 3)`,
    containing the 3D coordinates of feature centers/ends,
    or an array of shape `(n_features, n_features)`,
    containing pairwise distance values between features.
weights
    Optional second positional argument, corresponding to feature weights
    as a 1D NumPy array of floats of length `n_features`.
    If provided, the clustering function may use these weights
    to influence the clustering result.
    If not provided (or None), the clustering function should treat all features equally.

Returns
-------
    A 1D NumPy array or sequence of integers of length `n_features`,
    where each integer represents the cluster label assigned to the corresponding feature.
    Negative labels are considered background/noise.
"""


ClusterAggLinkageType: TypeAlias = Literal["average", "complete", "single", "ward"]
ClusterAggLinkageMetricType: TypeAlias = Literal["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"] | Callable


def agg(
    n_clusters: PositiveInt | None,
    distance_threshold: PositiveFloat | None,
    linkage: ClusterAggLinkageType,
    metric: ClusterAggLinkageMetricType,
    memory: Any | None = None,
) -> ClusteringFunction:
    """Create a clustering function using a [hierarchical agglomerative clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) algorithm.

    For the agglomerative clustering parameters,
    see the documentation for the underlying clustering routine
    [`sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering).

    Returns
    -------
    A hierarchical agglomerative clustering function
    that can be used as input for the `Pharmacophore.cluster` method.
    """
    def clustering_function(centers: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            linkage=linkage,
            distance_threshold=distance_threshold,
        ).fit_predict(centers)
    return clustering_function


def cnn(
    max_distance: PositiveFloat | Sequence[PositiveFloat],
    min_neighbors: PositiveInt | Sequence[PositiveInt],
    max_members: PositiveInt | None = None,
) -> ClusteringFunction:
    """Create a clustering function using the Common Nearest Neighbors (CNN) algorithm.

    Parameters
    ----------
    max_distance
        Maximum distance between two feature centers
        to consider them as neighbors during clustering.
        - If a single number is provided, it applies to all (re)clustering runs.
        - If a sequence of numbers is provided,
          the i-th number in the sequence corresponds to
          the input for the i-th clustering run
          (see the `max_members` parameter below for more details).
    min_neighbors
        Minimum number of common neighbors
        between two feature centers that belong to the same cluster.
        Similar to `max_distance`, this can be a single integer or a sequence of integers.
    max_members
        Optional cap for the maximum number of members in a cluster.
        If specified, clusters with more members than this
        are reclustered into smaller clusters.
        For this, either one or both of `max_distance` and `min_neighbors`
        must be a sequence of values,
        where the i-th value corresponds to the i-th clustering step.
        In each step, clusters from the last step
        with more members than `max_members`
        are reclustered until all clusters
        have maximum `max_members` members.
        If all `max_distance` and `min_neighbors`
        values are exhausted without reaching the desired number of members,
        an error is raised.
        If only one of `max_distance` or `min_neighbors`
        is a sequence, the other one is assumed to be constant
        for all clustering steps.
        If both are sequences,
        they must have the same length,
        and the i-th value of `max_distance` and `min_neighbors`
        is used for the i-th clustering step.
        If `None`, no reclustering is performed.

    Returns
    -------
    A CNN clustering function that can be used as input for the `Pharmacophore.cluster` method.
    """
    def clustering_function(centers: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        labels = scids.pointcloud.from_array(centers).cluster_cnn(
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            min_members=1,
            max_members=max_members,
        )
        if np.any(labels < 0):
            raise RuntimeError("CNN clustering returned negative labels.")
        # Adjust labels to start from 0, with -1 indicating noise
        labels -= 1
        return labels
    return clustering_function
