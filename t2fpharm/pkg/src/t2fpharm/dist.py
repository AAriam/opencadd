"""Distance matrix calculation functions for pharmacophore features.

This module provides functions that return distance matrix calculation functions
suitable for use with the `Pharmacophore.cluster` and `Pharmacophore.match` methods.
"""
from typing import Callable, Final, TypeAlias

import numpy as np
import pandas as pd
import scipy.spatial


DistanceMatrixFunction: TypeAlias = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
"""A compatible distance matrix function.

Parameters
----------
`f0`, `f1`
    Two pandas DataFrames of pharmacophore features,
    each containing the following columns:
    - `repr`: Integer specifying the feature representation:
        - 1: Point feature defined by `center` only.
        - 2: Vector feature defined by `center` and `end`.
        - 3: Radial feature defined by `end` and `radius`.
    - `radius`: A non-negative real number
        representing the radius for radial features,
        or the length of the vector for vector features.
        For point features, this column should be `NaN`.
    - `center`: NumPy array representing the 3D coordinates
        of the feature's center in some reference frame.
        For radial features, this column should be `None`.
    - `end`: NumPy array representing the 3D coordinates
        of the feature's end point in some reference frame.
        For point features, this column should be `None`.

Returns
-------
A 3D NumPy array `distance_matrix` of shape `(len(f0), len(f1), 5)`
with pairwise distance metrics between features in `f0` and `f1`.
The element `distance_matrix[i, j]` contains the distance metrics
between feature `f0.iloc[i]` and `f1.iloc[j]` as follows:
- `distance_matrix[i, j, 0]`: Distance between feature centers.
- `distance_matrix[i, j, 1]`: Distance between feature ends.
- `distance_matrix[i, j, 2]`: Radius difference.
- `distance_matrix[i, j, 3]`: Angle between vectors (in radians).
- `distance_matrix[i, j, 4]`: A single combined distance value, e.g., a linear combination of the above.
"""


def linear(
    w_center: float = 1.0,
    w_end: float = 1.0,
    w_radius: float = 1.0,
    w_angle: float = 1.0,
    p_end_point_point: float = 0.0,
    p_end_point_vector: float = 0.0,
    p_end_point_radial: float = 0.0,
    p_radius_point_point: float = 0.0,
    p_radius_point_vector: float = 0.0,
    p_radius_point_radial: float = 0.0,
    p_angle_point_point: float = 0.0,
    p_angle_point_vector: float = 0.0,
    p_angle_point_radial: float = 0.0,
    p_angle_radial_radial: float = 0.0,
    n_decimals: int = 5,
) -> DistanceMatrixFunction:
    """Create a distance matrix function using a linear combination of distance metrics.

    This function computes the four distance matrices
    between features in `f0` and `f1`, plus their weighted sum as follows:
    1. Pairwise distances between feature centers:
        - For point–point, point–vector, and vector–vector pairs,
            this is the distance between their `center` coordinates.
        - For point–radial and vector–radial pairs, this is the distance
            between the point/vector `center` and the surface of the radial feature.
        - For radial–radial pairs, this is the minimum distance between the surfaces
            of the two radial features.
    2. Pairwise distances between feature ends:
        - For vector–vector, vector–radial, and radial–radial pairs,
          this is the distance between their `end` coordinates.
        - For point–x pairs, the distance is set
          to the corresponding `p_end_*` parameter value,
          since points do not have an `end`.
    3. Pairwise differences between feature radii:
        - For vector–vector, vector–radial, and radial–radial pairs,
          this is the absolute difference between their `radius` values.
        - For point–x pairs, the difference is set
          to the corresponding `p_radius_*` parameter value,
          since points do not have a `radius`.
    4. Pairwise angles between feature vectors:
        - For vector–vector pairs, this is the angle in radians (in [0, π] range)
          between their unit vectors.
        - For vector–radial pairs, this is the angle in radians (in [0, π] range)
          between the vector's unit vector and the unit vector
          from the vector's center to the radial feature's end.
        - For all other pairs, the angle is set
          to the corresponding `p_angle_*` parameter value.
    5. Linear combination of the above four distance matrices, using the specified weights.

    Parameters
    ----------
    f0
        First DataFrame of features.
    f1
        Second DataFrame of features. This can be the same as `f0` for symmetric distance matrices.
    p_end_*
        Fixed end distance penalties for pairs for which end distance is undefined.
    p_radius_*
        Fixed radius difference penalties for pairs for which radius difference is undefined.
    p_angle_*
        Fixed angle (in radians) penalties for pairs for which angle is undefined.
    w_*
        Weights for each distance component when computing the weighted sum.
    n_decimals
        Number of decimal places to round the output distances to.

    Returns
    -------
    A distance matrix function that can be used as input for the `Pharmacophore.cluster`
    and `Pharmacophore.match` methods.

    Notes
    -----
    - Distances are symmetric by construction where applicable; parameter-based
      penalties for pairs involving a point are applied symmetrically.
    - For zero-length vectors (numerical degeneracy), the vector angle is set to π.
    """
    def distance_matrix_function(f0: pd.DataFrame, f1: pd.DataFrame) -> np.ndarray:
        # ---- extract arrays ------------------------------------------------------
        n0: Final[int] = len(f0)
        n1: Final[int] = len(f1)

        repr0 = f0["repr"].to_numpy(dtype=int)
        repr1 = f1["repr"].to_numpy(dtype=int)

        # Masks for feature types and combinations
        mask_p0 = repr0 == 1
        mask_v0 = repr0 == 2
        mask_r0 = repr0 == 3

        mask_np0 = ~mask_p0
        mask_nr0 = ~mask_r0

        mask_p1 = repr1 == 1
        mask_v1 = repr1 == 2
        mask_r1 = repr1 == 3

        mask_np1 = ~mask_p1
        mask_nr1 = ~mask_r1

        mask_pp = np.outer(mask_p0, mask_p1)
        mask_pv = np.outer(mask_p0, mask_v1)
        mask_pr = np.outer(mask_p0, mask_r1)

        mask_vp = np.outer(mask_v0, mask_p1)
        mask_vv = np.outer(mask_v0, mask_v1)
        mask_vr = np.outer(mask_v0, mask_r1)

        mask_rp = np.outer(mask_r0, mask_p1)
        mask_rv = np.outer(mask_r0, mask_v1)
        mask_rr = np.outer(mask_r0, mask_r1)

        mask_pv_or_vp = mask_pv | mask_vp
        mask_pr_or_rp = mask_pr | mask_rp

        mask_np_np = np.outer(mask_np0, mask_np1)
        mask_nr_nr = np.outer(mask_nr0, mask_nr1)
        mask_nr_r = np.outer(mask_nr0, mask_r1)
        mask_r_nr = np.outer(mask_r0, mask_nr1)
        mask_rr = np.outer(mask_r0, mask_r1)

        # Centers and ends as (n, 3) arrays (NaNs where not applicable)
        c0 = np.vstack([_as_xyz(x) for x in f0["center"].to_numpy(object)])
        c1 = np.vstack([_as_xyz(x) for x in f1["center"].to_numpy(object)])
        e0 = np.vstack([_as_xyz(x) for x in f0["end"].to_numpy(object)])
        e1 = np.vstack([_as_xyz(x) for x in f1["end"].to_numpy(object)])

        vec0_len = _norm(e0 - c0)
        vec1_len = _norm(e1 - c1)

        # Radii: for vectors, prefer DataFrame 'radius' if finite; otherwise derive from length.
        r0 = f0["radius"].to_numpy(float)
        r1 = f1["radius"].to_numpy(float)
        # Fill vector radii if missing
        vec0_no_rad_mask = np.where(mask_v0 & ~np.isfinite(r0))
        vec1_no_rad_mask = np.where(mask_v1 & ~np.isfinite(r1))
        r0[vec0_no_rad_mask] = vec0_len[vec0_no_rad_mask]
        r1[vec1_no_rad_mask] = vec1_len[vec1_no_rad_mask]

        # ---- initialize output ---------------------------------------------------
        distance_matrices = np.zeros((n0, n1, 5), dtype=float)

        # ===================== (0) CENTER DISTANCE ================================

        # Start with zeros and fill by case
        center_dist = np.zeros((n0, n1), dtype=float)

        # Case A: pairs where neither is radial -> use center-center distance
        if mask_nr_nr.any():
            center_dist[mask_nr_nr] = _distance_matrix(c0[mask_nr0], c1[mask_nr1]).ravel()

        # Case B: point/vector (has center) vs radial -> |‖c - e_r‖ - r|
        # f0 non-radial vs f1 radial
        if mask_nr_r.any():
            dist_c_to_er = _norm(c0[:, None, :] - e1[None, :, :], axis=2)  # (n0,n1)
            offset = dist_c_to_er - r1[None, :]  # broadcast across rows
            center_dist[mask_nr_r] = np.abs(offset)[mask_nr_r]
        # f0 radial vs f1 non-radial
        if mask_r_nr.any():
            dist_c_to_er = _norm(c1[None, :, :] - e0[:, None, :], axis=2)  # (n0,n1)
            offset = dist_c_to_er - r0[:, None]  # broadcast across cols
            center_dist[mask_r_nr] = np.abs(offset)[mask_r_nr]

        # Case C: radial–radial -> closest separation between spherical surfaces
        if mask_rr.any():
            ee_diff = e0[:, None, :] - e1[None, :, :]
            d = _norm(ee_diff, axis=2)
            surf = _sphere_surface_distance(d, r0[:, None], r1[None, :])
            center_dist[mask_rr] = surf[mask_rr]


        # ====================== (1) END DISTANCE =================================
        end_dist = np.zeros((n0, n1), dtype=float)

        # vector/vector, vector/radial, radial/radial -> Euclidean between ends
        if mask_np_np.any():
            end_dist[mask_np_np] = _distance_matrix(e0[mask_np0], e1[mask_np1]).ravel()


        # ====================== (2) RADIUS DIFFERENCE =============================
        rad_diff = np.zeros((n0, n1), dtype=float)

        # Non-point pairs (vector/vector, vector/radial, radial/radial): |r0 - r1|
        if mask_np_np.any():
            r0_mat = r0[:, None] * np.ones((1, n1))
            r1_mat = r1[None, :] * np.ones((n0, 1))
            rad_diff[mask_np_np] = np.abs(r0_mat[mask_np_np] - r1_mat[mask_np_np])


        # ====================== (3) ANGLE (RADIANS) ===============================
        angle = np.zeros((n0, n1), dtype=float)

        # Vector directions
        d0, n0_len = _unit(e0 - c0)
        d1, n1_len = _unit(e1 - c1)

        # Vector–vector
        if mask_vv.any():
            dots = (d0 @ d1.T)  # (n0, n1)
            vv_angles = _angles_from_unit_dot(dots)
            # set undefined angles (zero-length vectors) to π
            undefined = (
                np.outer(mask_v0, mask_v1)
                & ((n0_len[:, None] <= 1e-12) | (n1_len[None, :] <= 1e-12))
            )
            vv_angles[undefined] = np.pi
            angle[mask_vv] = vv_angles[mask_vv]

        # Vector–radial (f0 vector, f1 radial)
        if mask_vr.any():
            diff = e1[None, :, :] - c0[:, None, :]  # (n0,n1,3)
            u, ulen = _unit(diff)  # (n0,n1,3), (n0,n1)
            dots = np.sum(u * d0[:, None, :], axis=2)  # broadcasted dot product
            vr_angles = _angles_from_unit_dot(dots)
            undef = (np.outer(mask_v0, mask_r1)
                    & ((n0_len[:, None] <= 1e-12) | (ulen <= 1e-12)))
            vr_angles[undef] = np.pi
            angle[mask_vr] = vr_angles[mask_vr]

        # Vector–radial (f0 radial, f1 vector)
        if mask_rv.any():
            diff = e0[:, None, :] - c1[None, :, :]  # (n0,n1,3)
            u, ulen = _unit(diff)  # (n0,n1,3), (n0,n1)
            dots = np.sum(u * d1[None, :, :], axis=2)  # broadcasted dot product
            rv_angles = _angles_from_unit_dot(dots)
            undef = (np.outer(mask_r0, mask_v1)
                    & ((n1_len[None, :] <= 1e-12) | (ulen <= 1e-12)))
            rv_angles[undef] = np.pi
            angle[mask_rv] = rv_angles[mask_rv]


        # ====================== (4) Set Values ===============================
        if mask_pp.any():
            end_dist[mask_pp] = p_end_point_point
            rad_diff[mask_pp] = p_radius_point_point
            angle[mask_pp] = p_angle_point_point
        if mask_pv_or_vp.any():
            end_dist[mask_pv_or_vp] = p_end_point_vector
            rad_diff[mask_pv_or_vp] = p_radius_point_vector
            angle[mask_pv_or_vp] = p_angle_point_vector
        if mask_pr_or_rp.any():
            end_dist[mask_pr_or_rp] = p_end_point_radial
            rad_diff[mask_pr_or_rp] = p_radius_point_radial
            angle[mask_pr_or_rp] = p_angle_point_radial
        if mask_rr.any():
            angle[mask_rr] = p_angle_radial_radial

        distance_matrices[:, :, 0] = center_dist
        distance_matrices[:, :, 1] = end_dist
        distance_matrices[:, :, 2] = rad_diff
        distance_matrices[:, :, 3] = angle
        # Calculate weighted sum
        distance_matrices[:, :, 4] = (
            w_center * distance_matrices[:, :, 0]
            + w_end * distance_matrices[:, :, 1]
            + w_radius * distance_matrices[:, :, 2]
            + w_angle * distance_matrices[:, :, 3]
        )
        return distance_matrices.round(n_decimals)
    return distance_matrix_function


def _as_xyz(arr: object) -> np.ndarray:
    """Convert a feature coordinate to a 3D NumPy array.

    Parameters
    ----------
    arr
        Input object, expected to be either a length-3 array-like,
        None, or NaN (float).

    Returns
    -------
    np.ndarray
        1D array of shape (3,) with float dtype. If input is None/NaN,
        returns `[nan, nan, nan]`.

    Raises
    ------
    ValueError
        If `arr` is array-like but not of shape (3,).
    """
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    a = np.asarray(arr, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {a.shape}")
    return a


def _norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute Euclidean norm along a given axis.

    Parameters
    ----------
    v
        Input array.
    axis
        Axis along which to compute the norm.

    Returns
    -------
    np.ndarray
        Vector of norms along the specified axis.
    """
    return np.linalg.norm(v, axis=axis)


def _unit(v: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Normalize vectors to unit length, returning both unit vectors and norms.

    Parameters
    ----------
    v
        Input array of shape (..., 3).
    eps
        Threshold below which a norm is considered zero to avoid division by zero.

    Returns
    -------
    tuple
        - Unit vectors of same shape as `v`.
        - Norms as array of shape (...) .
    """
    n = _norm(v, axis=-1)
    safe = np.where(n > eps, n, 1.0)
    u = v / np.expand_dims(safe, -1)
    return u, n


def _angles_from_unit_dot(dot: np.ndarray) -> np.ndarray:
    """Compute angles from dot products of unit vectors.

    Parameters
    ----------
    dot
        Array of dot products in range [-1, 1].

    Returns
    -------
    np.ndarray
        Array of angles in radians, in [0, π].
    """
    np.clip(dot, -1.0, 1.0, out=dot)
    return np.arccos(dot)


def _sphere_surface_distance(
    d: np.ndarray, r0: np.ndarray, r1: np.ndarray
) -> np.ndarray:
    """Closest separation between surfaces of two spheres.

    Parameters
    ----------
    d
        Pairwise center-to-center distances.
    r0, r1
        Radii of the two spheres.

    Returns
    -------
    np.ndarray
        Minimum nonnegative surface-to-surface separation.
        Zero if the spheres intersect or are tangent.
    """
    # where spheres are separated: d > r0+r1 -> d - (r0+r1)
    sep = np.maximum(0.0, d - (r0 + r1))
    # where one sphere inside the other without touching: |r0-r1| > d -> |r0-r1| - d
    contain = np.maximum(0.0, np.abs(r0 - r1) - d)
    # if neither separation nor containment -> they intersect/tangent: distance 0
    return np.maximum(sep, contain)


def _distance_matrix(
    p0: np.ndarray,
    p1: np.ndarray,
    metric: str = 'euclidean',
    metric_kwargs: dict | None = None,
) -> np.ndarray:
    """Compute pairwise distance matrix between two sets of points.

    Parameters
    ----------
    p0
        First set of points as an (n0, 3) array.
    p1
        Second set of points as an (n1, 3) array.
    metric
        Distance metric to use.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape (n0, n1).
    """
    if (p0 is p1) or np.array_equal(p0, p1):
        return scipy.spatial.distance.squareform(
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            scipy.spatial.distance.pdist(p0, metric=metric, **(metric_kwargs or {}))
        )
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    return scipy.spatial.distance.cdist(p0, p1, metric=metric, **(metric_kwargs or {}))