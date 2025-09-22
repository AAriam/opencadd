"""Distance matrix calculation functions for pharmacophore features.

This module provides functions that return distance matrix calculation functions
suitable for use with the `Pharmacophore.cluster` and `Pharmacophore.match` methods.

A compatible distance matrix function takes two positional arguments
`f0` and `f1`, which are two DataFrames of pharmacophore features,
containing the following columns:
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

The function must return a 3D NumPy array `distance_matrix` of shape `(len(f0), len(f1), 5)`
with pairwise distance metrics between features in `f0` and `f1`.
The element `distance_matrix[i, j]` must contain the distance metrics
between feature `f0.iloc[i]` and `f1.iloc[j]` as follows:
- `distance_matrix[i, j, 0]`: Center distance.
- `distance_matrix[i, j, 1]`: End distance.
- `distance_matrix[i, j, 2]`: Radius difference.
- `distance_matrix[i, j, 3]`: Angle between vectors (in radians).
- `distance_matrix[i, j, 4]`: Weighted sum of distances.
"""

from typing import Callable, Final, TypeAlias

import numpy as np
import pandas as pd


DistanceMatrixFunction: TypeAlias = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def linear(
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
    w_center: float = 1.0,
    w_end: float = 1.0,
    w_radius: float = 1.0,
    w_angle: float = 1.0,
) -> DistanceMatrixFunction:
    """Create a distance matrix function using a linear combination of distance metrics.

    This function computes four distance matrices
    between features in `f0` and `f1`, plus their weighted sum:
    1. Pairwise distances between feature centers, calculated as follows:
        - For point–point, point–vector, and vector–vector pairs,
            this is the distance between their `center` coordinates.
        - For point–radial and vector–radial pairs, this is the distance
            between the point/vector `center` and the surface of the radial feature.
        - For radial–radial pairs, this is the minimum distance between the surfaces
            of the two radial features.
    2. Pairwise distances between feature ends, calculated as follows:
        - For point–point, point–vector, and point–radial pairs,
          the distance is set to the corresponding `p_end_*` parameter value,
          since points do not have an `end`.
        - For vector–vector, vector–radial, and radial–radial pairs,
          this is the distance between their `end` coordinates.
    3. Pairwise differences between feature radii, calculated as follows:
        - For point–point, point–vector, and point–radial pairs,
          the difference is set to the corresponding `p_radius_*` parameter value,
          since points do not have a `radius`.
        - For vector–vector, vector–radial, and radial–radial pairs,
          this is the absolute difference between their `radius` values.
    4. Pairwise angles between feature vectors, calculated as follows:
        - For vector–vector pairs, this is the angle in radians (in [0, π] range)
          between their unit vectors (from `center` to `end`, with `center` as origin).
        - For vector–radial pairs, this is the angle in radians (in [0, π] range)
          between the vector's unit vector and the unit vector
          from the vector's center to the radial feature's end.
        - For all other pairs, the angle is set
          to the corresponding `p_angle_*` parameter value.

    Parameters
    ----------
    f0
        DataFrame of features, containing the following columns:
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
    f1
        DataFrame of features, with the same required columns as `f0`.
    p_end_*
        Fixed end distance penalties for pairs for which end distance is undefined.
    p_radius_*
        Fixed radius difference penalties for pairs for which radius difference is undefined.
    p_angle_*
        Fixed angle (in radians) penalties for pairs for which angle is undefined.
    w_*
        Weights for each distance component when computing the weighted sum.

    Returns
    -------
    distance_matrix
        3D array of shape `(len(f0), len(f1), 5)` with pairwise distance metrics.
        The element `distance_matrix[i, j]` contains the distance metrics between
        feature `f0.iloc[i]` and `f1.iloc[j]` as follows:
        - `distance_matrix[i, j, 0]`: Center distance.
        - `distance_matrix[i, j, 1]`: End distance.
        - `distance_matrix[i, j, 2]`: Radius difference.
        - `distance_matrix[i, j, 3]`: Angle between vectors (in radians).
        - `distance_matrix[i, j, 4]`: Weighted sum of distances.

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

        pt0_mask = repr0 == 1
        pt1_mask = repr1 == 1
        vec0_mask = repr0 == 2
        vec1_mask = repr1 == 2
        rad0_mask = repr0 == 3
        rad1_mask = repr1 == 3

        mask_nonrad0 = ~rad0_mask
        mask_nonrad1 = ~rad1_mask

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
        vec0_no_rad_mask = np.where(vec0_mask & ~np.isfinite(r0))
        vec1_no_rad_mask = np.where(vec1_mask & ~np.isfinite(r1))
        r0[vec0_no_rad_mask] = vec0_len[vec0_no_rad_mask]
        r1[vec1_no_rad_mask] = vec1_len[vec1_no_rad_mask]

        # ---- initialize output ---------------------------------------------------
        distance_matrices = np.zeros((n0, n1, 5), dtype=float)
        tolerances = np.zeros((n0, n1, 4), dtype=float)

        # ===================== (0) CENTER DISTANCE ================================
        # default: treat center distance as Euclidean between centers when both have centers
        # Build full pairwise center distances (will be used for non-radial pairs)
        cc_diff = c0[:, None, :] - c1[None, :, :]
        cc_dist = _norm(cc_diff, axis=2)

        # Start with zeros and fill by case
        center_dist = np.zeros((n0, n1), dtype=float)

        # Case A: pairs where neither is radial -> use center-center distance
        a_mask = np.outer(mask_nonrad0, mask_nonrad1)
        center_dist[a_mask] = cc_dist[a_mask]

        # Case B: point/vector (has center) vs radial -> |‖c - e_r‖ - r|
        # f0 non-radial vs f1 radial
        b1_mask = np.outer(mask_nonrad0, rad1_mask)
        if b1_mask.any():
            dist_c_to_er = _norm(c0[:, None, :] - e1[None, :, :], axis=2)  # (n0,n1)
            offset = dist_c_to_er - rad1[None, :]  # broadcast across rows
            center_dist[b1_mask] = np.abs(offset)[b1_mask]

        # f0 radial vs f1 non-radial
        b2_mask = np.outer(rad0_mask, mask_nonrad1)
        if b2_mask.any():
            dist_c_to_er = _norm(c1[None, :, :] - e0[:, None, :], axis=2)  # (n0,n1)
            offset = dist_c_to_er - rad0[:, None]  # broadcast across cols
            center_dist[b2_mask] = np.abs(offset)[b2_mask]

        # Case C: radial–radial -> closest separation between spherical surfaces
        c_mask = np.outer(rad0_mask, rad1_mask)
        if c_mask.any():
            ee_diff = e0[:, None, :] - e1[None, :, :]
            d = _norm(ee_diff, axis=2)
            surf = _sphere_surface_distance(d, r0[:, None], r1[None, :])
            center_dist[c_mask] = surf[c_mask]

        distance_matrices[:, :, 0] = center_dist

        # ====================== (1) END DISTANCE =================================
        end_dist = np.zeros((n0, n1), dtype=float)

        # vector/vector, vector/radial, radial/radial -> Euclidean between ends
        vv_mask = np.outer(vec0_mask, vec1_mask)
        vr_mask = np.outer(vec0_mask, rad1_mask)
        rv_mask = np.outer(rad0_mask, vec1_mask)
        rr_mask = np.outer(rad0_mask, rad1_mask)

        ee_diff = e0[:, None, :] - e1[None, :, :]
        ee_dist = _norm(ee_diff, axis=2)

        for m in (vv_mask, vr_mask, rv_mask, rr_mask):
            if m.any():
                end_dist[m] = ee_dist[m]

        # pairs with at least one point: set according to parameters (symmetric)
        pp_mask = np.outer(pt0_mask, pt1_mask)
        pv_mask = np.outer(pt0_mask, vec1_mask) | np.outer(vec0_mask, pt1_mask)
        pr_mask = np.outer(pt0_mask, rad1_mask) | np.outer(rad0_mask, pt1_mask)

        if pp_mask.any():
            end_dist[pp_mask] = p_end_point_point
        if pv_mask.any():
            end_dist[pv_mask] = p_end_point_vector
        if pr_mask.any():
            end_dist[pr_mask] = p_end_point_radial

        distance_matrices[:, :, 1] = end_dist

        # ====================== (2) RADIUS DIFFERENCE =============================
        rad_diff = np.zeros((n0, n1), dtype=float)

        # Non-point pairs (vector/vector, vector/radial, radial/radial): |r0 - r1|
        nonpt0 = ~pt0_mask
        nonpt1 = ~pt1_mask
        np_mask = np.outer(nonpt0, nonpt1)
        if np_mask.any():
            r0_mat = r0[:, None] * np.ones((1, n1))
            r1_mat = r1[None, :] * np.ones((n0, 1))
            rad_diff[np_mask] = np.abs(r0_mat[np_mask] - r1_mat[np_mask])

        # Pairs with at least one point -> parameterized constants
        if pp_mask.any():
            rad_diff[pp_mask] = p_radius_point_point
        if pv_mask.any():
            rad_diff[pv_mask] = p_radius_point_vector
        if pr_mask.any():
            rad_diff[pr_mask] = p_radius_point_radial

        distance_matrices[:, :, 2] = rad_diff

        # ====================== (3) ANGLE (RADIANS) ===============================
        angle = np.zeros((n0, n1), dtype=float)

        # Vector directions
        d0, n0_len = _unit(e0 - c0)
        d1, n1_len = _unit(e1 - c1)

        # Vector–vector
        if vv_mask.any():
            dots = (d0 @ d1.T)  # (n0, n1)
            vv_angles = _angles_from_unit_dot(dots)
            # set undefined angles (zero-length vectors) to π
            undefined = (
                np.outer(vec0_mask, vec1_mask)
                & ((n0_len[:, None] <= 1e-12) | (n1_len[None, :] <= 1e-12))
            )
            vv_angles[undefined] = np.pi
            angle[vv_mask] = vv_angles[vv_mask]

        # Vector–radial (f0 vector, f1 radial)
        if vr_mask.any():
            diff = e1[None, :, :] - c0[:, None, :]  # (n0,n1,3)
            u, ulen = _unit(diff)  # (n0,n1,3), (n0,n1)
            dots = np.sum(u * d0[:, None, :], axis=2)  # broadcasted dot product
            vr_angles = _angles_from_unit_dot(dots)
            undef = (np.outer(vec0_mask, rad1_mask)
                    & ((n0_len[:, None] <= 1e-12) | (ulen <= 1e-12)))
            vr_angles[undef] = np.pi
            angle[vr_mask] = vr_angles[vr_mask]

        # Vector–radial (f0 radial, f1 vector)
        if rv_mask.any():
            diff = e0[:, None, :] - c1[None, :, :]  # (n0,n1,3)
            u, ulen = _unit(diff)  # (n0,n1,3), (n0,n1)
            dots = np.sum(u * d1[None, :, :], axis=2)  # broadcasted dot product
            rv_angles = _angles_from_unit_dot(dots)
            undef = (np.outer(rad0_mask, vec1_mask)
                    & ((n1_len[None, :] <= 1e-12) | (ulen <= 1e-12)))
            rv_angles[undef] = np.pi
            angle[rv_mask] = rv_angles[rv_mask]

        # All other pairs use parameter values
        # point–point
        if pp_mask.any():
            angle[pp_mask] = p_angle_point_point
        # point–vector (both orders)
        if pv_mask.any():
            angle[pv_mask] = p_angle_point_vector
        # point–radial (both orders)
        if pr_mask.any():
            angle[pr_mask] = p_angle_point_radial
        # radial–radial
        if rr_mask.any():
            angle[rr_mask] = p_angle_radial_radial

        distance_matrices[:, :, 3] = angle

        # Calculate weighted sum
        distance_matrices[:, :, 4] = (
            w_center * distance_matrices[:, :, 0]
            + w_end * distance_matrices[:, :, 1]
            + w_radius * distance_matrices[:, :, 2]
            + w_angle * distance_matrices[:, :, 3]
        )

        return distance_matrices
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
