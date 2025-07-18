import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import gamma


def isoperimetric_quotient(points: np.ndarray) -> float:
    """Calculate the normalized isoperimetric quotient of the point cloud's convex hull.

    The isoperimetric quotient is the reciprocal of the
    [isoperimetric ratio](https://en.wikipedia.org/wiki/Isoperimetric_ratio);
    it measures the "ball-likeness" (i.e., circle-likeness in 2D, sphere-likeness in 3D, etc.)
    of the point cloud's convex hull.

    Returns
    -------
    Normalized isoperimetric quotient of the point cloud
    in the range (0, 1], where 1 indicates a perfect d-ball and
    lower values indicate greater deviation from ball-likeness (i.e., wasting surface area),
    either by having indentations, protrusions, or elongations,
    relative to how much volume the convex hull encloses.

    Notes
    -----
    The isoperimetric quotient `Ψ` measures how efficiently a given shape
    "packs" volume into surface area, relative to an ideal `d`-dimensional ball.
    It compares the `d`-dimensional volume `V`
    to the `(d-1)`-dimensional boundary measure `A`
    (i.e., perimeter in 2D, surface area in 3D, hypersurface volume in higher dimensions)
    against the optimal ratio achieved by a perfect d-ball:

        Ψ = (d * ω**(1/d) * V**((d-1)/d)) / A,

    where
    - d is the point dimension (i.e., `self.point_dim`),
    - ω = π^{d/2} / Γ(d/2 + 1) is the d-dimensional volume of the unit d-ball,
    - V is the d-dimensional volume of the convex hull,
    - A is the (d-1)-dimensional surface measure of the convex hull boundary.

    Because Ψ is purely global, it is insensitive to interior point distributions:
    whether you have a thin shell of points on the surface or a full volumetric fill,
    only the outer shape matters.
    """
    points = np.asarray(points)
    # Determine dimension
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, d)")
    N, d = points.shape
    if N <= d:
        raise ValueError("Need more points than dimension to form a convex hull")

    hull = ConvexHull(points)
    V = hull.volume  # d-dimensional volume
    A = hull.area    # (d-1)-dim boundary measure
    omega = np.pi**(d/2) / gamma(d/2 + 1)  # volume of unit d-ball
    psi = (d * omega**(1/d) * V**((d - 1) / d)) / A  # isoperimetric quotient
    return float(psi)
