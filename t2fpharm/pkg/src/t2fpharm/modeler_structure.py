from typing import Literal, Sequence
from caddpy.bond import Bond
import caddpy

import pandas as pd
import numpy as np

from t2fpharm.pharm import Pharmacophore
from t2fpharm.system import System


class StructureBasedModeler:
    def __init__(self, system: System):
        self._system = system
        # Make sure autodock types are assigned
        self._system.composition.autodock_atom_type()
        self._trajectory = system.trajectory.points

        self._batch_shape = self._trajectory.shape[:-2]
        self._batch_ndim = len(self._batch_shape)
        self._bond = Bond(caddpy.chemsys._ccd("chem_comp_bond")).select(
            comp_id=self._system.composition.atoms["comp_id"]
        )
        self._atom = None
        self._len_hbond = None
        self._len_ionic = None
        self._len_hydrophobic = None
        self._type_hbond_donor = None
        self._type_hbond_acceptor = None
        return

    def model(
        self,
        *,
        type_hbond_acceptor: str | None = "OA",
        type_hbond_donor: str | None = "HD",
        type_anionic: str | None = "e-",
        type_cationic: str | None = "e+",
        type_hydrophobic: str | None = "C",
        atom_mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        len_hbond: float = 2.5,
        len_ionic: float = 3.0,
        len_hydrophobic: float = 4.0,
        len_tol_hbond: float = 1,
        len_tol_ionic: float = 1,
        len_tol_hydrophobic: float = 1,
        angle_tol_hbond: float = 1.4,  # ~80 degrees
    ) -> Pharmacophore:
        self._type_hbond_donor = type_hbond_donor
        self._type_hbond_acceptor = type_hbond_acceptor
        self._type_anionic = type_anionic
        self._type_cationic = type_cationic
        self._type_hydrophobic = type_hydrophobic

        self._atom = self._system.composition.atoms
        if atom_mask is not None:
            self._atom = self._atom[atom_mask]

        self._len_hbond = len_hbond
        self._len_ionic = len_ionic
        self._len_hydrophobic = len_hydrophobic
        self._len_tol_hbond = len_tol_hbond
        self._len_tol_ionic = len_tol_ionic
        self._len_tol_hydrophobic = len_tol_hydrophobic
        self._angle_tol_hbond = angle_tol_hbond

        self._feats = (
            pd.concat(
                [
                    *self._calc_non_directed_interactions(),
                    *self._calc_hbond_acceptors(),
                    *self._calc_hbond_donors(),
                ],
                ignore_index=True
            )
            .convert_dtypes()
            .sort_values(
                (["instance"] if self._batch_ndim > 0 else []) + ["type", "res_idx", "atom_idxs"]
            )
            .reset_index(drop=True)
        )
        return self._feats

    def _calc_non_directed_interactions(self) -> list[pd.DataFrame]:
        dfs = []
        charges = self._atom["charge"]
        for ftype, radius, radius_tol, df_atom in (
            (self._type_anionic, self._len_ionic, self._len_tol_ionic, self._atom[charges > 0]),
            (self._type_cationic, self._len_ionic, self._len_tol_ionic, self._atom[charges < 0]),
            (self._type_hydrophobic, self._len_hydrophobic, self._len_tol_hydrophobic, self._hydrophobic_atoms())
        ):
            if ftype is None or df_atom.empty:
                continue
            atom_idx = df_atom["atom_idx"].to_numpy()
            end_coords = self._trajectory[..., atom_idx, :]
            df = self._create_df(
                res_idx=df_atom["res_idx"].to_numpy(),
                atom_idx=[(int(idx),) for idx in atom_idx],
                feature_type=ftype,
                center=None,
                end=end_coords,
                radius=radius,
                radius_tol=radius_tol,
            )
            dfs.append(df)
        return dfs

    def _calc_hbond_acceptors(self) -> list[pd.DataFrame]:
        if self._type_hbond_acceptor is None:
            return []
        hd = self._atom[self._atom["autodock_atom_type"] == "HD"]
        if hd.empty:
            return []
        hd = self._merge_with_partners(hd)
        atom_idx = hd["atom_idx"].to_numpy()
        donor_coords = self._trajectory[..., atom_idx, :]
        accep_coords = self._trajectory[..., hd["partner_atom_idx"].to_numpy(), :]
        accep_to_donor = donor_coords - accep_coords
        distances = np.linalg.norm(accep_to_donor, axis=-1, keepdims=True)
        accep_to_donor_unit = accep_to_donor / distances
        feature_coords = donor_coords + accep_to_donor_unit * self._len_hbond
        df = self._create_df(
            res_idx=hd["res_idx"].to_numpy(),
            atom_idx=[(int(idx),) for idx in atom_idx],
            feature_type=self._type_hbond_acceptor,
            center=feature_coords,
            end=donor_coords,
            radius_tol=self._len_tol_hbond,
            angle_tol=self._angle_tol_hbond,
        )
        return [df]

    def _calc_hbond_donors(self) -> list[pd.DataFrame]:
        if self._type_hbond_donor is None:
            return []
        dfs = []
        oa = self._atom[self._atom["autodock_atom_type"] == "OA"]
        na = self._atom[self._atom["autodock_atom_type"] == "NA"]
        if not oa.empty:
            oam = self._merge_with_partners(oa)
            o_num_partners = oam.groupby("atom_idx")["atom_idx"].transform("count")
            oa1 = oam[o_num_partners == 1]
            oa2 = oam[o_num_partners == 2]
            dfs.extend([
                self._calc_oa1(oa1),
                self._calc_tetrahedral_acceptor(oa2, radius_tol=self._len_tol_hbond, angle_tol=self._angle_tol_hbond)
            ])
        if not na.empty:
            nam = self._merge_with_partners(na)
            # Add backbone bonding partner (C from previous residue) for amide nitrogens
            nam_n = nam[(nam["res_poly"]) & (nam["atom_id"]=="N") & (nam["res_idx"] != 1)].copy()
            nam_n["partner_res_num"] = nam_n["res_idx"] - 1
            nam_n["partner_atom_id"] = "C"
            nam_n = nam_n.merge(
                self._atom[["res_idx", "atom_id", "atom_idx"]].rename(
                        columns={"res_idx": "partner_res_num", "atom_id": "partner_atom_id", "atom_idx": "partner_atom_idx"}
                    ),
                    how="left",
                    on=["partner_res_num", "partner_atom_id"],
            )
            nam = pd.concat([nam, nam_n], ignore_index=True)
            atoms_with_missing_partners = nam[nam["partner_atom_idx"].isna()]["atom_idx"].unique()
            if len(atoms_with_missing_partners) > 0:
                nam = nam[~nam["atom_idx"].isin(atoms_with_missing_partners)].convert_dtypes()

            n_num_partners = nam.groupby("atom_idx")["atom_idx"].transform("count")
            na2 = nam[n_num_partners == 2]
            na3 = nam[n_num_partners == 3]
            dfs.extend([
                self._calc_na2(na2),
                self._calc_tetrahedral_acceptor(na3, radius_tol=self._len_tol_hbond, angle_tol=self._angle_tol_hbond)
            ])
        return dfs

    def _calc_tetrahedral_acceptor(self, df: pd.DataFrame, radius_tol: float, angle_tol: float) -> pd.DataFrame:
        # group each atom_idx and collect its partners
        grouped = df.groupby("atom_idx").agg({
            "res_idx": "first",
            "partner_atom_idx": list,
        })
        # now split into aligned vectors
        center_indices = grouped.index.to_numpy()
        partner_indices = [partners for partners in grouped["partner_atom_idx"]]
        partner_indices_columnwise = list(map(list, zip(*partner_indices)))
        center_coords = self._trajectory[..., center_indices, :]
        partners_coords = [self._trajectory[..., p_indices, :] for p_indices in partner_indices_columnwise]
        feat_coords = fill_tetrahedral(
            center_coords,
            *partners_coords,
            length=self._len_hbond,
        )
        n_partners_per_center = len(partner_indices[0])
        n_features_per_center = 4 - n_partners_per_center
        return self._create_df(
            res_idx=np.repeat(grouped["res_idx"], n_features_per_center),
            atom_idx=[(int(idx),) for idx in np.repeat(center_indices, n_features_per_center)],
            feature_type=self._type_hbond_donor,
            center=feat_coords.reshape(-1, 3),
            end=np.stack([center_coords]*n_features_per_center, axis=-2).reshape(-1, 3),
            radius_tol=radius_tol,
            angle_tol=angle_tol,
        )

    def _calc_oa1(self, oa):
        # Get the bonding partners of the (only one) partner, to define the in-plane direction
        partners = self._merge_with_partners(
            oa[["res_idx", "atom_idx", "partner_atom_idx"]].rename(
                columns={
                    "res_idx": "oxygen_res_idx",
                    "atom_idx": "oxygen_atom_idx",
                    "partner_atom_idx": "atom_idx",
                }
            ).merge(self._atom, on="atom_idx", how="left"),
            extra_cols=["element_index"],
            drop_if_any_missing=False
        )
        partners = partners[partners["partner_atom_idx"].notna()]
        # Exclude the original O atom from the partners
        partners = partners[~partners["partner_atom_idx"].isin(oa["atom_idx"])]
        # Select the partner with the highest atomic number as the anchor for the in-plane direction
        partners = partners.loc[partners.groupby(["atom_idx", "oxygen_atom_idx"])["partner_element_index"].idxmax()]

        p1_indices = partners["atom_idx"].to_numpy()
        in_plane_indices = partners["partner_atom_idx"].to_numpy()
        o_indices = partners["oxygen_atom_idx"].to_numpy()

        o_coords = self._trajectory[..., o_indices, :]
        feat_coords = fill_trigonal(
            o_coords,
            self._trajectory[..., p1_indices, :],
            in_plane=self._trajectory[..., in_plane_indices, :],
            length=self._len_hbond,
        )
        return self._create_df(
            res_idx=np.repeat(partners["oxygen_res_idx"], 2),
            atom_idx=[(int(idx),) for idx in np.repeat(o_indices, 2)],
            feature_type=self._type_hbond_donor,
            center=feat_coords.reshape(-1, 3),
            end=np.stack([o_coords]*2, axis=-2).reshape(-1, 3),
            radius_tol=self._len_tol_hbond,
            angle_tol=self._angle_tol_hbond,
        )

    def _calc_na2(self, na):
        # group each atom_idx and collect its partners
        grouped = na.groupby("atom_idx").agg({
            "res_idx": "first",
            "partner_atom_idx": list,
        })
        # now split into aligned vectors
        center_indices = grouped.index.to_numpy()
        partner_indices = [partners for partners in grouped["partner_atom_idx"]]
        partner_indices_columnwise = list(map(list, zip(*partner_indices)))
        center_coords = self._trajectory[..., center_indices, :]
        partners_coords = [self._trajectory[..., p_indices, :] for p_indices in partner_indices_columnwise]
        feat_coords = fill_trigonal(
            center_coords,
            *partners_coords,
            length=self._len_hbond,
        )
        n_partners_per_center = len(partner_indices[0])
        n_features_per_center = 3 - n_partners_per_center
        return self._create_df(
            res_idx=np.repeat(grouped["res_idx"], n_features_per_center),
            atom_idx=[(int(idx),) for idx in np.repeat(center_indices, n_features_per_center)],
            feature_type=self._type_hbond_donor,
            center=feat_coords.reshape(-1, 3),
            end=np.stack([center_coords]*n_features_per_center, axis=-2).reshape(-1, 3),
            radius_tol=self._len_tol_hbond,
            angle_tol=self._angle_tol_hbond,
        )

    def _create_df(
        self,
        res_idx: np.ndarray,
        atom_idx: list[tuple[int, ...]],
        feature_type: str,
        center: np.ndarray | None,
        end: np.ndarray,
        radius: float | None = None,
        radius_tol: float = 0.0,
        angle_tol: float = 0.0,
    ):
        if self._batch_ndim == 0:
            instance = None
        else:
            instance = (
                np.repeat(np.arange(self._batch_shape[0]), repeats=len(atom_idx))
                if self._batch_ndim == 1 else
                [tuple(x) for x in np.repeat(list(np.ndindex(self._batch_shape)), repeats=len(atom_idx), axis=0).tolist()]
            )
            end = end.reshape(-1, 3)
            if center is not None:
                center = center.reshape(-1, 3)
        n_instances = int(np.prod(self._batch_shape))
        cols = {
            "type": feature_type,
            "res_idx": np.tile(res_idx, n_instances),
            "atom_idxs": atom_idx * n_instances,
            "end": list(end),
            "radius_tol": radius_tol,
            "angle_tol": angle_tol,
        }
        if self._batch_ndim != 0:
            cols["instance"] = instance
        if center is not None:
            cols["center"] = list(center)
        if radius is not None:
            cols["radius"] = radius

        return pd.DataFrame(cols).convert_dtypes()

    def _merge_with_partners(
        self,
        df: pd.DataFrame,
        extra_cols: list[str] | None = None,
        drop_if_any_missing: bool = True
    ) -> pd.DataFrame:
        extra_cols = extra_cols or []
        dfm = (
            df
            .merge(
                self._bond.exploded[["comp_id", "atom_id", "partner_atom_id"]],
                how="left",
                on=["comp_id", "atom_id"],
            )
            .merge(
                self._atom[["res_idx", "atom_id", "atom_idx"] + extra_cols].rename(
                    columns={"atom_id": "partner_atom_id", "atom_idx": "partner_atom_idx"} | {
                        col: f"partner_{col}" for col in extra_cols
                    }
                ),
                how="left",
                on=["res_idx", "partner_atom_id"],
            )
        )
        if drop_if_any_missing:
            atoms_with_missing_partners = dfm[dfm["partner_atom_idx"].isna()]["atom_idx"].unique()
            if len(atoms_with_missing_partners) > 0:
                dfm = dfm[~dfm["atom_idx"].isin(atoms_with_missing_partners)]
        return dfm

    def _hydrophobic_atoms(self) -> pd.DataFrame:
        """Get a subset of `self._atom` containing only hydrophobic atoms.

        Hydrophobic atoms are defined as carbon atoms with autodock type "C"
        that are only bonded to carbon or hydrogen atoms.
        """
        atom_c = self._atom[self._atom["autodock_atom_type"] == "C"]
        bond_to_merge = self._bond.exploded[["comp_id", "atom_id", "partner_atom_id"]]
        atom_to_merge = (
            self._atom[["comp_id", "atom_id", "type_symbol"]]
            .drop_duplicates(keep="first")
            .rename(
                columns={"atom_id": "partner_atom_id", "type_symbol": "partner_type_symbol"}
            )
        )
        return (
            atom_c
            .merge(bond_to_merge, on=["comp_id", "atom_id"], how="left")
            .merge(atom_to_merge, on=["comp_id", "partner_atom_id"], how="left")
            .groupby("atom_idx")
            .filter(lambda group: set(group["partner_type_symbol"]).issubset({"C", "H"}))
        )


def fill_trigonal(
    center: np.ndarray,
    vertex1: np.ndarray,
    vertex2: np.ndarray | None = None,
    *,
    in_plane: np.ndarray | None = None,
    length: Literal["mean", "min", "max", "median"] | float | np.ndarray | None = None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Given the coordinates of a trigonal center and one or two vertices, calculate the remaining vertices.

    This function calculates unit vectors that point from the center
    to the remaining vertices to maintain trigonal geometry at the center.
    That is, distance of the provided vertices from the center does not matter;
    they are only used to determine the remaining vertex directions, as follows:
    - If only one vertex is provided, the remaining two are calculated
      such that all vertex-center-vertex angles are identical (120 degrees).
    - If two vertices are provided, the remaining one is calculated
      such that its vertex-center-vertex angle to each of the provided vertices is identical

    Parameters
    ----------
    center
        An array of shape (..., 3) giving the coordinates of the center points.
    vertex1
        An array of shape (..., 3) giving the coordinates of the first vertices.
    vertex2
        An array of shape (..., 3) giving the coordinates of the second vertices.
    vertex3
        An array of shape (..., 3) giving the coordinates of the third vertices.
    in_plane
        This is only used if only one vertex is provided.
        In that case, the two calculated remaining vertices are placed in the plane
        defined by the center, the provided vertex, and this point,
        otherwise an arbitrary but deterministic plane is chosen.
    length
        If not None, instead of returning direction vectors,
        return coordinates of vertices at this distance from the center:
        - If length is a scalar, the same distance is used for all vertices.
        - If length is an array, it must either have shape (..., N) where N is the number of vertices to be returned,
          or shape (...), in which case the same distance is used for all vertices of a given center.
        - If length is one of "mean", "min", "max", or "median", the respective statistic of the distances
          to the provided vertices is used.
    tolerance
        Tolerance for determining whether provided vertices differ from the center.

    Returns
    -------
    An array of shape (..., N, 3) containing N unit vectors
    pointing from each center to its remaining vertices,
    where N is the total number of calculated vertex directions per center
    (2 if vertex2 is None, and 1 otherwise).
    If `length` is not None, instead of unit vectors,
    coordinates of the vertices at the specified distance from the center are returned.
    """
    center = np.asarray(center, dtype=float)

    # Deltas from center to provided vertices
    d1 = np.asarray(vertex1, dtype=float) - center
    d2 = None if vertex2 is None else (np.asarray(vertex2, dtype=float) - center)

    # Validate non-zero
    _any_too_small([("vertex1", d1), ("vertex2", d2)], tolerance=tolerance)

    # Unit directions of provided
    v1 = _normalize(d1, tolerance=tolerance)

    cos120 = -0.5
    sin120 = np.sqrt(3.0) / 2.0

    if vertex2 is None:
        # -------- Case: 1 provided vertex -> compute the other two in a chosen plane --------
        # Determine plane normal n
        if in_plane is not None:
            ip = np.asarray(in_plane, dtype=float) - center
            n_raw = np.cross(v1, ip)
        else:
            n_raw = np.zeros_like(v1)

        # If degenerate (no usable in_plane or collinear), pick a deterministic plane via a perp to v1
        need_fallback = (np.linalg.norm(n_raw, axis=-1, keepdims=True) <= tolerance)
        e1p, e2p = _orthonormal_basis_perp_to(v1, tolerance=tolerance)  # both ⟂ v1
        # Build a plane normal that makes plane = span{v1, e1p}
        n_fallback = _normalize(np.cross(v1, e1p), tolerance=tolerance)
        n = np.where(need_fallback, n_fallback, _normalize(n_raw, tolerance=tolerance))

        # In-plane orthonormal basis: e1 along v1, e2 = n × e1
        e1 = v1
        e2 = _normalize(np.cross(n, e1), tolerance=tolerance)

        # Rotate e1 by ±120° within plane
        u_plus  = _normalize(cos120 * e1 +  sin120 * e2, tolerance=tolerance)
        u_minus = _normalize(cos120 * e1 + (-sin120) * e2, tolerance=tolerance)

        dirs = np.stack([u_plus, u_minus], axis=-2)  # (..., 2, 3)
        return _place_vertices(center, dirs, length, [d1])

    # -------- Case: 2 provided vertices -> compute the unique third with equal angles --------
    v2 = _normalize(d2, tolerance=tolerance)  # type: ignore[arg-type]

    # compute sum and detect degeneracies
    s = v1 + v2
    s_norm = np.linalg.norm(s, axis=-1, keepdims=True)
    same_dir = (np.linalg.norm(v1 - v2, axis=-1, keepdims=True) <= tolerance)
    opp_dir  = (np.linalg.norm(v1 + v2, axis=-1, keepdims=True) <= tolerance)

    # build deterministic in-plane perpendicular e2 to use for fallbacks
    e1p, _ = _orthonormal_basis_perp_to(v1, tolerance=tolerance)
    n = _normalize(np.cross(v1, e1p), tolerance=tolerance)  # plane normal
    e2 = _normalize(np.cross(n, v1), tolerance=tolerance)   # in-plane ⟂ v1

    # Preferred: "outer" equal-angle direction if s is non-degenerate and not same_dir
    w_pref = - _normalize(s, tolerance=tolerance)
    w = np.where((s_norm > tolerance) & (~same_dir), w_pref, e2)
    w = _normalize(w, tolerance=tolerance)

    dirs = w[..., None, :]  # (..., 1, 3)
    return _place_vertices(center, dirs, length, [d1, d2])  # type: ignore[list-item]


def fill_tetrahedral(
    center: np.ndarray,
    vertex1: np.ndarray,
    vertex2: np.ndarray | None = None,
    vertex3: np.ndarray | None = None,
    *,
    length: Literal["mean", "min", "max", "median"] | float | np.ndarray | None = None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Given the coordinates of a tetrahedral center and one or more vertices, calculate the remaining vertices.

    This function calculates unit vectors that point from the center
    to the remaining vertices to maintain tetrahedral geometry at the center.
    That is, distance of the provided vertices from the center does not matter;
    they are only used to determine the remaining vertex directions, as follows:
    - If only one vertex is provided, the remaining three are calculated
      such that all vertex-center-vertex angles are identical (109.47 degrees).
    - If two vertices are provided, the remaining two are calculated
      such that their vertex-center-vertex angle is 109.47 degrees,
      and they lie in a plane orthogonal to the plane defined by the center and the two provided vertices,
      symmetrically on either side of that plane.
    - If three vertices are provided, the remaining one is calculated
      such that its vertex-center-vertex angle to each of the provided vertices is identical.

    Parameters
    ----------
    center
        An array of shape (..., 3) giving the coordinates of the center points.
    vertex1
        An array of shape (..., 3) giving the coordinates of the first vertices.
    vertex2
        An array of shape (..., 3) giving the coordinates of the second vertices.
    vertex3
        An array of shape (..., 3) giving the coordinates of the third vertices.
    length
        If not None, instead of returning direction vectors,
        return coordinates of vertices at this distance from the center:
        - If length is a scalar, the same distance is used for all vertices.
        - If length is an array, it must either have shape (..., N) where N is the number of vertices to be returned,
          or shape (...), in which case the same distance is used for all vertices of a given center.
        - If length is one of "mean", "min", "max", or "median", the respective statistic of the distances
          to the provided vertices is used.
    tolerance
        Tolerance for determining whether provided vertices differ from the center.

    Returns
    -------
    An array of shape (..., N, 3) containing N unit vectors
    pointing from each center to its remaining vertices,
    where N is the total number of calculated vertex directions per center
    (3 if vertex2 and vertex3 are None, 2 if vertex2 or vertex3 is None,
    and 1 if both vertex2 and vertex3 are provided).
    If `length` is not None, instead of unit vectors,
    coordinates of the vertices at the specified distance from the center are returned.
    """
    center = np.asarray(center, dtype=float)

    # Deltas from center to the provided vertices
    d1 = np.asarray(vertex1, dtype=float) - center
    d2 = (None if vertex2 is None else (np.asarray(vertex2, dtype=float) - center))
    d3 = (None if vertex3 is None else (np.asarray(vertex3, dtype=float) - center))

    # Validate non-zero directions for any provided edges
    _any_too_small([("vertex1", d1), ("vertex2", d2), ("vertex3", d3)], tolerance=tolerance)

    v1 = _normalize(d1, tolerance=tolerance)

    if vertex2 is None and vertex3 is None:
        # --- Case: 1 provided edge -> compute the other three so that all mutual angles are 109.47°
        # Construction:
        #   Let v1 be known. Choose e1,e2 ⟂ v1. Define three unit directions in the plane ⟂ v1 at 0°,120°,240°
        #   and tilt them towards -v1 by α=1/3; β = 2*sqrt(2)/3 to keep unit length.
        e1_perp, e2_perp = _orthonormal_basis_perp_to(v1, tolerance=tolerance)
        cos120 = -0.5
        sin120 = np.sqrt(3.0) / 2.0
        u0 = e1_perp
        u1 = cos120 * e1_perp + sin120 * e2_perp
        u2 = cos120 * e1_perp - sin120 * e2_perp

        alpha = 1.0 / 3.0
        beta = 2.0 * np.sqrt(2.0) / 3.0
        m = -alpha * v1  # component along -v1

        w0 = _normalize(m + beta * u0, tolerance=tolerance)
        w1 = _normalize(m + beta * u1, tolerance=tolerance)
        w2 = _normalize(m + beta * u2, tolerance=tolerance)
        dirs = np.stack([w0, w1, w2], axis=-2)
        return _place_vertices(center, dirs, length, [d1])

    if (vertex2 is not None) ^ (vertex3 is not None):
        # --- Case: 2 provided edges -> compute the other two.
        v2 = _normalize(d2 if d2 is not None else d3, tolerance=tolerance)

        # Orthonormal basis of the plane ⟂ v1 (same frame used in 1-edge case)
        e1_perp, e2_perp = _orthonormal_basis_perp_to(v1, tolerance=tolerance)

        # In-plane unit for v2 (drop its component along v1, then normalize)
        v2_inplane_raw = v2 - np.sum(v2 * v1, axis=-1, keepdims=True) * v1
        v2_inplane = _normalize(v2_inplane_raw, tolerance=tolerance)

        # Coordinates of v2_inplane in (e1_perp, e2_perp)
        c1 = np.sum(v2_inplane * e1_perp, axis=-1, keepdims=True)
        c2 = np.sum(v2_inplane * e2_perp, axis=-1, keepdims=True)

        # Rotate by ±120° within the ⟂-to-v1 plane to get the other two in-plane directions
        cos120 = -0.5
        sin120 = np.sqrt(3.0) / 2.0

        # u+ corresponds to +120°, u- to -120°
        u_plus  = (cos120 * c1 - sin120 * c2) * e1_perp + (sin120 * c1 + cos120 * c2) * e2_perp
        u_minus = (cos120 * c1 + sin120 * c2) * e1_perp + (-sin120 * c1 + cos120 * c2) * e2_perp
        u_plus  = _normalize(u_plus, tolerance=tolerance)
        u_minus = _normalize(u_minus, tolerance=tolerance)

        # Same tilt as 1-edge case: component −1/3 along v1, and 2√2/3 in-plane
        alpha = 1.0 / 3.0
        beta  = 2.0 * np.sqrt(2.0) / 3.0
        m = -alpha * v1

        w_plus  = _normalize(m + beta * u_plus, tolerance=tolerance)
        w_minus = _normalize(m + beta * u_minus, tolerance=tolerance)
        dirs = np.stack([w_plus, w_minus], axis=-2)
        return _place_vertices(center, dirs, length, [d1, d2 if d2 is not None else d3])

    # --- Case: 3 provided edges -> compute the unique direction with equal angle to all three.
    # Requirement: find unit w with w·v1 = w·v2 = w·v3 (identical).
    # This is satisfied by w ∥ -(v1 + v2 + v3). (Magnitude sets the common cosine.)
    v2 = _normalize(d2, tolerance=tolerance)
    v3 = _normalize(d3, tolerance=tolerance)
    s3 = v1 + v2 + v3
    norm_s3 = np.linalg.norm(s3, axis=-1, keepdims=True)
    if np.any(norm_s3 <= tolerance):
        raise ValueError(
            "The three provided edge directions sum to ~0; the remaining direction is ambiguous."
        )
    w = _normalize(-s3, tolerance=tolerance)
    dirs = w[..., None, :]
    return _place_vertices(center, dirs, length, [d1, d2, d3])


def _normalize(v: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Normalize last-axis vectors; leaves zeros as zeros."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    safe_norms = np.where(norms > tolerance, norms, 1.0)
    out = v / safe_norms
    # for truly zero vectors, keep zeros (avoid NaNs)
    return np.where(norms > tolerance, out, 0.0)


def _any_too_small(vecs: list[tuple[str, np.ndarray]], tolerance: float = 1e-12) -> None:
    """Check whether any of the provided vectors has (near-)zero length; raise if so."""
    bad = []
    for name, vec in vecs:
        if vec is None:
            continue
        n = np.linalg.norm(vec, axis=-1)
        if np.any(n <= tolerance):
            bad.append(name)
    if bad:
        raise ValueError(
            f"Zero-length direction(s) found for: {', '.join(bad)}. "
            "Vertices must differ from centers."
        )
    return


def _orthonormal_basis_perp_to(u: np.ndarray, tolerance: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Given unit u (...,3), return e1,e2 unit and orthonormal with e1⊥u, e2⊥(u,e1)."""
    # Choose a reference not (near-)parallel to u for stability
    # Prefer x-axis unless |u_x| is too large
    ref_x = np.array([1.0, 0.0, 0.0])
    ref_y = np.array([0.0, 1.0, 0.0])
    use_y = (np.abs(u[..., 0]) > 0.9)[..., None]
    ref = np.where(use_y, ref_y, ref_x)
    e1 = _normalize(np.cross(u, ref), tolerance=tolerance)
    # If u is (near) parallel to chosen ref (rare but possible), fall back to z
    fallback = _normalize(np.cross(u, np.array([0.0, 0.0, 1.0])), tolerance=tolerance)
    e1 = np.where(
        (np.linalg.norm(e1, axis=-1, keepdims=True) > tolerance),
        e1,
        fallback,
    )
    e2 = _normalize(np.cross(u, e1), tolerance=tolerance)
    return e1, e2


def _place_vertices(
    center: np.ndarray,
    dirs: np.ndarray,                  # (..., N, 3) unit directions
    length: Literal["mean", "min", "max", "median"] | float | np.ndarray | None,
    provided_deltas: list[np.ndarray], # used for stats if length is str
) -> np.ndarray:
    """Return either unit directions or positioned vertices at requested length."""
    if length is None:
        return dirs
    scales = _resolve_scales(center, dirs, length, provided_deltas)  # (..., N)
    return center[..., None, :] + dirs * scales[..., None]


def _resolve_scales(
    center: np.ndarray,
    dirs: np.ndarray,                  # (..., N, 3)
    length: Literal["mean", "min", "max", "median"] | float | np.ndarray | None,
    provided_deltas: list[np.ndarray], # each (..., 3)
) -> np.ndarray:
    """Return per-vertex scales with shape (..., N)."""
    N = dirs.shape[-2]
    if length is None:
        # dummy, not used by caller in this case
        return np.ones((*center.shape[:-1], N), dtype=float)

    if isinstance(length, str):
        # stats over provided distances per center -> shape (...,)
        dists = np.stack([np.linalg.norm(d, axis=-1) for d in provided_deltas], axis=-1)
        if length == "mean":
            L = np.mean(dists, axis=-1)
        elif length == "min":
            L = np.min(dists, axis=-1)
        elif length == "max":
            L = np.max(dists, axis=-1)
        elif length == "median":
            L = np.median(dists, axis=-1)
        else:
            raise ValueError("length must be one of {'mean','min','max','median'}, a scalar, an array, or None.")
        return np.broadcast_to(L[..., None], (*center.shape[:-1], N))

    if np.isscalar(length):
        return np.broadcast_to(float(length), (*center.shape[:-1], N))

    arr = np.asarray(length, dtype=float)
    try:
        # Prefer per-vertex specification (..., N)
        return np.broadcast_to(arr, (*center.shape[:-1], N))
    except ValueError:
        # Fallback: per-center specification (...,)
        base = np.broadcast_to(arr, center[..., 0].shape)
        return np.broadcast_to(base[..., None], (*center.shape[:-1], N))


def _calculate_angles_at_center(
    center: np.ndarray,
    ligands: np.ndarray,
) -> np.ndarray:
    """Calculate angles between the center and each unique pair of ligands.

    Parameters
    ----------
    center
        An array of shape (..., 3) giving the coordinates of the center points.
    ligands
        An array of shape (..., N, 3) giving the coordinates of N ligand points
        associated with each center.

    Returns
    -------
    angles
        An array of shape (..., N, N) giving the angles (in radians)
        between each unique pair of ligand points, with respect to the center.
        That is, `angles[..., i, j]` is the angle at the center
        between `ligands[..., i, :]` and `ligands[..., j, :]`.
    """
    center = np.asarray(center, dtype=float)
    ligands = np.asarray(ligands, dtype=float)
    if ligands.ndim < 2 or ligands.shape[-1] != 3:
        raise ValueError("ligands must have shape (..., N, 3)")

    # Vectors from center to each ligand
    vecs = ligands - center[..., None, :]

    # Normalize to unit vectors
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    unit_vecs = vecs / safe_norms
    unit_vecs = np.where(norms > 1e-12, unit_vecs, 0.0)

    # Cosine of angles between all pairs of unit vectors
    cos_angles = np.einsum("...ik,...jk->...ij", unit_vecs, unit_vecs)

    # Clamp to [-1, 1] to avoid NaNs from numerical noise
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Angles in radians
    angles = np.arccos(cos_angles)
    return angles
