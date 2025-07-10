from typing import Any, Sequence, Self
from pathlib import Path

import numpy as np
import scishow
import caddpy

from t2fpharm.pharmacophore import Pharmacophore
from t2fpharm.receptor import Receptor
from t2fpharm.pocket import Pocket
from t2fpharm.typing import DataFrameLike


class LigandPharmacophore(Pharmacophore):
    """Ligand-based pharmacophore representation.

    Parameters
    ----------
    features
        DataFrame-like object containing pharmacophore feature data.
        It can be a `pandas.DataFrame`, or any object that can be
        converted to a DataFrame using the `pandas.DataFrame()` constructor.
        Each row in the resulting DataFrame must represent
        a pharmacophore feature with the following columns:
        - `type`: A string representing the feature type.
        - `center`: A sequence of three real numbers representing
           the 3D coordinates of the feature in some reference frame.
        - `radius`: A non-negative real number representing the feature radius.
           If not present, it will be added with a default value of 0.
    extra
        Optional dictionary to bundle additional information
        related to the pharmacophore, such as metadata or processing results.
        This is not used by this class, but can be useful for downstream processing.
    receptor
        Optional receptor associated with the pharmacophore.
        If provided, it is used by the `display()` method
        to visualize the pharmacophore in the context of the receptor structure.
    """
    def __init__(
        self,
        features: DataFrameLike,
        extra: dict[str, Any] | None = None,
        receptor: Receptor | None = None,
    ):
        super().__init__(features=features)
        self.extra = extra or {}
        self.receptor = receptor
        return

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        default_radius: float = 1.5,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
    ):
        nv = nglwidget or scishow.nglview.NGLWidget()
        if self.receptor:
            nv.add_trajectory(self.receptor)
        feature_colors = feature_colors or {}
        for feature_idx, feature in self.features.iterrows():
            nv.add_spheres(
                coords=feature["center"],
                radii=feature["radius"] or default_radius,
                name=f"{feature['type'].upper()}_{feature_idx}",
                colors=feature_colors.get(
                    feature["type"],
                    self._feature_colors.get(feature["type"], (0.5, 0.5, 0.5))
                ),
                representation_params=scishow.nglview.RepresentationParameters(
                    opacity=0.8,
                    visible=True,
                    lazy=True,
                )
            )
        return nv.display(gui=True)


def from_plip(
    pdb_files: str | bytes | Path | Sequence,
    ligands: Sequence[tuple[str, int | str, int]] | None = None,
    type_hbond_donor: str = "HD",
    type_hbond_acceptor: str = "OA",
    type_anion: str = "e-",
    type_cation: str = "e+",
    type_hydrophobic: str = "C",
    pocket: Pocket | None = None,
    receptor: Receptor | None = None,
):
    plip = caddpy.interaction.from_pdb(pdb_files, ligands=ligands)
    out = []
    for _, row in plip.all.iterrows():
        position_col = "l_position"
        match row["type"]:
            case "hbond":
                if row["r_is_d"]:
                    feature_type = type_hbond_acceptor
                else:
                    feature_type = type_hbond_donor
                    position_col = "h_position"
            case "water_bridge":
                position_col = "w_position"
                feature_type = type_hbond_acceptor if row["r_is_d"] else type_hbond_donor
            case "salt_bridge":
                feature_type = type_anion if row["r_is_cation"] else type_cation
            case "hydrophobic":
                feature_type = type_hydrophobic
            case _:
                continue
        position = row[position_col]
        # Sometimes a single atom can be involved
        # in multiple interactions of the same type, e.g., hydrophobic interactions with different residues.
        # Therefore, we only add a new feature if not already present
        for entry in out:
            if entry["type"] == feature_type and np.allclose(entry["center"], position):
                break
        else:
            out.append({"type": feature_type, "center": position})
    if pocket is not None:
        positions = np.stack([feature["center"] for feature in out])
        coverages = pocket.point_coverage(positions)
        out = [feature for feature, coverage in zip(out, coverages) if coverage]
    return LigandPharmacophore(features=out, extra={"plip": plip}, receptor=receptor)
