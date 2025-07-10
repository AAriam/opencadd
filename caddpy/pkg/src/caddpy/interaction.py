"""Analyze protein-ligand interactions using PLIP.

References
----------
- [PLIP documentation](https://github.com/pharmai/plip/blob/master/DOCUMENTATION.md)
"""

from pathlib import Path
import tempfile
from typing import Sequence, Literal, Any

import numpy as np
import scishow
from plip.structure.preparation import PDBComplex, PLInteraction
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot

from caddpy.typing import ArrayLike


INTERACTION_TYPES = (
    "hbond",
    "water_bridge",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "pi_cation",
    "halogen",
    "metal",
)
VISUALIZATION_PARTS = ("ligand", "receptor", "water", "line")

InteractionTypes = Sequence[Literal[*INTERACTION_TYPES]]
VisualizationParts = Sequence[Literal[*VISUALIZATION_PARTS]]

_ARRAY_COLUMNS = [
    "l_position",
    "r_position",
    "h_position",
    "w_position",
    "m_position",
    "t_position",
    "l_serials",
    "r_serials"
]

class ProteinLigandInteractions:
    """Protein-ligand interactions.

    This class contains all interactions
    between a protein and a ligand, as detected by PLIP.
    It provides properties to access different types of interactions
    and a method to visualize them using NGLView.
    """

    def __init__(self, data: pd.DataFrame):
        def to_ndarray(x):
            if isinstance(x, np.ndarray):
                return x
            if pd.api.types.is_scalar(x) and pd.isna(x):
                return x
            return np.asarray(x)

        for col in _ARRAY_COLUMNS:
            if col in data.columns:
                data[col] = data[col].apply(to_ndarray)

        self._all = data.convert_dtypes()
        for attr_name in INTERACTION_TYPES:
            subdf = (
                data[data["type"] == attr_name]
                .reset_index(drop=True)
                .dropna(axis=1, how="all")
            )
            setattr(self, f"_{attr_name}", subdf)
        return

    @property
    def all(self) -> pd.DataFrame:
        """All interactions combined."""
        return self._all

    @property
    def hbond(self) -> pd.DataFrame:
        """Hydrogen bonding interactions."""
        return self._hbond

    @property
    def water_bridge(self) -> pd.DataFrame:
        """Water bridge interactions."""
        return self._water_bridge

    @property
    def salt_bridge(self) -> pd.DataFrame:
        """Salt bridge interactions."""
        return self._salt_bridge

    @property
    def hydrophobic(self) -> pd.DataFrame:
        """Hydrophobic interactions."""
        return self._hydrophobic

    @property
    def pi_stacking(self) -> pd.DataFrame:
        """Pi stacking interactions."""
        return self._pi_stacking

    @property
    def pi_cation(self) -> pd.DataFrame:
        """Pi–cation interactions."""
        return self._pi_cation

    @property
    def halogen(self) -> pd.DataFrame:
        """Halogen interactions."""
        return self._halogen

    @property
    def metal(self) -> pd.DataFrame:
        """Metal interactions."""
        return self._metal

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        *,
        idx: int | tuple[int, ...] | None = None,
        interactions_include: InteractionTypes = INTERACTION_TYPES,
        interactions_exclude: InteractionTypes = (),
        vis: VisualizationParts | None = None,
        vis_hbond: VisualizationParts = VISUALIZATION_PARTS,
        vis_water_bridge: VisualizationParts = VISUALIZATION_PARTS,
        vis_salt_bridge: VisualizationParts = VISUALIZATION_PARTS,
        vis_hydrophobic: VisualizationParts = VISUALIZATION_PARTS,
        vis_pi_stacking: VisualizationParts = VISUALIZATION_PARTS,
        vis_pi_cation: VisualizationParts = VISUALIZATION_PARTS,
        vis_halogen: VisualizationParts = VISUALIZATION_PARTS,
        vis_metal: VisualizationParts = VISUALIZATION_PARTS,
        color_hbond_acceptor: tuple[float, float, float] = (0.6, 0, 0),
        color_hbond_donor: tuple[float, float, float] = (0, 0.6, 0),
        color_water: tuple[float, float, float] = (0, 0.4, 1),
        color_hydrophobic: tuple[float, float, float] = (1, 1, 0),
        color_aromatic: tuple[float, float, float] = (1, 0.6, 0),
        color_anion: tuple[float, float, float] = (1, 0, 0),
        color_cation: tuple[float, float, float] = (0, 0, 1),
        color_halogen_acceptor: tuple[float, float, float] = (0.8, 0.3, 0.2),
        color_halogen_donor: tuple[float, float, float] = (0.5, 0, 0.5),
        color_metal: tuple[float, float, float] = (0.4, 0.4, 0.4),
        color_metal_ligand: tuple[float, float, float] = (0, 1, 1),
        color_line: tuple[float, float, float] = (0.7, 0.7, 0.7),
        radius_sphere: float = 1,
        radius_line: float = 0.05,
        sphere_representation_params: scishow.nglview.RepresentationParameters | None = None,
    ):
        args = locals()
        ngl = nglwidget or scishow.nglview.NGLWidget()
        if not sphere_representation_params:
            sphere_representation_params = scishow.nglview.RepresentationParameters(
                opacity=0.7,
            )
        interaction_types = set(interactions_include) - set(interactions_exclude)
        for interaction_type in interaction_types:
            data = getattr(self, interaction_type)
            if data.empty:
                continue
            if idx is not None:
                data = data[data["idx"] == idx]

            # Set positions based on interaction type
            if interaction_type in ("hbond", "water_bridge"):
                l_pos = data['l_position'].where(data['r_is_d'], data['h_position'])
                r_pos = data['h_position'].where(data['r_is_d'], data['r_position'])
            elif interaction_type == "metal":
                l_pos = data["t_position"]
                r_pos = data["m_position"]
            else:
                l_pos = data["l_position"]
                r_pos = data["r_position"]
            l_pos = l_pos.to_list()
            r_pos = r_pos.to_list()
            w_pos = data["w_position"].to_list() if interaction_type == "water_bridge" else []

            # Set colors based on interaction type
            if interaction_type == "hydrophobic":
                l_color, r_color = color_hydrophobic, color_hydrophobic
            elif interaction_type in ("hbond", "water_bridge"):
                l_color = data["r_is_d"].apply(
                    lambda r_is_d: color_hbond_acceptor if r_is_d else color_hbond_donor
                ).to_list()
                r_color = data["r_is_d"].apply(
                    lambda r_is_d: color_hbond_donor if r_is_d else color_hbond_acceptor
                ).to_list()
            elif interaction_type == "salt_bridge":
                l_color = data["r_is_cation"].apply(
                    lambda r_is_cation: color_anion if r_is_cation else color_cation
                ).to_list()
                r_color = data["r_is_cation"].apply(
                    lambda r_is_cation: color_cation if r_is_cation else color_anion
                ).to_list()
            elif interaction_type == "pi_stacking":
                l_color, r_color = color_aromatic, color_aromatic
            elif interaction_type == "pi_cation":
                l_color = data["r_is_cation"].apply(
                    lambda r_is_cation: color_aromatic if r_is_cation else color_cation
                ).to_list()
                r_color = data["r_is_cation"].apply(
                    lambda r_is_cation: color_cation if r_is_cation else color_aromatic
                ).to_list()
            elif interaction_type == "halogen":
                l_color = color_halogen_donor
                r_color = color_halogen_acceptor
            elif interaction_type == "metal":
                l_color = color_metal_ligand
                r_color = color_metal
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

            vis_spec = vis or args[f"vis_{interaction_type}"]
            if "ligand" in vis_spec:
                ngl.add_spheres(
                    coords=l_pos,
                    colors=l_color,
                    radii=radius_sphere,
                    name=f"{interaction_type}-ligand",
                    representation_params=sphere_representation_params,
                )
            if "receptor" in vis_spec:
                ngl.add_spheres(
                    coords=r_pos,
                    colors=r_color,
                    radii=radius_sphere,
                    name=f"{interaction_type}-receptor",
                    representation_params=sphere_representation_params,
                )
            if interaction_type == "water_bridge" and "water" in vis_spec:
                ngl.add_spheres(
                    coords=w_pos,
                    colors=color_water,
                    radii=radius_sphere,
                    name=f"{interaction_type}-water",
                    representation_params=sphere_representation_params,
                )
            if "line" in vis_spec:
                if interaction_type == "water_bridge":
                    for w, l, r in zip(w_pos, l_pos, r_pos):
                        ngl.shape.add_arrow(w, l, color_line, radius_line, f"{interaction_type}-line")
                        ngl.shape.add_arrow(w, r, color_line, radius_line, f"{interaction_type}-line")
                else:
                    for l, r in zip(l_pos, r_pos):
                        ngl.shape.add_arrow(l, r, color_line, radius_line, f"{interaction_type}-line")
        # Color legend
        fig, _ = mpl.pyplot.subplots(nrows=2, ncols=6, figsize=(12, 1))
        mpl.pyplot.subplots_adjust(hspace=1)
        fig.suptitle("Interaction Colors", size=10, y=1.3)
        color_map = {
            "H-Bond Acceptor": [color_hbond_acceptor],
            "H-Bond Donor": [color_hbond_donor],
            "Water": [color_water],
            "Hydrophobic": [color_hydrophobic],
            "Aromatic": [color_aromatic],
            "Anion": [color_anion],
            "Cation": [color_cation],
            "Halogen Acceptor": [color_halogen_acceptor],
            "Halogen Donor": [color_halogen_donor],
            "Metal Center": [color_metal],
            "Metal Ligand": [color_metal_ligand],
            "Line": [color_line],
        }
        for ax, (interaction, color) in zip(fig.axes, color_map.items()):
            ax.imshow(np.zeros((1, 5)), cmap=mpl.colors.ListedColormap(color))
            ax.set_title(interaction, loc="center", fontsize=9)
            ax.set_axis_off()
        return ngl, fig


def from_pdb(
    files: str | bytes | Path | ArrayLike,
    ligands: Sequence[tuple[str, int | str, int]] | None = None,
) -> ProteinLigandInteractions:
    """Calculate protein-ligand interactions in PDB file(s).

    Parameters
    ----------
    files
        PDB file(s) containing the protein-ligand complex.
        This can be a single file or an array of files with any shape.
        Each entry can be either a PDB file content (as string or bytes)
        or a path to a PDB file (as a `pathlib.Path` object).
        If an array of files is provided,
        each interaction will have an additional column `idx`
        indicating the index of the file in the array
        (as a single integer for 1D or a tuple of integers for multi-dimensional arrays).
    ligands
        Ligand identifiers to filter interactions.
        If not provided, all ligands in the PDB file will be considered.
        Each ligand is identified by a tuple of three elements:
        1. Residue name (e.g., "ATP")
        2. Residue chain ID (e.g., "A")
        3. Residue sequence number (e.g., 1)

        For each ligand, you can provide the first n elements of the tuple,
        in which case the ligand will be matched against all ligands
        with the same specifications.
    """
    globs = globals()
    if isinstance(files, str | bytes | Path):
        files = [files]
        is_multifile = False
    else:
        is_multifile = True
    files = np.array(files, dtype=object)
    for file_idx in np.ndindex(files.shape):
        file = files[file_idx]
        pdb_complex = PDBComplex()
        # The `as_string` argument for `load_pdb` does not work: https://github.com/pharmai/plip/issues/186
        if isinstance(file, Path):
            pdb_complex.load_pdb(str(file))
        else:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb") as temp_file:
                pdb_str = file if isinstance(file, str) else file.decode("utf-8")
                temp_file.write(pdb_str)
                temp_file.flush()
                pdb_complex.load_pdb(temp_file.name)
        pdb_complex.analyze()

        interaction_sets = []
        for ligand_name, interaction_set in pdb_complex.interaction_sets.items():
            if not ligands:
                interaction_sets.append(interaction_set)
                continue
            ligand_name_parts = ligand_name.split(":")
            for ligand in ligands:
                if all(
                    ligand_part_input == ligand_part_plip
                    for ligand_part_input, ligand_part_plip in zip(ligand, ligand_name_parts)
                ):
                    interaction_sets.append(interaction_set)
                    break

        all_rows = []
        for attr_name in INTERACTION_TYPES:
            func = globs[f"_{attr_name}"]
            rows = func(interaction_sets)
            if is_multifile:
                idx = file_idx[0] if len(file_idx) == 1 else file_idx
                rows = [{"idx": idx, **row} for row in rows]
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows).convert_dtypes()
    return ProteinLigandInteractions(df)


def _hbond(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract hydrogen bond interaction data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in [*interaction.hbonds_pdon, *interaction.hbonds_ldon]:
            rows.append(
                {
                    "type": "hbond",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serial": entry.a_orig_idx if entry.protisdon else entry.d_orig_idx,
                    "l_type": entry.atype if entry.protisdon else entry.dtype,
                    "l_position": np.array(entry.a.coords if entry.protisdon else entry.d.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serial": entry.d_orig_idx if entry.protisdon else entry.a_orig_idx,
                    "r_type": entry.dtype if entry.protisdon else entry.atype,
                    "r_position": np.array(entry.d.coords if entry.protisdon else entry.a.coords),

                    "r_is_d": entry.protisdon,
                    "is_sidechain": entry.sidechain,
                    "h_position": np.array(entry.h.coords),
                    "dist_a_h": entry.distance_ah,
                    "dist_a_d": entry.distance_ad,
                    "angle": entry.angle,
                }
            )
    return rows


def _water_bridge(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract water bridge interaction data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in interaction.water_bridges:
            rows.append(
                {
                    "type": "water_bridge",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serial": entry.a_orig_idx if entry.protisdon else entry.d_orig_idx,
                    "l_type": entry.atype if entry.protisdon else entry.dtype,
                    "l_position": np.array(entry.a.coords if entry.protisdon else entry.d.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serial": entry.d_orig_idx if entry.protisdon else entry.a_orig_idx,
                    "r_type": entry.dtype if entry.protisdon else entry.atype,
                    "r_position": np.array(entry.d.coords if entry.protisdon else entry.a.coords),

                    "r_is_d": entry.protisdon,

                    "w_serial": entry.water_orig_idx,
                    "w_position": np.array(entry.water.coords),
                    'w_angle': entry.w_angle,

                    "h_position": np.array(entry.h.coords),

                    'dist_w_a': entry.distance_aw,
                    'dist_w_d': entry.distance_dw,
                    "d_angle": entry.d_angle,
                }
            )
    return rows


def _salt_bridge(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract salt bridge interaction data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for sb in [*interaction.saltbridge_lneg, *interaction.saltbridge_pneg]:
            lig, prot = (sb.negative, sb.positive) if sb.protispos else (sb.positive, sb.negative)
            rows.append(
                {
                    "type": "salt_bridge",
                    "l_res_name": sb.restype_l,
                    "l_res_seq": sb.resnr_l,
                    "l_chain_id": sb.reschain_l,
                    "l_group": lig.fgroup,
                    "l_serials": np.array(lig.atoms_orig_idx),
                    "l_position": np.array(lig.center),

                    "r_res_name": sb.restype,
                    "r_res_seq": sb.resnr,
                    "r_chain_id": sb.reschain,
                    "r_serials": np.array(prot.atoms_orig_idx),
                    "r_position": np.array(prot.center),

                    "r_is_cation": sb.protispos,
                    "dist": sb.distance,
                }
            )
    return rows


def _hydrophobic(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract hydrophobic interaction data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in interaction.hydrophobic_contacts:
            rows.append(
                {
                    "type": "hydrophobic",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serial": entry.ligatom_orig_idx,  # Carbon atom
                    "l_position": np.array(entry.ligatom.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serial": entry.bsatom_orig_idx,   # Carbon atom
                    "r_position": np.array(entry.bsatom.coords),

                    "dist": entry.distance,
                }
            )
    return rows


def _pi_stacking(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract pi-stacking interaction data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in interaction.pistacking:
            rows.append(
                {
                    "type": "pi_stacking",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serials": np.array(entry.ligandring.atoms_orig_idx),
                    "l_position": np.array(entry.ligandring.center),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": np.array(entry.proteinring.atoms_orig_idx),
                    "r_position": np.array(entry.proteinring.center),

                    "dist": entry.distance,
                    "angle": entry.angle,
                    "offset": entry.offset,
                    "stack_type": entry.type,
                }
            )
    return rows


def _pi_cation(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract pi-cation interaction data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in [*interaction.pication_laro, *interaction.pication_paro]:
            lig, prot = (entry.ring, entry.charge) if entry.protcharged else (entry.charge, entry.ring)
            rows.append(
                {
                    "type": "pi_cation",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serials": np.array(lig.atoms_orig_idx),
                    "l_group": 'aromatic' if entry.protcharged else entry.charge.fgroup,
                    "l_position": np.array(lig.center),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": np.array(prot.atoms_orig_idx),
                    "r_position": np.array(prot.center),

                    "r_is_cation": entry.protcharged,
                    "dist": entry.distance,
                    "offset": entry.offset,
                }
            )
    return rows


def _halogen(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract halogen bond data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in interaction.halogen_bonds:
            rows.append(
                {
                    "type": "halogen",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_position": np.array(entry.don.x.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_position": np.array(entry.acc.o.coords),

                    "is_sidechain": entry.sidechain,
                    "dist": entry.distance,
                    "d_angle": entry.don_angle,
                    'a_angle': entry.acc_angle,
                    'd_serial': entry.don_orig_idx,
                    "d_type": entry.donortype,
                    'a_serial': entry.acc_orig_idx,
                    "a_type": entry.acctype,
                }
            )
    return rows


def _metal(interactions: Sequence[PLInteraction]) -> list[dict[str, Any]]:
    """Extract metal complex data from PLIP interaction objects."""
    rows = []
    for interaction in interactions:
        for entry in interaction.metal_complexes:
            rows.append(
                {
                    "type": "metal",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,

                    "m_serial": entry.metal_orig_idx,
                    "m_type": entry.metal_type,
                    "m_position": np.array(entry.metal.coords),

                    't_serial': entry.target_orig_idx,
                    't_type': entry.target_type,
                    't_position': np.array(entry.target.atom.coords),

                    "dist": entry.distance,
                    "location": entry.location,
                    "rms": entry.rms,
                    "geometry": entry.geometry,
                    'coordination_num': entry.coordination_num,
                    'complex_num': entry.complexnum,
                }
            )
    return rows
