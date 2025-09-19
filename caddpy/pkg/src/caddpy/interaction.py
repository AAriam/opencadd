"""Analyze protein-ligand interactions using PLIP.

References
----------
- [PLIP documentation](https://github.com/pharmai/plip/blob/master/DOCUMENTATION.md)
"""

from pathlib import Path
import tempfile
from typing import Sequence, Literal, Any
import warnings

import numpy as np
import scishow
import scifile
from plip.structure.preparation import PDBComplex, PLInteraction
import plip.basic.config as plip_config
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot

from caddpy.chemsys import ChemicalSystem
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
    "w_o_position",
    "w_h_position",
]

_TUPLE_COLUMNS = [
    "l_serials",
    "r_serials",
    "h_serials",
    "w_o_serials",
    "w_h_serials",
]

_ALL_COLUMNS = [
    "instance",
    "type",

    "r_res_idx",
    "r_serials",
    "r_type",
    "r_position",
    "r_is_d",
    "r_is_cation",
    "is_sidechain",

    "l_res_idx",
    "l_serials",
    "l_type",
    "l_position",

    "w_res_idx",
    "w_o_serials",
    "w_o_position",
    "w_h_serials",
    "w_h_position",
    "w_angle",

    "h_serials",
    "h_position",

    "dist",
    "dist_a_h",
    "dist_a_d",
    "dist_w_a",
    "dist_w_d",

    "angle",
    "d_angle",
    "a_angle",

    "offset",
    "stack_type",
    "location",
    "rms",
    "geometry",
    "coordination_num",
    "complex_num",
]

class ProteinLigandInteractions:
    """Protein-ligand interactions.

    This class contains all interactions
    between a protein and a ligand, as detected by PLIP.
    It provides properties to access different types of interactions
    and a method to visualize them using NGLView.
    """

    def __init__(self, data: pd.DataFrame, complex: ChemicalSystem | None = None):
        def to_ndarray(x):
            if isinstance(x, np.ndarray):
                return x
            if pd.api.types.is_scalar(x) and pd.isna(x):
                return x
            return np.asarray(x)

        self._complex = complex

        for col in _ARRAY_COLUMNS:
            if col in data.columns:
                data[col] = data[col].apply(to_ndarray)
        for col in _TUPLE_COLUMNS:
            if col in data.columns:
                data[col] = data[col].apply(
                    lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
                )

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
    def complex(self) -> ChemicalSystem | None:
        """The chemical system associated with the interactions."""
        return self._complex

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
        instance: int | tuple[int, ...] | None = None,
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
        color_water_oxygen: tuple[float, float, float] = (0, 0.4, 1),
        color_water_hydrogen: tuple[float, float, float] = (0.4, 0.4, 1),
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
        add_complex: bool = True,
        gui: bool = True,
    ):
        args = locals()
        ngl = nglwidget or scishow.nglview.NGLWidget()
        if gui:
            ngl.display(gui=True)
        system = ngl.add_trajectory(self.complex, name=self.complex.name) if add_complex and self.complex is not None else None
        if not sphere_representation_params:
            sphere_representation_params = scishow.nglview.RepresentationParameters(
                opacity=0.6,
            )
        interaction_types = set(interactions_include) - set(interactions_exclude)
        for interaction_type in interaction_types:
            data = getattr(self, interaction_type)
            if data.empty:
                continue
            if instance is not None:
                data = data[data["instance"] == instance]

            receptor_selection = data["r_res_seq"].astype(str) + "^" + data["r_icode"] + ":" + data["r_chain_id"]
            system.add_ball_and_stick(" ".join(receptor_selection.unique()), name=interaction_type)
            if interaction_type == "water_bridge":
                water_selection = data["w_res_seq"].astype(str) + "^" + data["w_icode"] + ":" + data["w_chain_id"]
                system.add_ball_and_stick(" ".join(water_selection.unique()), name="water")

            # Set positions based on interaction type
            if interaction_type in ("hbond", "water_bridge"):
                l_pos = data['l_position'].where(data['r_is_d'], data['h_position'])
                r_pos = data['h_position'].where(data['r_is_d'], data['r_position'])
            else:
                l_pos = data["l_position"]
                r_pos = data["r_position"]
            l_pos = l_pos.to_list()
            r_pos = r_pos.to_list()
            w_o_pos = data["w_o_position"].to_list() if interaction_type == "water_bridge" else []
            w_h_pos = data["w_h_position"].to_list() if interaction_type == "water_bridge" else []

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
                    coords=w_o_pos,
                    colors=color_water_oxygen,
                    radii=radius_sphere,
                    name=f"{interaction_type}-water-oxygen",
                    representation_params=sphere_representation_params,
                )
                ngl.add_spheres(
                    coords=w_h_pos,
                    colors=color_water_hydrogen,
                    radii=radius_sphere,
                    name=f"{interaction_type}-water-hydrogen",
                    representation_params=sphere_representation_params,
                )
            if "line" in vis_spec:
                if interaction_type == "water_bridge":
                    acc_pos = data['l_position'].where(data['r_is_d'], data['r_position'])
                    don_pos = data['h_position']
                    for w_o, w_h, acc, don in zip(w_o_pos, w_h_pos, acc_pos, don_pos):
                        ngl.shape.add_arrow(w_o, don, color_line, radius_line, f"{interaction_type}-line-donor")
                        ngl.shape.add_arrow(w_h, acc, color_line, radius_line, f"{interaction_type}-line-acceptor")
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
            "Water Oxygen": [color_water_oxygen],
            "Water Hydrogen": [color_water_hydrogen],
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


def from_chemsys(
    complex: ChemicalSystem,
    add_polar_hydrogens: bool = False,
) -> ProteinLigandInteractions:
    """Calculate protein-ligand interactions in PDB file(s).

    Parameters
    ----------
    complex
        Chemical system containing the protein and ligand(s).
    add_polar_hydrogens
        Whether to add polar hydrogens to the structure before analysis.
        It is recommended to add hydrogens to the entire structure beforehand,
        as PLIP's hydrogen addition is quite basic and does not handle pH-dependant protonation states
        and tautomerism well.
    """
    globs = globals()

    # PLIP has a bug where if a residue has an insertion code,
    # it re-numbers the residue number since it can't handle insertion codes properly.
    # This results in reported atom serial numbers not matching those in the input PDB file.
    # Here we work around this by temporarily removing insertion codes and renumbering residues.
    atoms = complex.composition.atoms.copy()
    atoms["i_code"] = ""
    atoms["res_seq"] = atoms["res_idx"]
    complex = complex.new(composition=atoms)

    pdbs = complex.to_pdb(multimodel=False)
    if isinstance(pdbs, scifile.pdb.PDBFile):
        pdbs = np.array([pdbs], dtype=object)

    plip_config.NOHYDRO = not add_polar_hydrogens

    for idx, pdb in np.ndenumerate(pdbs):
        index = idx[0] if len(idx) == 1 else idx
        pdb_complex = PDBComplex()
        # The `as_string` argument for `load_pdb` does not work: https://github.com/pharmai/plip/issues/186
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb") as temp_file:
            temp_file.write(str(pdb))
            temp_file.flush()
            pdb_complex.load_pdb(temp_file.name)
        pdb_complex.analyze()
        interaction_sets = list(pdb_complex.interaction_sets.values())
        all_rows = []
        for attr_name in INTERACTION_TYPES:
            func = globs[f"_{attr_name}"]
            rows = func(interaction_sets)
            corrected_rows = [
                {
                    "instance": index,
                    **_correct_res(row, pdb.atom, complex.name, index)
                } for row in rows
            ]
            all_rows.extend(corrected_rows)
    df = pd.DataFrame(all_rows).convert_dtypes()
    df.rename(
        columns={
            f"{prefix}_res_seq": f"{prefix}_res_idx"
            for prefix in ("l", "r", "w") if f"{prefix}_res_seq" in df.columns
        },
        inplace=True,
    )
    df = df[[col for col in _ALL_COLUMNS if col in df.columns]]
    return ProteinLigandInteractions(df, complex)


def _correct_res(
    row: dict[str, Any],
    atom: pd.DataFrame,
    complex_name: str,
    complex_instance: int | tuple[int, ...]
) -> dict[str, Any]:
    """Correct residue numbers to match those in the PDB file."""
    def get_hydrogen_serial() -> int:
        prefix = "r" if row["r_is_d"] else "l"
        res_atoms = atom[(atom["res_seq"] == row[f"{prefix}_res_seq"])]
        dists = ((res_atoms[["x","y","z"]].values - row["h_position"]) ** 2).sum(axis=1)
        min_dist = dists.min()
        if min_dist > 1e-8:
            warnings.warn(
                f"Error in PLIP results for complex '{complex_name}' at instance {complex_instance}: "
                f"No matching hydrogen found for the reported interaction:\n{row}\n"
                f"The minimum distance is {min_dist} for the residue atoms:\n{res_atoms}"
            )
            return np.nan
        return int(res_atoms.iloc[dists.argmin()]["serial"])

    def complete_water():
        oxygen_serial = row["w_o_serials"][0]
        w_o_atoms = atom[atom["serial"] == oxygen_serial]
        if w_o_atoms.empty or not np.allclose(
            w_o_atoms.iloc[0][["x","y","z"]].to_numpy(),
            row["w_o_position"]
        ):
            dists = ((atom[["x","y","z"]].values - row["w_o_position"]) ** 2).sum(axis=1)
            min_dist = dists.min()
            if min_dist > 1e-8:
                raise ValueError(f"Water oxygen atom not found for position {row['w_o_position']}")
            w_o_atom = atom.iloc[dists.argmin()]
            row["w_o_serials"] = [int(w_o_atom["serial"])]
            if w_o_atoms.empty:
                serial_situation = "does not exist in the PDB file"
            else:
                position_based_on_serial = w_o_atoms.iloc[0][["x","y","z"]].to_numpy()
                serial_situation = f"corresponds to position {position_based_on_serial}"
            warnings.warn(
                f"Error in PLIP results for complex '{complex_name}' at instance {complex_instance}: "
                f"Mismatch in water oxygen position for the reported interaction:\n{row}\n"
                f"Position of the reported oxygen is {row['w_o_position']} "
                f"which matches the oxygen serial {w_o_atom['serial']}, "
                f"but the reported serial {oxygen_serial} {serial_situation}. "
                f"Changed to the matching serial {w_o_atom['serial']} in atom:\n{w_o_atom}"
            )
        else:
            w_o_atom = w_o_atoms.iloc[0]

        if w_o_atom["element"] != "O":
            raise ValueError(f"Water oxygen atom not found for serial {row['w_o_serials']}")

        row["w_res_seq"] = w_o_atom["res_seq"]

        water_atoms = atom[(atom["res_seq"] == row["w_res_seq"])]
        if len(water_atoms) < 3:
            raise ValueError(f"Less than 3 atoms found for water residue {row['w_res_name']} {row['w_chain_id']}{row['w_res_seq']}{row['w_icode']}")
        w_h_atoms = water_atoms[water_atoms["element"] == "H"]
        if len(w_h_atoms) != 2:
            raise ValueError(f"Expected 2 hydrogen atoms for water residue {row['w_res_name']} {row['w_chain_id']}{row['w_res_seq']}{row['w_icode']}, found {len(w_h_atoms)}")
        acceptor_pos = row["l_position"] if row["r_is_d"] else row["r_position"]
        dists = ((w_h_atoms[["x","y","z"]].values - acceptor_pos) ** 2).sum(axis=1)
        donor_hydrogen = w_h_atoms.iloc[dists.argmin()]
        row["w_h_serials"] = [int(donor_hydrogen["serial"])]
        row["w_h_position"] = np.stack(donor_hydrogen[["x","y","z"]].values)
        return

    typ = row["type"]

    for prefix in ("l", "r"):
        serials = row[f"{prefix}_serials"]
        atoms = atom[atom["serial"].isin(serials)]
        # Interactions with one atom (serial)
        if typ in ("hbond", "water_bridge", "hydrophobic", "halogen", "metal"):
            if atoms.empty or not np.allclose(row[f"{prefix}_position"], np.stack(atoms.iloc[0][["x","y","z"]])):
                dists = ((atom[["x","y","z"]].values - row[f"{prefix}_position"]) ** 2).sum(axis=1)
                min_dist = dists.min()
                if min_dist > 1e-8:
                    raise ValueError(
                        f"Position mismatch for {prefix}_position in {typ} interaction: "
                        f"{row[f'{prefix}_position']} not found in PDB atoms"
                    )
                correct_atom = atom.iloc[dists.argmin()]
                old_serial = serials[0]
                row[f"{prefix}_serials"] = serials = [int(correct_atom["serial"])]
                warnings.warn(
                    f"Error in PLIP results for complex '{complex_name}' at instance {complex_instance}: "
                    f"Mismatch in {prefix}_position for the reported interaction:\n{row}\n"


                    f"Expected {row[f'{prefix}_position']} but got {atoms.iloc[0][['x','y','z']].values if not atoms.empty else "N/A"}. "
                    f"Changed to serial {serials[0]}"
                )
                atoms = atom[atom["serial"].isin(serials)]
        # Interactions with multiple atoms
        else:
            if atoms.empty or np.linalg.norm(
                row[f"{prefix}_position"] - atoms[["x","y","z"]].mean(axis=0).to_numpy()
            ) > 1:
                row[f"{prefix}_serials"] = serials = [ser + 1 for ser in serials]
                atoms = atom[atom["serial"].isin(serials)]
                center = atoms[["x","y","z"]].mean(axis=0).to_numpy() if not atoms.empty else None
                if atoms.empty or np.linalg.norm(row[f"{prefix}_position"] - center) > 1:
                    raise ValueError(
                        f"Position mismatch for {prefix}_position in {typ} interaction: "
                        f"{row[f'{prefix}_position']} != {center if center is not None else 'N/A'}. "
                        f"Interaction: {row}"
                    )
                warnings.warn(
                    f"Mismatch in {prefix}_position in {typ} interaction for serials {serials}: "
                    f"Expected {row[f'{prefix}_position']} but got {center if center is not None else 'N/A'}. "
                    f"Changed to serials {serials}"
                )

        # Verify chain ID and residue name match
        for col in ("chain_id", "res_name", "res_seq"):
            values = atoms[col].unique()
            if len(values) != 1:
                raise ValueError(f"Multiple values found for {prefix}_{col} in {typ} interaction: {values}")
            orig_val = values[0]
            plip_val = row[f"{prefix}_{col}"]
            if orig_val != plip_val:
                raise ValueError(
                    f"Mismatch in {prefix}_{col} between PLIP and original PDB: {plip_val} != {orig_val}"
                )

    if typ in ("hbond", "water_bridge"):
        row["h_serials"] = [get_hydrogen_serial()]
    if typ == "water_bridge":
        complete_water()
    return row


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
                    "l_serials": [entry.a_orig_idx if entry.protisdon else entry.d_orig_idx],
                    "l_type": entry.atype if entry.protisdon else entry.dtype,
                    "l_position": np.array(entry.a.coords if entry.protisdon else entry.d.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": [entry.d_orig_idx if entry.protisdon else entry.a_orig_idx],
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
    # Plip only detects bridges where ligand and receptor have different roles
    # i.e., ligand is acceptor and receptor is donor, or vice versa.
    rows = []
    for interaction in interactions:
        for entry in interaction.water_bridges:
            rows.append(
                {
                    "type": "water_bridge",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serials": [entry.a_orig_idx if entry.protisdon else entry.d_orig_idx],
                    "l_type": entry.atype if entry.protisdon else entry.dtype,
                    "l_position": np.array(entry.a.coords if entry.protisdon else entry.d.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": [entry.d_orig_idx if entry.protisdon else entry.a_orig_idx],
                    "r_type": entry.dtype if entry.protisdon else entry.atype,
                    "r_position": np.array(entry.d.coords if entry.protisdon else entry.a.coords),

                    "r_is_d": entry.protisdon,

                    "w_o_serials": [entry.water_orig_idx],
                    "w_o_position": np.array(entry.water.coords),
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
                    "l_type": lig.fgroup,
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
                    "l_serials": [entry.ligatom_orig_idx],  # Carbon atom
                    "l_position": np.array(entry.ligatom.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": [entry.bsatom_orig_idx],   # Carbon atom
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
                    "l_type": 'aromatic' if entry.protcharged else entry.charge.fgroup,
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
            # PLIP only detects halogen bonds where the receptor is acceptor and ligand is donor.
            # See `plip.structure.preparation.PLInteraction.__init__` and `plip.structure.detection.halogen()`.
            rows.append(
                {
                    "type": "halogen",
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serials": [entry.don_orig_idx],
                    "l_type": entry.donortype,
                    "l_position": np.array(entry.don.x.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": [entry.acc_orig_idx],
                    "r_type": entry.acctype,
                    "r_position": np.array(entry.acc.o.coords),

                    "r_is_d": False,
                    "is_sidechain": entry.sidechain,
                    "dist": entry.distance,
                    "d_angle": entry.don_angle,
                    "a_angle": entry.acc_angle,
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
                    "l_serials": [entry.metal_orig_idx],
                    "l_type": entry.metal_type,
                    "l_position": np.array(entry.metal.coords),

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": [entry.target_orig_idx],
                    "r_type": entry.target_type,
                    "r_position": np.array(entry.target.atom.coords),

                    "dist": entry.distance,
                    "location": entry.location,
                    "rms": entry.rms,
                    "geometry": entry.geometry,
                    "coordination_num": entry.coordination_num,
                    "complex_num": entry.complexnum,
                }
            )
    return rows
