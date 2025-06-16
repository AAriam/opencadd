"""Analyze protein-ligand interactions."""

from pathlib import Path
import logging
import tempfile
from typing import Sequence, Literal

import scishow
from plip.structure.preparation import PDBComplex, PLInteraction
import pandas as pd

INTERACTION_TYPES = (
    "hydrophobic",
    "hbond",
    "water_bridge",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
    "halogen",
    "metal",
)
VISUALIZATION_PARTS = ("ligand", "receptor", "water", "line")

InteractionTypes = Sequence[Literal[*INTERACTION_TYPES]]
VisualizationParts = Sequence[Literal[*VISUALIZATION_PARTS]]


class ProteinLigandInteractions:

    def __init__(
        self,
        hydrophobic: pd.DataFrame,
        hbond: pd.DataFrame,
        water_bridge: pd.DataFrame,
        salt_bridge: pd.DataFrame,
        pi_stacking: pd.DataFrame,
        pi_cation: pd.DataFrame,
        halogen: pd.DataFrame,
        metal: pd.DataFrame,
    ):
        self._hydrophobic = hydrophobic
        self._hbond = hbond
        self._water_bridge = water_bridge
        self._salt_bridge = salt_bridge
        self._pi_stacking = pi_stacking
        self._pi_cation = pi_cation
        self._halogen = halogen
        self._metal = metal
        self._all = None
        return

    @property
    def all(self) -> pd.DataFrame:
        """All interactions combined."""
        if self._all is not None:
            return self._all
        self._all = pd.concat(
            [
                self.hydrophobic,
                self.hbond,
                self.water_bridge,
                self.salt_bridge,
                self.pi_stacking,
                self.pi_cation,
                self.halogen,
                self.metal,
            ],
            ignore_index=True,
        )
        return self._all

    @property
    def hydrophobic(self) -> pd.DataFrame:
        """Hydrophobic interactions."""
        return self._hydrophobic

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
        return ngl



def from_pdb(
    file: str | bytes | Path,
    ligands: Sequence[tuple[str, int | str, int]] | None = None,
) -> ProteinLigandInteractions:
    """Calculate protein-ligand interactions in a PDB file.

    Parameters
    ----------
    pdb_filepath : str or pathlib.Path
        Filepath of the PDB file containing the protein-ligand complex.

    Returns
    -------
    dict of dict
        Dictionary of all different interaction data for all detected ligands.
        - The keys of first dictionary correspond to the ligand-IDs of detected ligands in the
          PDB file.
        - The keys of each sub-dictionary correspond to interaction types, as defined in
          `PLIP.Consts.InteractionTypes`.
    """
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


    ligands = [":".join(map(str, ligand)) for ligand in ligands] if ligands else None
    interaction_sets = [
        interaction_set for ligand_name, interaction_set in pdb_complex.interaction_sets.items()
        if not ligands or ligand_name in ligands
    ]
    interaction_data = {}
    for attr_name, func in zip(
        INTERACTION_TYPES,
        (_hydrophobic, _hbond, _water_bridge, _salt_bridge, _pi_stacking, _pi_cation, _halogen, _metal),
    ):
        interaction_data[attr_name] = func(interaction_sets)
    return ProteinLigandInteractions(**interaction_data)


def _hydrophobic(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in interaction.hydrophobic_contacts:
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serial": entry.ligatom_orig_idx,  # Carbon atom
                    "l_position": entry.ligatom.coords,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serial": entry.bsatom_orig_idx,   # Carbon atom
                    "r_position": entry.bsatom.coords,

                    "dist": entry.distance,
                }
            )
    return pd.DataFrame(rows)


def _hbond(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in [*interaction.hbonds_pdon, *interaction.hbonds_ldon]:
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serial": entry.a_orig_idx if entry.protisdon else entry.d_orig_idx,
                    "l_type": entry.atype if entry.protisdon else entry.dtype,
                    "l_position": entry.a.coords if entry.protisdon else entry.d.coords,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serial": entry.d_orig_idx if entry.protisdon else entry.a_orig_idx,
                    "r_type": entry.dtype if entry.protisdon else entry.atype,
                    "r_position": entry.d.coords if entry.protisdon else entry.a.coords,

                    "r_is_d": entry.protisdon,
                    "is_sidechain": entry.sidechain,
                    "h_position": entry.h.coords,
                    "dist_a_h": entry.distance_ah,
                    "dist_a_d": entry.distance_ad,
                    "angle": entry.angle,
                }
            )
    return pd.DataFrame(rows)


def _water_bridge(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in interaction.water_bridges:
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serial": entry.a_orig_idx if entry.protisdon else entry.d_orig_idx,
                    "l_type": entry.atype if entry.protisdon else entry.dtype,
                    "l_position": entry.a.coords if entry.protisdon else entry.d.coords,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serial": entry.d_orig_idx if entry.protisdon else entry.a_orig_idx,
                    "r_type": entry.dtype if entry.protisdon else entry.atype,
                    "r_position": entry.d.coords if entry.protisdon else entry.a.coords,

                    "r_is_d": entry.protisdon,

                    "w_serial": entry.water_orig_idx,
                    "w_position": entry.water.coords,
                    'w_angle': entry.w_angle,

                    "h_position": entry.h.coords,

                    'dist_w_a': entry.distance_aw,
                    'dist_w_d': entry.distance_dw,
                    "d_angle": entry.d_angle,
                }
            )
    return pd.DataFrame(rows)


def _salt_bridge(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for sb in [*interaction.saltbridge_lneg, *interaction.saltbridge_pneg]:
            lig, prot = (sb.negative, sb.positive) if sb.protispos else (sb.positive, sb.negative)
            rows.append(
                {
                    "l_res_name": sb.restype_l,
                    "l_res_seq": sb.resnr_l,
                    "l_chain_id": sb.reschain_l,
                    "l_group": lig.fgroup,
                    "l_serials": lig.atoms_orig_idx,
                    "l_position": lig.center,

                    "r_res_name": sb.restype,
                    "r_res_seq": sb.resnr,
                    "r_chain_id": sb.reschain,
                    "r_serials": prot.atoms_orig_idx,
                    "r_position": prot.center,

                    "r_is_cation": sb.protispos,
                    "dist": sb.distance,
                }
            )
    return pd.DataFrame(rows)


def _pi_stacking(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in interaction.pistacking:
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serials": entry.ligandring.atoms_orig_idx,
                    "l_position": entry.ligandring.center,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": entry.proteinring.atoms_orig_idx,
                    "r_position": entry.proteinring.center,

                    "dist": entry.distance,
                    "angle": entry.angle,
                    "offset": entry.offset,
                    "type": entry.type,
                }
            )
    return pd.DataFrame(rows)


def _pi_cation(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in [*interaction.pication_laro, *interaction.pication_paro]:
            lig, prot = (entry.ring, entry.charge) if entry.protcharged else (entry.charge, entry.ring)
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_serials": lig.atoms_orig_idx,
                    "l_group": 'aromatic' if entry.protcharged else entry.charge.fgroup,
                    "l_position": lig.center,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_serials": prot.atoms_orig_idx,
                    "r_position": prot.center,

                    "r_is_cation": entry.protcharged,
                    "dist": entry.distance,
                    "offset": entry.offset,
                }
            )
    return pd.DataFrame(rows)


def _halogen(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in interaction.halogen_bonds:
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,
                    "l_position": entry.don.x.coords,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,
                    "r_position": entry.acc.o.coords,

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
    return pd.DataFrame(rows)


def _metal(interactions: Sequence[PLInteraction]):
    rows = []
    for interaction in interactions:
        for entry in interaction.metal_complexes:
            rows.append(
                {
                    "l_res_name": entry.restype_l,
                    "l_res_seq": entry.resnr_l,
                    "l_chain_id": entry.reschain_l,

                    "r_res_name": entry.restype,
                    "r_res_seq": entry.resnr,
                    "r_chain_id": entry.reschain,

                    "m_serial": entry.metal_orig_idx,
                    "m_type": entry.metal_type,
                    "m_position": entry.metal.coords,

                    't_serial': entry.target_orig_idx,
                    't_type': entry.target_type,
                    't_position': entry.target.atom.coords,

                    "dist": entry.distance,
                    "location": entry.location,
                    "rms": entry.rms,
                    "geometry": entry.geometry,
                    'coordination_num': entry.coordination_num,
                    'complex_num': entry.complexnum,
                }
            )
    return pd.DataFrame(rows)
