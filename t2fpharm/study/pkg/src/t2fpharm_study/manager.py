from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
from typing import Any, Sequence, Literal

import arrayer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import t2fpharm.pharmacophore_receptor
from tqdm.auto import tqdm

import pyserials
import pkgdata

import sciapi
import scifile
import caddpy

import t2fpharm


class Manager:
    def __init__(
        self,
        dataset: pd.DataFrame,
        pocket_inputs: dict,
        field_inputs: dict,
        group_color: dict[str, dict[str, str]],
        dirpath_data: Path,
        dirpath_pdb_raw: Path | str = "structure/1-pdb-raw",
        dirpath_pdb_fixed: Path | str = "structure/2-pdb-fixed",
        dirpath_pdb_aligned: Path | str = "structure/3-pdb-aligned",
        dirpath_pdb_apo: Path | str = "structure/4-pdb-apo",
        dirpath_pdbqt: Path | str = "structure/5-pdbqt",
        dirpath_affinity: Path | str = "affinity",
        dirpath_pocket: Path | str = "pocket",
        dirpath_autogrid: Path | str = "autogrid",
        dirpath_field: Path | str = "field",
        dirpath_ligand_plip: Path | str = "ligand/plip",
        dirpath_ligand_features: Path | str = "ligand/features",
    ):
        self._data = dataset
        self.pocket_params = pocket_inputs
        self.field_params = field_inputs
        self._group_color = group_color
        self.dirpath_data = Path(dirpath_data)
        self._dirpath = {
            "pdb_raw": dirpath_pdb_raw,
            "pdb_fixed": dirpath_pdb_fixed,
            "pdb_aligned": dirpath_pdb_aligned,
            "pdb_apo": dirpath_pdb_apo,
            "pdbqt": dirpath_pdbqt,
            "affinity": dirpath_affinity,
            "pocket": dirpath_pocket,
            "autogrid": dirpath_autogrid,
            "field": dirpath_field,
            "ligand_plip": dirpath_ligand_plip,
            "ligand_features": dirpath_ligand_features,
        }
        self._file_ext = {
            "pdb_raw": "pdb",
            "pdb_fixed": "pdb",
            "pdb_aligned": "pdb",
            "pdb_apo": "pdb",
            "pdbqt": "pdbqt",
            "affinity": "json",
            "pocket": "yaml",
            "field": "json",
            "ligand_plip": "json",
            "ligand_features": "json",
        }
        self._pdb = None
        self._cache_enabled = True
        self._cache = {}
        return

    @property
    def dataset(self) -> pd.DataFrame:
        """Dataset as a pandas DataFrame."""
        return self._data

    @property
    def pdb_all(self) -> scifile.pdb.PDBDataset:
        class StyledDF(pd.DataFrame):
            @property
            def _constructor(self):
                return self.__class__

            def _repr_html_(self2):
                def color_groups(group_ids: pd.Series) -> list[str]:
                    return [
                        f"background-color: {self._group_color[group_id]["bg"] or '#ffffff'}; color: {self._group_color[group_id]["text"] or '#000000'}"
                        for group_id in group_ids
                    ]
                def color_is_ref(is_ref: pd.Series) -> list[str]:
                    return [
                        "background-color: rgb(0 200 0); color: #000000" if ref else ""
                        for ref in is_ref
                    ]
                out = self2.style
                if "group_id" in out.columns:
                    out = out.apply(color_groups, subset='group_id')
                if "is_ref" in out.columns:
                    out = out.apply(color_is_ref, subset='is_ref')
                return out._repr_html_()

        if self._pdb:
            return self._pdb
        pdb = scifile.pdb.merge(self._data["pdb_raw"].values)
        get_group_id = lambda pdb_id: self.data.loc[pdb_id]["group_id"]
        is_ref_structure = lambda pdb_id: self.data.loc[pdb_id]["is_ref"]
        styled_params = {}
        for record in (
            "header",
            "obslte",
            "title",
            "split",
            "caveat",
            "compnd",
            "source",
            "keywds",
            "expdta",
            "nummdl",
            "mdltyp",
            "author",
            "revdat",
            "sprsde",
            "jrnl",
            "dbref",
            "seqadv",
            "seqres",
            "modres",
            "het",
            "hetnam",
            "helix",
            "sheet",
            "ssbond",
            "link",
            "cispep",
            "site",
            "cryst1",
            "origx",
            "scale",
            "mtrix",
            "atom",
            "anisou",
            "ter",
            "conect",
        ):
            df = getattr(pdb, record)
            if df is None:
                continue
            group_ids = df["id_code"].map(get_group_id)
            is_ref = df["id_code"].map(is_ref_structure)
            df.insert(0, "group_id", group_ids)
            df.insert(1, "is_ref", is_ref)
            styled_params[record] = StyledDF(df)
        if hasattr(pdb, "remark"):
            remark_params = {}
            for attr_name in ("full_text", "related_publications", "resolution", "format"):
                df = getattr(pdb.remark, attr_name)
                if df is None:
                    continue
                group_ids = df["id_code"].map(get_group_id)
                is_ref = df["id_code"].map(is_ref_structure)
                df.insert(0, "group_id", group_ids)
                df.insert(1, "is_ref", is_ref)
                init_param_name = attr_name if attr_name != "full_text" else "full"
                remark_params[init_param_name] = StyledDF(df)
            styled_params["remark"] = scifile.pdb.records.RemarkDataset(**remark_params)
        self._pdb = scifile.pdb.PDBDataset(**styled_params)
        return self._pdb

    def load(self):
        self.caching(enabled=True)
        for _, entry in tqdm(
            self.dataset.iterrows(),
            total=len(self.dataset),
            desc="Loading data",
            unit="job",
        ):
            pdb_id = entry["pdb_id"]
            self.pdb_raw(pdb_id)
            self.complex(pdb_id)
            self.receptor(pdb_id)
            self.pdbqt(pdb_id)
            self.pocket(pdb_id)
            self.field(pdb_id)
            self.modeler(pdb_id)
            self.ligand_pharmacophore(pdb_id)
            self.affinity(pdb_id)
        return

    def caching(self, enabled: bool = True):
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._cache = {}
        return

    def run(self, jobs: Sequence[dict[str, Any]]):
        # Prepare jobs
        full_jobs = []
        caching_was_enabled = self._cache_enabled
        self.caching(enabled=True)
        for _, entry in tqdm(
            self.dataset.iterrows(),
            total=len(self.dataset),
            desc="Creating jobs",
            unit="job",
        ):
            pdb_id = entry["pdb_id"]
            modeler = self.modeler(pdb_id)
            ligand_pharm = self.ligand_pharmacophore(pdb_id)
            if not caching_was_enabled:
                self._cache = {}
            for job_idx, job in enumerate(jobs):
                job_inputs = job | {
                    "pdb_id": pdb_id,
                    "job_idx": job_idx,
                    "modeler": modeler,
                    "ligand_pharm": ligand_pharm,
                }
                full_jobs.append(job_inputs)
        self.caching(enabled=caching_was_enabled)
        # Execute jobs
        outputs: dict[tuple[str, int], dict[str, t2fpharm.pharmacophore_receptor.ReceptorPharmacophore | pd.DataFrame]] = {}
        statistics: list[dict] = []

        for job_inputs in tqdm(
            full_jobs,
            total=len(full_jobs),
            desc="Running jobs",
            unit="job",
        ):
            pharm, matches, stats = _run_job(**job_inputs)
            outputs[(stats["pdb_id"], stats["method"], stats["job_idx"])] = {
                "pharm": pharm,
                "matches": matches,
            }
            statistics.append(stats)

        # with ProcessPoolExecutor() as exe:
        #     futures = [exe.submit(_run_job, **job_inputs) for job_inputs in full_jobs]
        #     for future in tqdm(
        #         as_completed(futures),
        #         total=len(futures),
        #         desc="Running jobs",
        #         unit="job",
        #     ):
        #         pharm, matches, stats = future.result()
        #         outputs[(stats["pdb_id"], stats["method"], stats["job_idx"])] = {
        #             "pharm": pharm,
        #             "matches": matches,
        #         }
        #         statistics.append(stats)
        stats_df = pd.DataFrame(statistics).convert_dtypes()
        return stats_df, outputs

    def pdb_raw(self, pdb_id: str) -> scifile.pdb.PDBFile:
        cached = self._cache.get(pdb_id, {}).get("pdb_raw")
        if cached:
            return cached
        filepath_pdb_raw = self.filepath(pdb_id, "pdb_raw")
        if filepath_pdb_raw.is_file():
            pdb = scifile.pdb.read(filepath_pdb_raw)
        else:
            pdb_raw_bytes = sciapi.pdb.file.entry(pdb_id=pdb_id, file_format="pdb")
            filepath_pdb_raw.write_bytes(pdb_raw_bytes)
            pdb = scifile.pdb.read(filepath_pdb_raw)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pdb_raw"] = pdb
        return pdb

    def complex(self, pdb_id: str) -> t2fpharm.receptor.Receptor:
        cached = self._cache.get(pdb_id, {}).get("complex")
        if cached:
            return cached
        filepath_pdb_aligned = self.filepath(pdb_id, "pdb_aligned")
        if filepath_pdb_aligned.is_file():
            pdb_aligned_str = filepath_pdb_aligned.read_text()
        else:
            (
                pdb_fixed_str,
                missing_residues,
                nonstandard_residues,
                missing_atoms,
                missing_terminals
            ) = caddpy.chemsys.fix_pdb(
                file=self.filepath(pdb_id, "pdb_raw"),
                keep_chain_ids=self.dataset.loc[pdb_id, "chain_id"],
                add_missing_residues=True,
                replace_nonstandard_residues=False,
                add_missing_heavy_atoms=True,
                add_missing_atoms_seed=42,
                add_missing_hydrogens=7.0,
                keep_ids=True,
            )
            self.filepath(pdb_id, "pdb_fixed").write_text(pdb_fixed_str)
            if self.dataset.loc[pdb_id, "is_ref"]:
                pdb_aligned_str = pdb_fixed_str
            else:
                group_id = self.dataset.loc[pdb_id, "group_id"]
                is_group_ref = (self.dataset["group_id"] == group_id) & self.dataset["is_ref"]
                ref_pdb_id = self.dataset.loc[is_group_ref, "pdb_id"].iloc[0]
                complex_aligned = _align_to_ref_structure(
                    ref_complex=self.complex(ref_pdb_id),
                    query_complex=t2fpharm.receptor.from_pdb(pdb_fixed_str),
                    ref_chain_id=self.dataset.loc[ref_pdb_id, "chain_id"],
                    query_chain_id=self.dataset.loc[pdb_id, "chain_id"],
                    ref_pocket=self.pocket(ref_pdb_id),
                )
                pdb_aligned_str = str(complex_aligned.to_pdb())
                filepath_pdb_aligned.write_text(pdb_aligned_str)
        rcomplex = t2fpharm.receptor.from_pdb(pdb_aligned_str)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["complex"] = rcomplex
        return rcomplex

    def receptor(self, pdb_id: str) -> t2fpharm.receptor.Receptor:
        cached = self._cache.get(pdb_id, {}).get("receptor")
        if cached:
            return cached
        filepath_pdb_apo = self.filepath(pdb_id, "pdb_apo")
        if filepath_pdb_apo.is_file():
            receptor = t2fpharm.receptor.from_pdb(filepath_pdb_apo)
        else:
            rcomplex = self.complex(pdb_id)
            receptor = rcomplex.select(rcomplex.composition.atoms["res_poly"])
            filepath_pdb_apo.write_text(str(receptor.to_pdb()))
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["receptor"] = receptor
        return receptor

    def pdbqt(self, pdb_id: str) -> str:
        cached = self._cache.get(pdb_id, {}).get("pdbqt")
        if cached:
            return cached
        filepath_pdbqt = self.filepath(pdb_id, "pdbqt")
        if filepath_pdbqt.is_file():
            pdbqt_str = filepath_pdbqt.read_text()
        else:
            pdbqt_str = self.receptor(pdb_id).to_pdbqt(
                autobond=False,
                rigid=True,
                combine=False,
                flexible=False,
                preserve_serials=True,
                preserve_hydrogens=False,
                preserve_names=True,
                charge_model="gasteiger",
                add_hydrogens=False,
            )
            filepath_pdbqt.write_text(pdbqt_str)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pdbqt"] = pdbqt_str
        return pdbqt_str

    def pocket(self, pdb_id: str) -> t2fpharm.pocket.Pocket:
        """Prepare the pocket for the structure."""
        cached = self._cache.get(pdb_id, {}).get("pocket")
        if cached:
            return cached
        filepath_pocket = self.filepath(pdb_id, "pocket")
        rcomplex = self.complex(pdb_id)
        if filepath_pocket.is_file():
            pocket_data = pyserials.read.yaml_from_file(filepath_pocket)
            pocket = t2fpharm.pocket.from_data(**pocket_data, receptor=rcomplex)
        else:
            atoms = rcomplex.composition.atoms
            ligand_res_name = self.dataset.loc[pdb_id, "ligand_res_name"]
            ligand_chain_id = self.dataset.loc[pdb_id, "ligand_chain_id"]
            ligand_res_seq = self.dataset.loc[pdb_id, "ligand_res_seq"]
            ligand_mask = (
                (atoms["res_name"] == ligand_res_name) &
                (atoms["chain_id"] == ligand_chain_id) &
                (atoms["res_seq"] == ligand_res_seq)
            )
            if ligand_mask.sum() == 0:
                raise ValueError(
                    f"Ligand {ligand_res_name} "
                    f"not found in structure {pdb_id}"
                )
            pocket = t2fpharm.pocket.from_ligand(
                system=rcomplex,
                ligand_mask=ligand_mask,
                ligand_radii=None,
                grid=self.pocket_params["grid_spacing"],
                ligand_radii_offset=self.pocket_params["ligand_radii_offset"],
            )
            pocket_data = pocket.to_dict()
            pyserials.write.to_yaml_file(
                data=pocket_data,
                path=filepath_pocket,
            )
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pocket"] = pocket
        return pocket

    def field(self, pdb_id: str):
        cached = self._cache.get(pdb_id, {}).get("field")
        if cached:
            return cached
        filepath_field = self.filepath(pdb_id, "field")
        if filepath_field.is_file():
            field_data = pyserials.read.json_from_file(filepath_field)
            field = t2fpharm.field.from_data(**field_data)
        else:
            dirpath_autogrid = self._dirpath["autogrid"] / pdb_id
            if dirpath_autogrid.exists():
                shutil.rmtree(dirpath_autogrid)
            dirpath_autogrid.mkdir(parents=True, exist_ok=True)
            pocket_data = self.pocket(pdb_id).to_dict()
            grid_data = {k: v for k, v in pocket_data.items() if k.startswith("grid_")}
            field = t2fpharm.field.from_autogrid(
                receptor_files=self.filepath(pdb_id, "pdbqt"),
                receptor_file_ids=pdb_id,
                ligand_types=self.field_params["ligand_types"],
                smooth=self.field_params["smooth"],
                dielectric=self.field_params["dielectric"],
                output_dir=dirpath_autogrid,
                **grid_data,
            )
            field_data = field.to_dict()
            pyserials.write.to_json_file(
                data=field_data,
                path=filepath_field,
            )
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["field"] = field
        return field

    def modeler(self, pdb_id: str):
        return t2fpharm.modeler(
            field=self.field(pdb_id),
            pocket=self.pocket(pdb_id),
            receptor=self.complex(pdb_id),
        )

    def ligand_pharmacophore(self, pdb_id: str):
        cached = self._cache.get(pdb_id, {}).get("ligand_pharm")
        if cached:
            return cached
        filepath_ligand_plip = self.filepath(pdb_id, "ligand_plip")
        filepath_ligand_features = self.filepath(pdb_id, "ligand_features")
        if filepath_ligand_plip.is_file() and filepath_ligand_features.is_file():
            plip_data = pyserials.read.json_from_file(filepath_ligand_plip)
            features_data = pyserials.read.json_from_file(filepath_ligand_features)
            plip_df = pd.DataFrame(plip_data)
            ligand_pharm = t2fpharm.ligand.LigandPharmacophore(
                features=features_data,
                extra={"plip": caddpy.interaction.ProteinLigandInteractions(plip_df)},
                receptor=self.complex(pdb_id),
            )
        else:
            ligand_pharm = t2fpharm.ligand.from_plip(
                pdb_files=self.filepath(pdb_id, "pdb_fixed"),
                pocket=self.pocket(pdb_id),
                receptor=self.complex(pdb_id),
            )
            filepath_ligand_plip.write_text(
                ligand_pharm.extra["plip"].all.to_json(orient="records", indent=4)
            )
            filepath_ligand_features.write_text(
                ligand_pharm.features.to_json(orient="records", indent=4)
            )
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["ligand_pharm"] = ligand_pharm
        return ligand_pharm

    def affinity(self, pdb_id: str) -> dict:
        """Get affinity data for a given PDB ID and ligand."""
        filepath_affinity = self.filepath(pdb_id, "affinity")
        if filepath_affinity.is_file():
            affinity_data = pyserials.read.json_from_file(filepath_affinity)
        else:
            data = sciapi.pdb.data.entry(pdb_id)
            affinity_data = data.get("rcsb_binding_affinity", [])
            pyserials.write.to_json_file(
                data=affinity_data,
                path=filepath_affinity,
            )
        results = {}
        ligand_res_name = self.dataset.loc[pdb_id, "ligand_res_name"]
        for affinity in affinity_data:
            if affinity["comp_id"] == ligand_res_name:
                values, weights = results.setdefault(affinity["type"], ([], []))
                values.append(affinity["value"])
                weights.append(affinity["reference_sequence_identity"])
        out = {}
        for affinity_type, (values, weights) in results.items():
            weighted_average = np.average(values, weights=weights)
            out[affinity_type] = weighted_average
        return out

    def filepath(
        self,
        pdb_id: str,
        filetype: Literal[
            "pdb_raw",
            "pdb_fixed",
            "pdb_apo",
            "pdbqt",
            "affinity",
            "pocket",
            "field",
            "ligand_plip",
            "ligand_features"
        ]
    ) -> Path:
        dirpath = self.dirpath_data / self._dirpath[filetype]
        dirpath.mkdir(parents=True, exist_ok=True)
        file_ext = self._file_ext[filetype]
        return Path(dirpath) / f"{pdb_id}.{file_ext}"

    @staticmethod
    def plot_match_ratio_heatmap(
        df: pd.DataFrame,
        row_label: str | None = None,
        col_label: str | None = None,
        val_label: str | None = None,
        fig_width: float = 10.0,
        cmap: str = 'viridis',
        n_colorbar_ticks: int = 11,
        annotate: bool = True
    ) -> None:
        """Plot heatmap of DataFrame values.

        Parameters
        ----------
        df
            DataFrame with numeric values to plot.
        row_label
            Label for the y-axis (rows).
        col_label
            Label for the x-axis (columns).
        val_label
            Label for the colorbar (values).
        fig_width
            Figure width in inches.
        cmap
            Matplotlib colormap name.
        n_colorbar_ticks
            How many ticks to show on the colorbar.
        annotate
            Whether to display each cell's value on the heatmap.
        """
        n_rows, n_cols = df.shape

        # Figure sizing
        fig_height = fig_width * (n_rows / n_cols)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Draw heatmap
        im = ax.imshow(
            df.values,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            interpolation='nearest'
        )

        # Ticks & labels
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(
            [f"{x:.0f}" if isinstance(x, float) else [f"{xx:.2f}" for xx in x] for x in df.columns],
            rotation=45, ha='right'
        )
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([f"{y:.2f}" for y in df.index])
        if row_label:
            ax.set_ylabel(row_label)
            ax.yaxis.labelpad = 20
        if col_label:
            ax.set_xlabel(col_label)

        # Annotate each cell
        if annotate:
            # choose a threshold to decide text color
            vmin, vmax = im.get_clim()
            mid = (vmin + vmax) / 2
            for i in range(n_rows):
                for j in range(n_cols):
                    val = df.values[i, j]
                    color = 'white' if val < mid else 'black'
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha='center', va='center', color=color, fontsize=8
                    )

        # Make colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        vmin, vmax = im.get_clim()
        ticks = np.linspace(vmin, vmax, n_colorbar_ticks)
        cbar = fig.colorbar(im, cax=cax, ticks=ticks)
        if val_label:
            cbar.set_label(val_label, labelpad=15)
            cbar.ax.yaxis.labelpad = 15
        plt.show()
        return


def load(
    dirpath_data: Path | str | None = None,
    *,
    filepath_inputs: Path | str  = "inputs.yaml",
    dirpath_pdb_raw: Path | str = "structure/1-pdb-raw",
    dirpath_pdb_fixed: Path | str = "structure/2-pdb-fixed",
    dirpath_pdb_aligned: Path | str = "structure/3-pdb-aligned",
    dirpath_pdb_apo: Path | str = "structure/4-pdb-apo",
    dirpath_pdbqt: Path | str = "structure/5-pdbqt",
    dirpath_affinity: Path | str = "affinity",
    dirpath_pocket: Path | str = "pocket",
    dirpath_autogrid: Path | str = "autogrid",
    dirpath_field: Path | str = "field",
    dirpath_ligand_plip: Path | str = "ligand/plip",
    dirpath_ligand_features: Path | str = "ligand/features",
) -> Manager:
    """Load the manager.

    Parameters
    ----------
    dirpath_data
        Path to the data directory.
        If not provided, the default data directory is used.
    filepath_inputs
        Path to the inputs file (JSON, YAML, or TOML)
        relative to `dirpath`.
    """
    dirpath_data = (
        Path(dirpath_data) if dirpath_data else
        pkgdata.get_package_path_from_caller(top_level=True) / "data"
    )
    input_data = pyserials.read.from_file(
        path=dirpath_data / filepath_inputs,
        json_strict=True,
        yaml_safe=True,
        toml_as_dict=True,
    )
    inputs = input_data["data"]
    rows = []
    group_color = {}
    for group_data in inputs["receptor_groups"]:
        group_color[group_data["id"]] = {
            "bg": group_data.get("color_bg"),
            "text": group_data.get("color_text"),
        }
        group = {
            "group_id": group_data["id"],
            "group_name": group_data["name"],
            "uniprot_id": group_data.get("uniprot_id"),
        }
        row = _make_structure(
            group=group,
            structure=group_data["ref_structure"],
            is_ref=True
        )
        rows.append(row)
        for structure_data in group_data.get("structures", []):
            row = _make_structure(
                group=group,
                structure=structure_data,
                is_ref=False
            )
            rows.append(row)
    df = pd.DataFrame(rows).convert_dtypes()
    df.set_index("pdb_id", inplace=True, drop=False)
    return Manager(
        dataset=df,
        pocket_inputs=inputs["pocket"],
        field_inputs=inputs["field"],
        group_color=group_color,
        dirpath_data=dirpath_data,
        dirpath_pdb_raw=dirpath_pdb_raw,
        dirpath_pdb_fixed=dirpath_pdb_fixed,
        dirpath_pdb_apo=dirpath_pdb_apo,
        dirpath_pdbqt=dirpath_pdbqt,
        dirpath_affinity=dirpath_affinity,
        dirpath_pocket=dirpath_pocket,
        dirpath_autogrid=dirpath_autogrid,
        dirpath_field=dirpath_field,
        dirpath_ligand_plip=dirpath_ligand_plip,
        dirpath_ligand_features=dirpath_ligand_features,
    )


def _make_structure(
    group: dict,
    structure: dict,
    is_ref: bool = False
):
    structure_full = group | {
        "pdb_id": structure["pdb_id"].upper(),
        "is_ref": is_ref,
        "chain_id": structure.get("chain_id"),
        "ligand_res_name": structure.get("ref_ligand", {}).get("res_name"),
        "ligand_chain_id": structure.get("ref_ligand", {}).get("chain_id"),
        "ligand_res_seq": structure.get("ref_ligand", {}).get("res_seq"),
    }
    return structure_full


def _align_to_ref_structure(
    ref_complex: t2fpharm.receptor.Receptor,
    query_complex: t2fpharm.receptor.Receptor,
    ref_chain_id: str,
    query_chain_id: str,
    ref_pocket: t2fpharm.pocket.Pocket,
) -> t2fpharm.receptor.Receptor:
    """Align the query receptor to the reference receptor."""
    ref_chain = ref_complex.composition.atoms_chain(ref_chain_id, poly=True)
    query_chain = query_complex.composition.atoms_chain(query_chain_id, poly=True)
    ref_aligned_atoms, query_aligned_atoms = caddpy.alignment.align_sequences(ref_chain, query_chain)
    ref_pocket_c_alpha_atoms = ref_pocket.atoms[ref_pocket.atoms["name"]=="CA"]
    c_alpha_mask = ref_aligned_atoms["serial"].isin(ref_pocket_c_alpha_atoms["serial"])
    ref_pocket_c_alpha_serials = ref_aligned_atoms["serial"][c_alpha_mask]
    query_pocket_c_alpha_serials = query_aligned_atoms["serial"][c_alpha_mask]
    ref_selection_mask = ref_complex.composition.atoms["serial"].isin(ref_pocket_c_alpha_serials)
    query_selection_mask = query_complex.composition.atoms["serial"].isin(query_pocket_c_alpha_serials)
    ref_selection_coordinates = ref_complex.trajectory.points[ref_selection_mask.to_numpy()]
    query_selection_coordinates = query_complex.trajectory.points[query_selection_mask.to_numpy()]
    rotation, translation, rmsd = arrayer.kabsch.kabsch_unweighted(ref_selection_coordinates, query_selection_coordinates)
    query_complex_aligned = query_complex.new(trajectory=query_complex.trajectory.points @ rotation + translation)
    return query_complex_aligned


def _run_job(
    modeler: t2fpharm.Modeler,
    ligand_pharm: t2fpharm.ligand.LigandPharmacophore,
    pdb_id: str,
    job_idx: int,
    method: str,
    kwargs: dict[str, Any],
    match_max_dist: float | None,
) -> dict[str, int | float | t2fpharm.pharmacophore.Pharmacophore | pd.DataFrame]:
    try:
        func = getattr(modeler, method)
        pharm = func(**kwargs)
        matches = pharm.match_spherical(ligand_pharm, max_distance=match_max_dist)
    except Exception as e:
        raise RuntimeError(
            f"Error running job {job_idx} for PDB ID {pdb_id} with method {method}: {e}"
        ) from e
    n_matches = matches["match"].sum()
    n_lig_feats = len(ligand_pharm.features)
    stats = {
        "pdb_id": pdb_id,
        "method": method,
        "job_idx": job_idx,
        "feats_receptor": len(pharm.features),
        "feats_ligand": n_lig_feats,
        "matches": n_matches,
        "match_percent": 100 * n_matches / n_lig_feats,
    }
    return pharm, matches, stats
