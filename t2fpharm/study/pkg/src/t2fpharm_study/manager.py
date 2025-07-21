import os
from pathlib import Path
import shutil
from typing import Any, Sequence, Literal, TypeAlias
import copy
import json

import arrayer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm
import ray

import pyserials

import sciapi
import scifile
import caddpy

import t2fpharm

from t2fpharm_study.job_gen import create_job_inputs
from t2fpharm_study.job_runner import run
from t2fpharm_study.typing import PDBID


_remote_job_runner = ray.remote(run)


class Manager:
    def __init__(
        self,
        dataset: pd.DataFrame,
        pocket_inputs: dict,
        field_inputs: dict,
        job_inputs: dict[str, dict[str, Any]],
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
        dirpath_results: Path | str = "results",
    ):
        self._data = dataset
        self.pocket_params = pocket_inputs
        self.field_params = field_inputs
        self.job_params = job_inputs
        self._group_color = group_color
        self.dirpath_data = Path(dirpath_data)
        self._path = {
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

            "results": dirpath_results,

            "results_job_inputs": f"{dirpath_results}/jobs",
            "results_summary": f"{dirpath_results}/summary",
            "results_pharm": f"{dirpath_results}/pharmacophore",
            "results_matches": f"{dirpath_results}/matches",
        }
        self._file_ext = {
            "pdb_raw": "pdb",
            "pdb_fixed": "pdb",
            "pdb_aligned": "pdb",
            "pdb_apo": "pdb",
            "pdbqt": "pdbqt",
            "affinity": "json",
            "pocket": "npz",
            "field": "npz",
            "ligand_plip": "json",
            "ligand_features": "json",
            "results_job_inputs": "json",
            "results_summary": "json",
            "results_pharm": "json",
            "results_matches": "json",
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

    def load(
        self,
        pdb_ids: Sequence[PDBID] | None = None,
        pdb_raw: bool = False,
        pdbqt: bool = False,
    ):
        self.caching(enabled=True)
        for _, entry in tqdm(
            self.dataset.iterrows(),
            total=len(self.dataset),
            desc="Loading data",
            unit="job",
        ):
            pdb_id = entry["pdb_id"]
            if pdb_ids is not None and pdb_id not in pdb_ids:
                continue
            self.complex(pdb_id)
            self.receptor(pdb_id)
            self.pocket(pdb_id)
            self.field(pdb_id)
            self.modeler(pdb_id)
            self.ligand_pharmacophore(pdb_id)
            self.affinity(pdb_id)
            if pdb_raw:
                self.pdb_raw(pdb_id)
            if pdbqt:
                self.pdbqt(pdb_id)
        return

    def caching(self, enabled: bool = True):
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._cache = {}
        return

    def run(self, job_name: str, ref_only: bool = False):
        # Enable caching to avoid recomputing heavy objects during modeler creation
        caching_was_enabled = self._cache_enabled
        self.caching(enabled=True)

        # Load summary to skip already completed jobs
        try:
            summary_df = self.job_summary(job_name=job_name)
        except FileNotFoundError:
            summary_df = pd.DataFrame(columns=["pdb_id", "job_idx"])

        # Get job inputs
        jobs = self.job_inputs(job_name=job_name)

        # Prepare jobs
        ligand_pharms: dict[PDBID, t2fpharm.pharm_ligand.LigandPharmacophore] = {}
        futures = []
        ray.init(ignore_reinit_error=True)
        with tqdm(
            total=(self.dataset["is_ref"].sum() if ref_only else len(self.dataset)) * len(jobs),
            desc="Creating jobs",
            unit="job"
        ) as pbar:
            for _, group in self.dataset.groupby("group_id"):
                ligand_pharms = ray.put(
                    {
                        pdb_id: self.ligand_pharmacophore(pdb_id)
                        for pdb_id in group["pdb_id"]
                    }
                )
                for pdb_id in group["pdb_id"]:
                    if ref_only and not self.is_ref(pdb_id):
                        continue
                    modeler = ray.put(self.modeler(pdb_id))
                    if not caching_was_enabled:
                        self._cache = {}
                    for job_idx, job in enumerate(jobs):
                        if (summary_df["pdb_id"].eq(pdb_id) & summary_df["job_idx"].eq(job_idx)).any():
                            pbar.update(1)
                            continue
                        remote_job = _remote_job_runner.remote(
                            modeler=modeler,
                            ligand_pharms=ligand_pharms,
                            target_pdb_id=pdb_id,
                            job_idx=job_idx,
                            method=job["method"],
                            kwargs=job["kwargs"],
                            feature_types=self.field_params["ligand_types"],
                            filepath_features=str(self.path_results(job_name=job_name, filetype="pharm", pdb_id=pdb_id, job_idx=job_idx)),
                            filepath_matches=str(self.path_results(job_name=job_name, filetype="matches", pdb_id=pdb_id, job_idx=job_idx)),
                            return_pharm=False,
                            return_matches=False,
                        )
                        futures.append(remote_job)
                        pbar.update(1)

        # Restore caching state
        self.caching(enabled=caching_was_enabled)

        # Create summary file if it does not exist
        summary_path = self.path_results(job_name=job_name, filetype="summary")
        if not summary_path.is_file():
            summary_path.write_text("")

        # Gather job results
        write_batch_size = 100
        count = 0
        remaining_futures = futures[:]
        with summary_path.open("a") as f, tqdm(total=len(futures), desc="Running jobs", unit="job") as pbar:
            while remaining_futures:
                done_futures, remaining_futures = ray.wait(remaining_futures, num_returns=1)
                summary = ray.get(done_futures[0])[0]
                pdb_id = summary["pdb_id"]
                job_index = summary["job-idx"]
                group_id = self.group_id(pdb_id)
                summary.update(
                    {
                        "group_id": group_id,
                        "job-method": jobs[job_index]["method"],
                        **{f"job-{k}": v for k, v in jobs[job_index]["identifier"].items()},
                    }
                )
                f.write(json.dumps(summary, separators=(",", ":")) + "\n")
                count += 1
                if count == write_batch_size:
                    f.flush()
                    os.fsync(f.fileno())
                    count = 0
                pbar.update(1)
            f.flush()
            os.fsync(f.fileno())
        return self.job_summary(job_name=job_name)

    def job_summary(self, job_name: str) -> pd.DataFrame:
        path = self.path_results(job_name=job_name, filetype="summary")
        if not path.is_file():
            raise FileNotFoundError(f"Summary file not found: {path}")
        df = pd.read_json(path, lines=True).convert_dtypes()
        main_cols = ["job-method", "job-idx", "group_id", "pdb_id"]
        extra_cols = sorted([col for col in df.columns if col not in main_cols])
        all_cols = main_cols + extra_cols
        df_final = df[all_cols].sort_values(["job-method", "job-idx", "group_id", "pdb_id"]).reset_index(drop=True)
        return df_final

    def job_inputs(self, job_name: str) -> list[dict[str, Any]]:
        """Get the inputs for a given job."""
        path = self.path_results(job_name=job_name, filetype="inputs")
        if path.is_file():
            jobs = pyserials.read.yaml_from_file(path)
            for job in jobs:
                if "min_distance" in job["kwargs"]:
                    job["kwargs"]["min_distance"] = {
                        tuple(k.split("__")): v
                        for k, v in job["kwargs"]["min_distance"].items()
                    }
            return jobs

        job_spec = self.job_params.get(job_name)
        if job_spec is None:
            raise ValueError(f"Job '{job_name}' not found in job parameters.")
        if "cnn" in job_spec:
            job_spec["cnn"]["grid_spacing"] = self.pocket_params["grid_spacing"]
            job_spec["cnn"]["grid_unique_distances"] = self.field(pdb_id=self.dataset["pdb_id"].iloc[0]).grid.unique_distances
        jobs = create_job_inputs(**job_spec)
        jobs_serialized = copy.deepcopy(jobs)
        for job in jobs_serialized:
            if "min_distance" in job["kwargs"]:
                job["kwargs"]["min_distance"] = {f"{k[0]}__{k[1]}":v for k, v in job["kwargs"]["min_distance"].items()}
        pyserials.write.to_yaml_file(
            data=jobs_serialized,
            path=path,
        )
        return jobs

    def job_pharmacophore(self, job_name: str, pdb_id: str, job_idx: int) -> t2fpharm.Pharmacophore:
        """Get the pharmacophore for a given job and PDB ID."""
        filepath = self.path_results(job_name=job_name, filetype="pharm", pdb_id=pdb_id, job_idx=job_idx)
        if not filepath.is_file():
            raise FileNotFoundError(f"Pharmacophore file not found: {filepath}")
        features = pd.read_parquet(filepath, engine="pyarrow")
        if features['label'].dtype == object:
            features['label'] = features['label'].apply(tuple)
        return t2fpharm.Pharmacophore(
            features=features,
            feature_types=self.field_params["ligand_types"],
            system=self.complex(pdb_id),
            pocket=self.pocket(pdb_id),
            field=self.field(pdb_id),
        )

    def job_matches(self, job_name: str, pdb_id: str, job_idx: int) -> pd.DataFrame:
        """Get the matches for a given job and PDB ID."""
        filepath = self.path_results(job_name=job_name, filetype="matches", pdb_id=pdb_id, job_idx=job_idx)
        if not filepath.is_file():
            raise FileNotFoundError(f"Matches file not found: {filepath}")
        matches = pd.read_parquet(filepath, engine="pyarrow")
        if matches['target_label'].dtype == object:
            matches['target_label'] = matches['target_label'].apply(tuple)
        return matches

    def pdb_raw(self, pdb_id: str) -> scifile.pdb.PDBFile:
        cached = self._cache.get(pdb_id, {}).get("pdb_raw")
        if cached:
            return cached
        filepath_pdb_raw = self.path("pdb_raw", pdb_id)
        if filepath_pdb_raw.is_file():
            pdb = scifile.pdb.read(filepath_pdb_raw)
        else:
            pdb_raw_bytes = sciapi.pdb.file.entry(pdb_id=pdb_id, file_format="pdb")
            filepath_pdb_raw.write_bytes(pdb_raw_bytes)
            pdb = scifile.pdb.read(filepath_pdb_raw)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pdb_raw"] = pdb
        return pdb

    def complex(self, pdb_id: str) -> t2fpharm.System:
        cached = self._cache.get(pdb_id, {}).get("complex")
        if cached:
            return cached
        filepath_pdb_aligned = self.path("pdb_aligned", pdb_id)
        if filepath_pdb_aligned.is_file():
            pdb_aligned_str = filepath_pdb_aligned.read_text()
        else:
            filepath_pdb_raw = self.path("pdb_raw", pdb_id)
            if not filepath_pdb_raw.is_file():
                self.pdb_raw(pdb_id)
            (
                pdb_fixed_str,
                missing_residues,
                nonstandard_residues,
                missing_atoms,
                missing_terminals
            ) = caddpy.chemsys.fix_pdb(
                file=filepath_pdb_raw,
                keep_chain_ids=self.dataset.loc[pdb_id, "chain_id"],
                add_missing_residues=True,
                replace_nonstandard_residues=False,
                add_missing_heavy_atoms=True,
                add_missing_atoms_seed=42,
                add_missing_hydrogens=7.0,
                keep_ids=True,
            )
            self.path("pdb_fixed", pdb_id).write_text(pdb_fixed_str)
            if self.dataset.loc[pdb_id, "is_ref"]:
                pdb_aligned_str = pdb_fixed_str
            else:
                group_id = self.dataset.loc[pdb_id, "group_id"]
                is_group_ref = (self.dataset["group_id"] == group_id) & self.dataset["is_ref"]
                ref_pdb_id = self.dataset.loc[is_group_ref, "pdb_id"].iloc[0]
                complex_aligned = _align_query_to_ref(
                    ref=self.complex(ref_pdb_id),
                    query=t2fpharm.system.from_pdb(pdb_fixed_str),
                    ref_chain_id=self.dataset.loc[ref_pdb_id, "chain_id"],
                    query_chain_id=self.dataset.loc[pdb_id, "chain_id"],
                    ref_pocket=self.pocket(ref_pdb_id),
                )
                pdb_aligned_str = str(complex_aligned.to_pdb())
            filepath_pdb_aligned.write_text(pdb_aligned_str)
        rcomplex = t2fpharm.system.from_pdb(pdb_aligned_str)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["complex"] = rcomplex
        return rcomplex

    def receptor(self, pdb_id: str) -> t2fpharm.System:
        cached = self._cache.get(pdb_id, {}).get("receptor")
        if cached:
            return cached
        filepath_pdb_apo = self.path("pdb_apo", pdb_id)
        if filepath_pdb_apo.is_file():
            receptor = t2fpharm.system.from_pdb(filepath_pdb_apo)
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
        filepath_pdbqt = self.path("pdbqt", pdb_id)
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

    def pocket(self, pdb_id: str) -> t2fpharm.Pocket:
        """Prepare the pocket for the structure."""
        cached = self._cache.get(pdb_id, {}).get("pocket")
        if cached:
            return cached
        filepath_pocket = self.path("pocket", pdb_id)
        rcomplex = self.complex(pdb_id)
        if filepath_pocket.is_file():
            pocket = t2fpharm.pocket.from_npz(filepath=filepath_pocket, receptor=rcomplex)
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
                ligand_radii_offset=self.pocket_params["ligand_radii_offset"],
                erosion_radius=self.pocket_params.get("erosion_radius", 0),
                grid=self.pocket_params["grid_spacing"],
            )
            pocket.to_npz(filepath=filepath_pocket)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pocket"] = pocket
        return pocket

    def field(self, pdb_id: str) -> t2fpharm.Field:
        cached = self._cache.get(pdb_id, {}).get("field")
        if cached:
            return cached
        filepath_field = self.path("field", pdb_id)
        if filepath_field.is_file():
            field = t2fpharm.field.from_npz(filepath=filepath_field)
        else:
            dirpath_autogrid = self.dirpath_data / self._path["autogrid"] / pdb_id
            if dirpath_autogrid.exists():
                shutil.rmtree(dirpath_autogrid)
            dirpath_autogrid.mkdir(parents=True, exist_ok=True)
            pocket_data = self.pocket(pdb_id).to_dict()
            grid_data = {k: v for k, v in pocket_data.items() if k.startswith("grid_")}
            field = t2fpharm.field.from_autogrid(
                receptor_files=self.path("pdbqt", pdb_id),
                receptor_file_ids=pdb_id,
                ligand_types=self.field_params["ligand_types"],
                smooth=self.field_params["smooth"],
                dielectric=self.field_params["dielectric"],
                output_dir=dirpath_autogrid,
                **grid_data,
            )
            field.to_npz(filepath=filepath_field)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["field"] = field
        return field

    def modeler(self, pdb_id: str) -> t2fpharm.Modeler:
        return t2fpharm.modeler(
            field=self.field(pdb_id),
            pocket=self.pocket(pdb_id),
            receptor=self.complex(pdb_id),
        )

    def ligand_pharmacophore(self, pdb_id: str):
        cached = self._cache.get(pdb_id, {}).get("ligand_pharm")
        if cached:
            return cached
        filepath_ligand_plip = self.path("ligand_plip", pdb_id)
        filepath_ligand_features = self.path("ligand_features", pdb_id)
        if filepath_ligand_plip.is_file() and filepath_ligand_features.is_file():
            plip_data = pyserials.read.from_file(filepath_ligand_plip, toml_as_dict=True)
            features_data = pyserials.read.from_file(filepath_ligand_features, toml_as_dict=True)
            plip_df = pd.DataFrame(plip_data)
            ligand_pharm = t2fpharm.pharm.Pharmacophore(
                features=features_data,
                extra={"plip": caddpy.interaction.ProteinLigandInteractions(plip_df)},
                system=self.complex(pdb_id),
                pocket=self.pocket(pdb_id),
            )
        else:
            ligand_pharm = t2fpharm.pharm.from_complex(
                pdb_files=self.path("pdb_aligned", pdb_id),
                pocket=self.pocket(pdb_id),
                receptor=self.complex(pdb_id),
                type_hbond_acceptor="OA",
                type_hbond_donor="HD",
                type_water_bridge_ligand_acceptor=None,
                type_water_bridge_ligand_donor=None,
                type_water_bridge_water_acceptor="OA",
                type_anion="e-",
                type_cation="e+",
                type_hydrophobic="C",
                type_aromatic=None,
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
        filepath_affinity = self.path("affinity", pdb_id)
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

    def path(
        self,
        filetype: Literal[
            "pdb_raw",
            "pdb_fixed",
            "pdb_aligned",
            "pdb_apo",
            "pdbqt",
            "affinity",
            "pocket",
            "field",
            "ligand_plip",
            "ligand_features",
            "results_job_inputs",
            "results_summary",
            "results_pharm",
            "results_matches",
            "results_matches_ref",
        ],
        pdb_id: str | None = None,
        job_idx: int | None = None,
    ) -> Path:
        path = self.dirpath_data / self._path[filetype]
        dirpath = path.parent if filetype in ("results_job_inputs", "results_summary") else path
        dirpath.mkdir(parents=True, exist_ok=True)
        file_ext = self._file_ext[filetype]
        if filetype in ("results_job_inputs", "results_summary"):
            return path.with_suffix(f".{file_ext}")
        if filetype in ("results_pharm", "results_matches", "results_matches_ref"):
            if pdb_id is None or job_idx is None:
                raise ValueError(
                    "For results files, both `pdb_id` and `job_idx` must be provided."
                )
            return path / f"{pdb_id}_{job_idx}.{file_ext}"
        return path / f"{pdb_id}.{file_ext}"

    def path_results(
        self,
        job_name: str,
        filetype: Literal[
            "inputs",
            "summary",
            "pharm",
            "matches",
        ],
        pdb_id: str | None = None,
        job_idx: int | None = None,
    ) -> Path:
        path = self.dirpath_data / self._path["results"] / job_name
        path.mkdir(parents=True, exist_ok=True)
        if filetype == "inputs":
            return path / "inputs.yaml"
        if filetype == "summary":
            return path / "summary.yaml"
        if pdb_id is None or job_idx is None:
            raise ValueError(
                "For results files, both `pdb_id` and `job_idx` must be provided."
            )
        subdir = path / filetype
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{pdb_id}_{job_idx}.parquet"

    def group_id(self, pdb_id: str) -> str:
        """Get the group ID for a given PDB ID."""
        return self.dataset.loc[pdb_id, "group_id"]

    def group_pdb_ids(self, group_id: str, include_ref: bool = True) -> list[str]:
        """Get all PDB IDs for a given group ID."""
        mask = self.dataset["group_id"] == group_id
        if not include_ref:
            mask &= ~self.dataset["is_ref"]
        return self.dataset[mask]["pdb_id"].tolist()

    def is_ref(self, pdb_id: str) -> bool:
        """Check if the structure is a reference structure."""
        return self.dataset.loc[pdb_id, "is_ref"]

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


def _align_query_to_ref(
    ref: t2fpharm.system.System,
    query: t2fpharm.system.System,
    ref_chain_id: str,
    query_chain_id: str,
    ref_pocket: t2fpharm.pocket.Pocket,
) -> t2fpharm.system.System:
    """Align the query system to the reference system.

    This function first runs a sequence alignment
    between the reference and query chains
    in the corresponding chemical systems
    to find all pairwise correlations
    between the atoms in the two chains.
    It then selects the C-alpha atoms
    of the binding pocket in the reference that have
    a corresponding match in the sequence alignment.
    Lastly, it applies the Kabsch algorithm
    to find the optimal rotation and translation
    that aligns the selected C-alpha atoms
    of the query to the reference system.
    The rotation and translation are then applied
    to the entire query system.

    Parameters
    ----------
    ref
        Reference chemical system.
    query
        Query chemical system to align.
    ref_chain_id
        Chain ID of the polymer chain of interest in the reference.
    query_chain_id
        Chain ID of the corresponding polymer chain in the query.
    ref_pocket
        Binding pocket of the reference.

    Returns
    -------
    The same query chemical system with its trajectory aligned to the reference.
    """
    ref_chain = ref.composition.atoms_chain(ref_chain_id, poly=True)
    query_chain = query.composition.atoms_chain(query_chain_id, poly=True)
    ref_aligned_atoms, query_aligned_atoms = caddpy.alignment.align_sequences(ref_chain, query_chain)
    ref_pocket_c_alpha_atoms = ref_pocket.atoms[ref_pocket.atoms["name"]=="CA"]
    c_alpha_mask = ref_aligned_atoms["serial"].isin(ref_pocket_c_alpha_atoms["serial"])
    ref_pocket_c_alpha_serials = ref_aligned_atoms["serial"][c_alpha_mask]
    query_pocket_c_alpha_serials = query_aligned_atoms["serial"][c_alpha_mask]
    ref_selection_mask = ref.composition.atoms["serial"].isin(ref_pocket_c_alpha_serials)
    query_selection_mask = query.composition.atoms["serial"].isin(query_pocket_c_alpha_serials)
    ref_selection_coordinates = ref.trajectory.points[ref_selection_mask.to_numpy()]
    query_selection_coordinates = query.trajectory.points[query_selection_mask.to_numpy()]
    rotation, translation, rmsd = arrayer.kabsch.kabsch_unweighted(ref_selection_coordinates, query_selection_coordinates)
    query_complex_aligned = query.new(trajectory=query.trajectory.points @ rotation + translation)
    return query_complex_aligned
