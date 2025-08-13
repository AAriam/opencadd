import os
from pathlib import Path
import shutil
from typing import Any, Sequence, Literal
import json
import warnings

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

from t2fpharm_study import io
from t2fpharm_study.job_gen import generate_job_inputs
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
        dirpath_pdb_raw: Path | str,
        dirpath_pdb_fixed: Path | str,
        dirpath_pdb_aligned: Path | str,
        dirpath_pdb_apo: Path | str,
        dirpath_pdbqt: Path | str,
        dirpath_affinity: Path | str,
        dirpath_pocket: Path | str,
        dirpath_autogrid: Path | str,
        dirpath_field: Path | str,
        dirpath_ligand_plip: Path | str,
        dirpath_ligand_features: Path | str,
        dirpath_ref_features: Path | str,
        dirpath_jobs: Path | str,
        dirname_job_pharms: str,
        dirname_job_matches: str,
    ):
        self._data = dataset
        self.pocket_params = pocket_inputs
        self.field_params = field_inputs
        self.job_params = job_inputs
        self._group_color = group_color
        self.dirpath_data = Path(dirpath_data)
        self._dirname = {
            "job_pharms": dirname_job_pharms,
            "job_matches": dirname_job_matches,
        }
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
            "ref_features": dirpath_ref_features,

            "results": dirpath_jobs,

            "results_job_inputs": f"{dirpath_jobs}/jobs",
            "results_summary": f"{dirpath_jobs}/summary",
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
            "ref_features": "json",
            "results_job_inputs": "json",
            "results_summary": "json",
            "results_pharm": "json",
            "results_matches": "json",
        }
        self._pdb = None
        self._cache_enabled = True
        self._cache = {}
        self._group_grid: dict[str, t2fpharm.Grid] = {}
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
            self.ref_pharmacophore(pdb_id)
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

    def run_all(self, ref_only: bool = True):
        """Run all jobs defined in the job parameters."""
        for job_name in self.job_params:
            self.run(job_name=job_name, ref_only=ref_only)
        return

    def run(self, job_name: str, ref_only: bool = True):
        def keep_row(pdb_id: str, job_idx: int) -> bool:
            """Check if the entire job batch is completed."""
            # generate all the slots in this batch
            slots = ((pdb_id, job_idx + offset)
                    for offset in range(batch_size_per_job))
            # if *all* of them are in completed, drop; otherwise keep
            return not all(slot in completed_jobs for slot in slots)

        # Get job inputs
        jobs = self._job_inputs(job_name=job_name, grouped=True)
        batch_size_per_job = len(jobs.iloc[0].get("min_members_dicts", [None])) * len(jobs.iloc[0].get("center_types", [None]))

        # Get job dataset
        dataset = self.dataset
        if ref_only:
            dataset = dataset[dataset["is_ref"]]
        dataset = dataset[["group_id", "pdb_id"]].reset_index(drop=True)

        # Create a dataframe with one row per job/PDB ID pair
        all_jobs = dataset.merge(jobs, how="cross")

        # Load summary dataframe to skip already completed jobs
        try:
            summary_df = self.job_summary(job_name=job_name)
            completed_jobs = set(zip(summary_df["pdb_id"], summary_df["job_idx"]))
        except FileNotFoundError:
            completed_jobs = None
        if completed_jobs is not None:
            all_jobs["keep"] = [
                keep_row(pdb_id, job_idx)
                for pdb_id, job_idx in zip(all_jobs["pdb_id"], all_jobs["job_idx"])
            ]
            print(f"Skipping {all_jobs[~all_jobs['keep']].shape[0] * batch_size_per_job} already completed jobs.")
            all_jobs = all_jobs[all_jobs["keep"]].drop(columns=["keep"])

        # Common parameters for all jobs
        method = self.job_params[job_name]["method"]
        dirpath_features = str(self.job_dirpath(job_name, dirtype="pharms"))
        dirpath_matches = str(self.job_dirpath(job_name, dirtype="matches"))

        # Enable caching to avoid recomputing heavy objects during modeler creation
        caching_was_enabled = self._cache_enabled
        self.caching(enabled=True)

        # Prepare jobs
        futures = []
        ray.init(ignore_reinit_error=True)
        with tqdm(
            total=len(all_jobs) * batch_size_per_job,
            desc="Creating jobs",
            unit="job"
        ) as pbar:
            for pdb_id, pdb_group in all_jobs.groupby("pdb_id"):
                modeler = ray.put(self.modeler(pdb_id))
                ligand_pharm = ray.put(self.ref_pharmacophore(pdb_id, include_extras=False))
                if not caching_was_enabled:
                    self._cache = {}
                pdb_group_jobs = pdb_group.to_dict(orient="records")
                for job in pdb_group_jobs:
                    remote_job = _remote_job_runner.remote(
                        modeler=modeler,
                        ligand_pharm=ligand_pharm,
                        method=method,
                        job=job,
                        dirpath_features=dirpath_features,
                        dirpath_matches=dirpath_matches,
                        return_pharm=False,
                        return_matches=False,
                    )
                    futures.append(remote_job)
                    pbar.update(batch_size_per_job)

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
        with summary_path.open("a") as f, tqdm(
            total=len(futures) * batch_size_per_job,
            desc="Running jobs",
            unit="job"
        ) as pbar:
            while remaining_futures:
                done_futures, remaining_futures = ray.wait(remaining_futures, num_returns=1)
                summaries = ray.get(done_futures[0])[0]
                for summary in summaries:
                    f.write(json.dumps(summary, separators=(",", ":")) + "\n")
                    count += 1
                if count >= write_batch_size:
                    f.flush()
                    os.fsync(f.fileno())
                    count = 0
                pbar.update(1 * batch_size_per_job)
            f.flush()
            os.fsync(f.fileno())
        return self.job_summary(job_name=job_name)

    def job_summary(
        self,
        job_name: str | Sequence[str] | None = None,
        group_cols: Sequence[str] | None = ("job_name", "job_idx")
    ) -> pd.DataFrame:
        """Get the summary DataFrame for a given job or jobs.

        Parameters
        ----------
        job_name
            Name of the job or a list of job names to summarize.
            If None, summary of all available jobs is returned.
        group_cols
            Columns to group by when calculating the mean of the summary DataFrame.
            If None, no grouping is applied and all rows are returned.
            The default value of `("job_name", "job_idx")` calculates the mean values
            for each unique job averaged over all PDB structures.
        """
        if job_name is None:
            job_names = self.job_params.keys()
        elif isinstance(job_name, str):
            job_names = [job_name]
        else:
            job_names = job_name

        summaries = []
        for job_name in job_names:
            try:
                summary = self._job_summary(job_name=job_name)
            except FileNotFoundError:
                warnings.warn(f"No summary found for job '{job_name}'. Skipping.")
                continue
            summary["job_name"] = job_name
            summaries.append(summary)
        df = pd.concat(summaries, ignore_index=True)
        main_cols = ["job_name", "job_idx", "group_id", "pdb_id"]
        all_type_cols = sorted([col for col in df.columns if col.startswith("t_all-")])
        per_type_cols = sorted([col for col in df.columns if col not in main_cols and col not in all_type_cols])
        all_cols = main_cols + all_type_cols + per_type_cols
        df_final = df[all_cols].sort_values(main_cols).reset_index(drop=True)
        if group_cols:
            df_final = self._group_mean(df_final, columns=group_cols)
        return df_final

    def _job_summary(self, job_name: str) -> pd.DataFrame:
        dfs = []
        path = self.path_results(job_name=job_name, filetype="summary")
        if path.is_file():
            df = pd.read_json(path, lines=True).convert_dtypes()
            lig_count_cols = df.columns[df.columns.str.endswith(('-nl_all','-nl_self'))]
            df[lig_count_cols] = df[lig_count_cols].fillna(0)
            dfs.append(df)
            path.unlink()
            write_parquet = True
        else:
            write_parquet = False
        path_final = self.path_results(job_name=job_name, filetype="summary_final")
        if path_final.is_file():
            df = io.read_df(path_final)
            dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"No summary found for job '{job_name}'.")
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if write_parquet:
            io.write_df(df=df, filepath=path_final)
        self._add_f1_score(summary=df)
        return df

    def _add_f1_score(self, summary: pd.DataFrame, char_dist: float = 2) -> None:
        """Calculate precision, sensitivity and F1 scores to each job in the summary DataFrame.

        The calculated values are added to the summary DataFrame as new columns.

        Parameters
        ----------
        summary
            Summary DataFrame containing job results.
        char_dist
            Characteristic distance for generating weights for the F1 score.
        """
        for feature_type in ["all"] + self.field_params["ligand_types"]:
            col_type = f"t_{feature_type}"
            # In order for NaN values calculated by NumPy to be recognized by pandas,
            # we have to cast the input columns to NumPy first.
            # see: https://github.com/pandas-dev/pandas/issues/61758
            n_predicted = summary[f"{col_type}-n_pred"].astype(int)
            n_ref = summary[f"{col_type}-n_ref"].astype(int)
            for match_type in ("greedy", "linear"):
                true_pos = summary[f"{col_type}-dn_lt2-{match_type}"].astype(float)
                mean_dist = summary[f"{col_type}-d_mean-{match_type}"].astype(float)

                precision = (true_pos / n_predicted) if n_predicted > 0 else 0
                sensitivity = true_pos / n_ref
                f1_score = (2 * precision * sensitivity / (precision + sensitivity)).fillna(0)
                weights = np.maximum(0, 1 - mean_dist / char_dist)
                f1_score_weighted = f1_score * weights

                summary[f"{col_type}-precision-m_{match_type}"] = precision
                summary[f"{col_type}-sensitivity-m_{match_type}"] = sensitivity
                summary[f"{col_type}-f1-m_{match_type}"] = f1_score
                summary[f"{col_type}-f1_weighted-m_{match_type}"] = f1_score_weighted
        return

    @staticmethod
    def _group_mean(summary: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Calculate mean of all numeric columns in the summary DataFrame, grouped by the specified columns.

        Parameters
        ----------
        columns
            List of column names to group by.

        Returns
        -------
        A new DataFrame with the group columns and the mean of each numeric column.
        """
        # Ensure grouping columns exist
        missing = set(columns) - set(summary.columns)
        if missing:
            raise ValueError(f"Grouping columns not in DataFrame: {missing}")
        # Identify numeric columns excluding the grouping columns
        numeric_cols = [col for col in summary.columns if col not in ["job_name", "job_idx", "group_id", "pdb_id"]]
        # Perform the groupby and mean aggregation
        return summary.groupby(list(columns))[numeric_cols].mean().reset_index()

    def job_inputs(self, job_name: str | None = None) -> pd.DataFrame:
        """Get the inputs for a given job."""
        if job_name:
            return self._job_inputs(job_name=job_name, grouped=False)
        input_dfs = []
        for job_name in self.job_params:
            inputs = self._job_inputs(job_name=job_name, grouped=False)
            inputs["job_name"] = job_name
            input_dfs.append(inputs)
        df = pd.concat(input_dfs, ignore_index=True)
        main_cols = ["job_name", "job_idx"]
        extra_cols = sorted([col for col in df.columns if col not in main_cols])
        all_cols = main_cols + extra_cols
        df_final = df[all_cols].sort_values(main_cols).reset_index(drop=True)
        return df_final

    def _job_inputs(self, job_name: str, grouped: bool) -> pd.DataFrame:
        """Get the inputs for a given job."""
        job_spec = self.job_params.get(job_name)
        if job_spec is None:
            raise ValueError(f"Job '{job_name}' not found in job parameters.")
        method = job_spec["method"]
        filetype = "jobs" if method == "largest_peaks" or not grouped else "inputs"
        path = self.path_results(job_name=job_name, filetype=filetype)
        if path.is_file():
            jobs = io.read_df(path)
            if method == "largest_peaks":
                jobs["min_distance"] = jobs["min_distance"].apply(
                    lambda x: {tuple(k.split("__")): v for k, v in x.items()}
                )
            return jobs
        if method == "cnn":
            # Add a `Grid` object to the job spec.
            # The grid is only required to get the spacing,
            # unique distances, and number of common neighbors,
            # all of which are only dependent on the grid spacing.
            # Since grid spacing is constant for all entries,
            # it doesn't matter which PDB ID we use here.
            job_spec["grid"] = self.grid(pdb_id=self.dataset["pdb_id"].iloc[0])
        single_jobs, grouped_jobs = generate_job_inputs(**job_spec)
        single_jobs = pd.DataFrame(single_jobs)
        grouped_jobs = pd.DataFrame(grouped_jobs)
        if method == "largest_peaks":
            jobs_serialized = single_jobs.copy()
            jobs_serialized["min_distance"] = jobs_serialized["min_distance"].apply(
                    lambda x: {f"{k[0]}__{k[1]}": v for k, v in x.items()}
                )
            io.write_df(df=jobs_serialized, filepath=path)
        else:
            io.write_df(
                df=single_jobs,
                filepath=self.path_results(job_name=job_name, filetype="jobs")
            )
            io.write_df(
                df=grouped_jobs,
                filepath=self.path_results(job_name=job_name, filetype="inputs")
            )
        return grouped_jobs if grouped else single_jobs

    def job_pharmacophore(self, job_name: str, pdb_id: str, job_idx: int) -> t2fpharm.Pharmacophore:
        """Get the pharmacophore for a given job and PDB ID."""
        features = io.read_pharm_df(
            dirpath=self.job_dirpath(job_name, "pharms"),
            pdb_id=pdb_id,
            job_idx=job_idx,
        )
        return t2fpharm.Pharmacophore(
            features=features,
            feature_types=self.field_params["ligand_types"],
            system=self.complex(pdb_id),
            pocket=self.pocket(pdb_id),
            field=self.field(pdb_id),
        )

    def job_matches(self, job_name: str, pdb_id: str, job_idx: int) -> pd.DataFrame:
        """Get the matches for a given job and PDB ID."""
        matches = io.read_pharm_df(
            dirpath=self.job_dirpath(job_name, "matches"),
            pdb_id=pdb_id,
            job_idx=job_idx,
        )
        return matches

    def job_dirpath(self, job_name: str, dirtype: Literal["root", "pharms", "matches"] = "root") -> Path:
        """Get the directory path for a given job."""
        root_path = self.dirpath_data / self._path["results"] / job_name
        if dirtype == "root":
            root_path.mkdir(parents=True, exist_ok=True)
            return root_path
        path = root_path / self._dirname[f"job_{dirtype}"]
        path.mkdir(parents=True, exist_ok=True)
        return path

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

    def pdb_fixed(self, pdb_id: str) -> scifile.pdb.PDBFile:
        cached = self._cache.get(pdb_id, {}).get("pdb_fixed")
        if cached:
            return cached
        filepath_pdb_fixed = self.path("pdb_fixed", pdb_id)
        if filepath_pdb_fixed.is_file():
            pdb = scifile.pdb.read(filepath_pdb_fixed)
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
            filepath_pdb_fixed.write_text(pdb_fixed_str)
            pdb = scifile.pdb.read(filepath_pdb_fixed)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pdb_fixed"] = pdb
        return pdb

    def complex(self, pdb_id: str) -> t2fpharm.System:
        cached = self._cache.get(pdb_id, {}).get("complex")
        if cached:
            return cached
        filepath_pdb_aligned = self.path("pdb_aligned", pdb_id)
        if filepath_pdb_aligned.is_file():
            pdb_aligned_str = filepath_pdb_aligned.read_text()
        else:
            filepath_pdb_fixed = self.path("pdb_fixed", pdb_id)
            if not filepath_pdb_fixed.is_file():
                self.pdb_fixed(pdb_id)
            pdb_fixed_str = filepath_pdb_fixed.read_text()
            if self.dataset.loc[pdb_id, "is_ref"]:
                pdb_aligned_str = str(t2fpharm.system.from_pdb(pdb_fixed_str).minimize_aabb().to_pdb())
            else:
                group_id = self.dataset.loc[pdb_id, "group_id"]
                is_group_ref = (self.dataset["group_id"] == group_id) & self.dataset["is_ref"]
                ref_pdb_id = self.dataset.loc[is_group_ref, "pdb_id"].iloc[0]
                complex_aligned = _align_query_to_ref(
                    ref=self.complex(ref_pdb_id),
                    query=t2fpharm.system.from_pdb(pdb_fixed_str),
                    ref_chain_id=self.dataset.loc[ref_pdb_id, "chain_id"],
                    query_chain_id=self.dataset.loc[pdb_id, "chain_id"],
                    ref_pocket_atoms=self._ref_pocket_atoms(ref_pdb_id),
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
            pocket = t2fpharm.pocket.from_npz(
                filepath=filepath_pocket,
                receptor=rcomplex,
                trim=False,
            )
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
                erosion_radius=self.pocket_params["erosion_radius"],
                opening_radius=self.pocket_params["opening_radius"],
                morphology_order=self.pocket_params.get("morphology_order", ("opening", "erosion")),
                grid=self.grid(pdb_id),
                trim=False,
            )
            if pocket.holes().any():
                raise ValueError(f"Pocket for {pdb_id} contains holes.")
            if not pocket.point_coverage(rcomplex.trajectory.points[ligand_mask.to_numpy()]).all():
                raise ValueError(f"Pocket for {pdb_id} does not cover ligand.")
            pocket.to_npz(filepath=filepath_pocket)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["pocket"] = pocket
        return pocket

    def _ref_pocket_atoms(self, ref_pdb_id: str) -> pd.DataFrame:
        cached = self._cache.get(ref_pdb_id, {}).get("pocket_atoms")
        if cached is not None:
            return cached
        rcomplex = self.complex(ref_pdb_id)
        atoms = rcomplex.composition.atoms
        ligand_res_name = self.dataset.loc[ref_pdb_id, "ligand_res_name"]
        ligand_chain_id = self.dataset.loc[ref_pdb_id, "ligand_chain_id"]
        ligand_res_seq = self.dataset.loc[ref_pdb_id, "ligand_res_seq"]
        ligand_mask = (
            (atoms["res_name"] == ligand_res_name) &
            (atoms["chain_id"] == ligand_chain_id) &
            (atoms["res_seq"] == ligand_res_seq)
        )
        if ligand_mask.sum() == 0:
            raise ValueError(
                f"Ligand {ligand_res_name} "
                f"not found in structure {ref_pdb_id}"
            )
        pocket = t2fpharm.pocket.from_ligand(
            system=rcomplex,
            ligand_mask=ligand_mask,
            ligand_radii=None,
            ligand_radii_offset=self.pocket_params["ligand_radii_offset"],
            erosion_radius=self.pocket_params["erosion_radius"],
            opening_radius=self.pocket_params["opening_radius"],
            morphology_order=self.pocket_params.get("morphology_order", ("opening", "erosion")),
            grid=self.pocket_params["grid_spacing"],
        )
        if self._cache_enabled:
            self._cache.setdefault(ref_pdb_id, {})["pocket_atoms"] = pocket.atoms
        return pocket.atoms

    def grid(self, pdb_id: str) -> t2fpharm.Grid:
        """Get the grid for a given PDB ID.

        The grid is identical for all PDB IDs in the same group.
        """
        group_id = self.group_id(pdb_id)
        if group_id in self._group_grid:
            return self._group_grid[group_id]
        group_pdb_ids = self.group_pdb_ids(group_id=group_id, include_ref=True)
        if not group_pdb_ids:
            raise ValueError(f"No PDB IDs found for group ID: {group_id}")

        lower_bounds = np.empty((len(group_pdb_ids), 3), dtype=float)
        upper_bounds = np.empty((len(group_pdb_ids), 3), dtype=float)
        for idx, pdb_id in enumerate(group_pdb_ids):
            rcomplex = self.complex(pdb_id)
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
            ligand = rcomplex.select(selection=ligand_mask)
            ligand_bounding_box = ligand.trajectory.aabb(per_instance=False)
            ligand_atoms_radii = ligand.composition.vdw_radius
            ligand_atoms_max_radius = ligand_atoms_radii.max()
            lower_bounds[idx] = ligand_bounding_box.lower_bounds - ligand_atoms_max_radius
            upper_bounds[idx] = ligand_bounding_box.upper_bounds + ligand_atoms_max_radius
        padding = self.pocket_params["ligand_radii_offset"] + self.pocket_params["grid_spacing"]
        grid = t2fpharm.grid.from_bounds_spacing(
            lower=lower_bounds.min(axis=0) - padding,
            upper=upper_bounds.max(axis=0) + padding,
            spacing=self.pocket_params["grid_spacing"],
        )
        self._group_grid[group_id] = grid
        return grid

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
            # pocket_data = self.pocket(pdb_id).to_dict()
            # grid_data = {k: v for k, v in pocket_data.items() if k.startswith("grid_")}
            filepath_pdbqt = self.path("pdbqt", pdb_id)
            if not filepath_pdbqt.is_file():
                self.pdbqt(pdb_id)
            field = t2fpharm.field.from_autogrid(
                grid=self.pocket(pdb_id).grid,
                receptor_files=filepath_pdbqt,
                receptor_file_ids=pdb_id,
                ligand_types=self.field_params["ligand_types"],
                smooth=self.field_params["smooth"],
                dielectric=self.field_params["dielectric"],
                output_dir=dirpath_autogrid,
                # **grid_data,
            )
            field.to_npz(filepath=filepath_field)
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["field"] = field
        return field

    def modeler(self, pdb_id: str) -> t2fpharm.Modeler:
        return t2fpharm.modeler(
            field=self.field(pdb_id),
            pocket=self.pocket(pdb_id),
            system=self.complex(pdb_id),
        )

    def ref_pharmacophore(self, pdb_id: str, include_extras: bool = True) -> t2fpharm.pharm.Pharmacophore:
        cached = self._cache.get(pdb_id, {}).get("ref_pharm")
        if cached:
            return cached
        filepath_ref_features = self.path("ref_features", pdb_id)
        if filepath_ref_features.is_file():
            features_data = pyserials.read.from_file(filepath_ref_features, toml_as_dict=True)
            ref_pharm = t2fpharm.pharm.Pharmacophore(
                features=features_data,
                feature_types=self.field_params["ligand_types"],
                system=self.complex(pdb_id) if include_extras else None,
                pocket=self.pocket(pdb_id) if include_extras else None,
            )
        else:
            # Get all complex-based pharmacophores for the group
            complex_pharms = [self.complex_pharmacophore(pdb_id) for pdb_id in self.group_pdb_ids(self.group_id(pdb_id))]
            # Merge all coomplex-based pharmacophores into one
            merged_pharm = t2fpharm.pharm.merge(complex_pharms)
            # Select only the features that are within the pocket
            pocket = self.pocket(pdb_id)
            merged_feats = merged_pharm.features
            feat_centers = np.stack(merged_feats["center"])
            feat_mask = pocket.point_coverage(feat_centers)
            selected_feats = merged_feats[feat_mask]
            filtered_pharm = merged_pharm.new(
                features=selected_feats,
                system=self.complex(pdb_id) if include_extras else None,
                pocket=pocket if include_extras else None,
            )
            ref_pharm = filtered_pharm.cluster_agg(
                distance_threshold=3,
                min_members=1,
                noise_as_singleton=True,
                center_type="mean",
                radius_type="max",
                per_instance=False,
            )
            filepath_ref_features.write_text(
                ref_pharm.features.to_json(orient="records", indent=4)
            )
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["ref_pharm"] = ref_pharm
        return ref_pharm

    def complex_pharmacophore(self, pdb_id: str):
        cached = self._cache.get(pdb_id, {}).get("complex_pharm")
        if cached:
            return cached
        filepath_ligand_plip = self.path("ligand_plip", pdb_id)
        filepath_ligand_features = self.path("ligand_features", pdb_id)
        if filepath_ligand_plip.is_file() and filepath_ligand_features.is_file():
            plip_data = pyserials.read.from_file(filepath_ligand_plip, toml_as_dict=True)
            features_data = pyserials.read.from_file(filepath_ligand_features, toml_as_dict=True)
            plip_df = pd.DataFrame(plip_data)
            complex_pharm = t2fpharm.pharm.Pharmacophore(
                features=features_data,
                feature_types=self.field_params["ligand_types"],
                extra={"plip": caddpy.interaction.ProteinLigandInteractions(plip_df)},
                system=self.complex(pdb_id),
            )
        else:
            complex_pharm = t2fpharm.pharm.from_complex(
                pdb_files=self.path("pdb_aligned", pdb_id),
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
                complex_pharm.extra["plip"].all.to_json(orient="records", indent=4)
            )
            filepath_ligand_features.write_text(
                complex_pharm.features.to_json(orient="records", indent=4)
            )
        if self._cache_enabled:
            self._cache.setdefault(pdb_id, {})["complex_pharm"] = complex_pharm
        return complex_pharm

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
            "ref_features",
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
            "jobs",
            "inputs",
            "summary",
            "summary_final",
        ],
    ) -> Path:
        path = self.dirpath_data / self._path["results"] / job_name
        path.mkdir(parents=True, exist_ok=True)
        if filetype == "jobs":
            return path / "jobs.parquet"
        if filetype == "inputs":
            return path / ".job_inputs.parquet"
        if filetype == "summary":
            return path / "summary.jsonl"
        if filetype == "summary_final":
            return path / "summary.parquet"
        raise ValueError(f"Unknown filetype: {filetype}")

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

    def ref_pdb_id(self, group_id: str) -> str:
        """Get the reference PDB ID for a given group ID."""
        return self.dataset[(self.dataset["group_id"]==group_id) & self.dataset["is_ref"]].index[0]

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
    ref_pocket_atoms: pd.DataFrame,
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
    ref_pocket_c_alpha_atoms = ref_pocket_atoms[ref_pocket_atoms["name"]=="CA"]
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
