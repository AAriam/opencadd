from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
import warnings

import pandas as pd

from t2fpharm import Pharmacophore

from t2fpharm_study.io import write_pharm_df

if TYPE_CHECKING:
    from typing import Any, Literal
    from t2fpharm import Modeler
    from t2fpharm_study.typing import PDBID


def run(
    modeler: Modeler,
    ligand_pharms: dict[PDBID, Pharmacophore],
    target_pdb_id: str,
    method: Literal["largest_peaks", "cnn"],
    job: dict[str, Any],
    dirpath_features: Path | str | None = None,
    dirpath_matches: Path | str | None = None,
    return_pharm: bool = True,
    return_matches: bool = True,
):
    try:
        summaries, pharms, matches = _run(
            modeler=modeler,
            ligand_pharms=ligand_pharms,
            target_pdb_id=target_pdb_id,
            method=method,
            job=job,
            dirpath_features=dirpath_features,
            dirpath_matches=dirpath_matches,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error running job {job["job_idx"]} for PDB ID {target_pdb_id} with method {method}: {e}"
        ) from e
    # Prepare output
    output = [summaries]
    if return_pharm:
        output.append(pharms)
    if return_matches:
        output.append(matches)
    return tuple(output)


def _run(
    modeler: Modeler,
    ligand_pharms: dict[PDBID, Pharmacophore],
    target_pdb_id: str,
    method: Literal["largest_peaks", "cnn"],
    job: dict[str, Any],
    dirpath_features: Path | str | None = None,
    dirpath_matches: Path | str | None = None,
):
    summaries = []
    pharms = []
    matches = []
    if method == "largest_peaks":
        pharm = modeler.largest_peaks(
            min_distance=job["min_distance"],
            priority_factor=job["priority_factor"],
            max_features=job["max_features"],
            filter_function=job.get("filter_function"),
            filter_radius=job.get("filter_radius"),
            filter_extension_mode="constant",
            filter_extension_constant_value=0,
            filter_gaussian_sigma=job.get("filter_gaussian_sigma"),
            filter_percentile=job.get("filter_percentile", 0),
            peak_type=job.get("peak_type", "min"),
            best_per_point=False,
            threshold_value=job.get("threshold_value"),
            threshold_percentile=job.get("threshold_percentile"),
            threshold_include_equal=False,
        )
        summary, pharm, match = _run_single(
            pharm=pharm,
            ligand_pharms=ligand_pharms,
            target_pdb_id=target_pdb_id,
            job_idx=job["job_idx"],
            include_radii=False,
            dirpath_features=dirpath_features,
            dirpath_matches=dirpath_matches,
        )
        summaries.append(summary)
        pharms.append(pharm)
        matches.append(match)
        return summaries, pharms, matches
    pharm_base = modeler.cnn(
        max_distance=job["max_distance"],
        min_neighbors=job["min_neighbors"],
        min_members=1,
        max_members=job["max_members"],
        filter_function=job.get("filter_function"),
        filter_radius=job.get("filter_radius"),
        filter_extension_mode="constant",
        filter_extension_constant_value=0,
        filter_gaussian_sigma=job.get("filter_gaussian_sigma"),
        filter_percentile=job.get("filter_percentile", 0),
        peak_type=job.get("peak_type", "min"),
        best_per_point=job["best_per_point"],
        threshold_value=job.get("threshold_value"),
        threshold_percentile=job.get("threshold_percentile"),
        threshold_include_equal=False,
    )
    features = pharm_base.features
    job_idx = job["job_idx"]
    for min_members in job["min_members_dicts"]:
        min_members_per_feature = features["type"].map(min_members)
        mask = features["n_members"] >= min_members_per_feature
        features_filtered = features[mask].copy()
        for center_type in job["center_types"]:
            features_filtered["value"] = features_filtered[f"value_{center_type}"]
            features_filtered["center"] = features_filtered[f"center_{center_type}"]
            features_filtered["radius"] = features_filtered[f"radius_{center_type}_max"]
            features_final = features_filtered[["instance", "type", "label", "center", "radius", "value", "n_members"]]
            pharm = Pharmacophore(
                features=features_final,
                feature_types=pharm_base.feature_types,
            )
            summary, pharm, match = _run_single(
                pharm=pharm,
                ligand_pharms=ligand_pharms,
                target_pdb_id=target_pdb_id,
                job_idx=job_idx,
                include_radii=True,
                dirpath_features=dirpath_features,
                dirpath_matches=dirpath_matches,
            )
            summaries.append(summary)
            pharms.append(pharm)
            matches.append(match)
            job_idx += 1
    return summaries, pharms, matches


def _run_single(
    pharm: Pharmacophore,
    ligand_pharms: dict[PDBID, Pharmacophore],
    target_pdb_id: str,
    job_idx: int,
    include_radii: bool,
    dirpath_features: Path | str | None = None,
    dirpath_matches: Path | str | None = None,
):
    pharm_summary = _pharm_summary(
        features=pharm.features,
        feature_types=pharm.feature_types,
        include_radii=include_radii,
    )
    matches = _match_df(
        target_pharm=pharm,
        ligand_pharms=ligand_pharms,
    )
    matches["target_pdb_id"] = target_pdb_id
    matches_summary = _match_summary(matches)

    pharm_summary["pdb_id"] = target_pdb_id
    summary = pharm_summary | matches_summary

    summary["job_idx"] = job_idx
    matches["job_idx"] = job_idx

    if dirpath_features:
        write_pharm_df(
            df=pharm.features,
            dirpath=dirpath_features,
            pdb_id=target_pdb_id,
            job_idx=job_idx,
        )
    if dirpath_matches:
        write_pharm_df(
            df=matches,
            dirpath=dirpath_matches,
            pdb_id=target_pdb_id,
            job_idx=job_idx,
        )
    return summary, pharm, matches


def _pharm_summary(
    features: pd.DataFrame,
    feature_types: list[str],
    include_radii: bool,
):
    n_features = len(features)
    summary = {"t_all-n": n_features} | {
        f"t_{feature_type}-n": int(features["type"].eq(feature_type).sum())
        for feature_type in feature_types
    }
    if n_features == 0:
        return summary
    available_feature_types = features["type"].unique()
    summary_all_types = _pharm_summary_feature_type(
        values=features["value"],
        radii=features["radius"] if include_radii else None,
        feature_type="all",
    )
    summary.update(summary_all_types)
    for feature_type in available_feature_types:
        type_features = features[features["type"] == feature_type]
        type_summary = _pharm_summary_feature_type(
            values=type_features["value"],
            radii=type_features["radius"] if include_radii else None,
            feature_type=feature_type,
        )
        summary.update(type_summary)
    return summary


def _pharm_summary_feature_type(
    values: pd.Series,
    feature_type: str,
    radii: pd.Series | None = None,
):
    summary = {
        f"t_{feature_type}-v_min": float(values.min()),
        f"t_{feature_type}-v_max": float(values.max()),
        f"t_{feature_type}-v_mean": float(values.mean()),
    }
    if radii is not None:
        summary_radii = {
            f"t_{feature_type}-r_min": float(radii.min()),
            f"t_{feature_type}-r_max": float(radii.max()),
            f"t_{feature_type}-r_mean": float(radii.mean()),
        }
        summary.update(summary_radii)
    return summary


def _match_df(
    target_pharm: Pharmacophore,
    ligand_pharms: dict[PDBID, Pharmacophore],
) -> pd.DataFrame:
    """Calculate matches between receptor and ligand pharmacophores."""
    def calculate(
        target_pharm: Pharmacophore,
        ligand_pharm: Pharmacophore
    ) -> pd.DataFrame:
        return (
            target_pharm.match(ligand_pharm, max_distance=None)
            .drop(columns=["instance", "target_instance", "radius_sum"])
            .rename(columns={"label": "ligand_label"})
        )
    matches_dfs = []
    for ligand_pdb_id, ligand_pharm in ligand_pharms.items():
        matches = calculate(target_pharm=target_pharm, ligand_pharm=ligand_pharm)
        matches["ligand_pdb_id"] = ligand_pdb_id
        matches_dfs.append(matches)
    with warnings.catch_warnings():
        # Ignore 'The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.'
        warnings.simplefilter("ignore", FutureWarning)
        match_df = pd.concat(matches_dfs, ignore_index=True)
    return match_df


def _match_summary(matches: pd.DataFrame) -> dict[str, Any]:
    matches_total = _match_summary_ligand_type(
        matches,
        ligand_type="all",
    )
    matches_self = _match_summary_ligand_type(
        matches[matches["ligand_pdb_id"] == matches["target_pdb_id"]],
        ligand_type="self",
    )
    return matches_total | matches_self


def _match_summary_ligand_type(
    matches: pd.DataFrame,
    ligand_type: str,
) -> dict[str, Any]:
    out = _match_summary_feature_type(
        matches,
        ligand_type=ligand_type,
        feature_type="all",
    )
    for feature_type in matches["type"].unique():
        out_type = _match_summary_feature_type(
            matches,
            ligand_type=ligand_type,
            feature_type=feature_type,
        )
        out.update(out_type)
    return out


def _match_summary_feature_type(
    matches: pd.DataFrame,
    ligand_type: str,
    feature_type: str,
) -> dict[str, Any]:
    def name(dist_type: str) -> str:
        if dist_type in ("min", "max", "mean", "median"):
            return f"t_{feature_type}-d_{dist_type}-l_{ligand_type}"
        return f"t_{feature_type}-dp_{dist_type}-l_{ligand_type}"

    n_ligand_features = len(matches)
    dists = matches["distance"]
    return {
        f"t_{feature_type}-nl_{ligand_type}": n_ligand_features,
        name("min"): float(dists.min()),
        name("max"): float(dists.max()),
        name("mean"): float(dists.mean()) if dists.notna().any() else float("nan"),
        name("median"): float(dists.median()) if dists.notna().any() else float("nan"),
        name("inf"): float(dists.isna().sum()) / n_ligand_features,
        name("lt1"): float((dists < 1.0).sum()) / n_ligand_features,
        name("lt2"): float((dists < 2.0).sum()) / n_ligand_features,
        name("lt3"): float((dists < 3.0).sum()) / n_ligand_features,
    }
