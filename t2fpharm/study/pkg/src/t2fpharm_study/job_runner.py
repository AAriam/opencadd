from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import pandas as pd

from t2fpharm import Pharmacophore


if TYPE_CHECKING:
    from typing import Any, Sequence
    from t2fpharm import Modeler
    from t2fpharm_study.typing import PDBID


def run(
    modeler: Modeler,
    ligand_pharms: dict[PDBID, Pharmacophore],
    target_pdb_id: str,
    job_idx: int,
    method: str,
    kwargs: dict[str, Any],
    feature_types: list[str],
    filepath_features: Path | str | None = None,
    filepath_matches: Path | str | None = None,
    return_pharm: bool = True,
    return_matches: bool = True,
):
    try:
        # Calculate target pharmacophore
        func = getattr(modeler, method)
        target_pharm = func(**{k: v for k, v in kwargs.items() if k != "min_members_percents"})
        # Save target pharmacophore
        if filepath_features:
            target_pharm.features.to_parquet(
                path=Path(filepath_features),
                engine="pyarrow",
                compression="zstd",
                compression_level=3,
                index=False,
            )

        # Generate target pharmacophore summary
        summary, matches = _analyze(
            target_pharm=target_pharm,
            ligand_pharms=ligand_pharms,
            target_pdb_id=target_pdb_id,
            method=method,
            feature_types=feature_types,
            max_members=kwargs.get("max_members"),
            min_members_percents=kwargs.get("min_members_percents"),
            filepath_matches=filepath_matches,
        )
        summary["job-idx"] = job_idx
    except Exception as e:
        raise RuntimeError(
            f"Error running job {job_idx} for PDB ID {target_pdb_id} with method {method}: {e}"
        ) from e

    # Prepare output
    output = [summary]
    if return_pharm:
        output.append(target_pharm)
    if return_matches:
        matches["job_idx"] = job_idx
        output.append(matches)
    return tuple(output)


def _analyze(
    target_pharm: Pharmacophore,
    ligand_pharms: dict[PDBID, Pharmacophore],
    target_pdb_id: PDBID,
    method: str,
    feature_types: list[str],
    max_members: dict[str, int] | None = None,
    min_members_percents: Sequence[float] | None = None,
    filepath_matches: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if method == "largest_peaks":
        pharm_summary = _pharm_summary_lp(
            features=target_pharm.features,
            feature_types=feature_types,
        )
        # Calculate matches with ligand pharmacophores
        matches = _matches_df(
            target_pharm=target_pharm,
            ligand_pharms=ligand_pharms,
            target_pdb_id=target_pdb_id,
            method=method,
        )
        matches_summary = _calculate_matches_summary(matches)
    else:
        pharm_summary = _pharm_summary_cnn(
            features=target_pharm.features,
            feature_types=feature_types,
            cluster_min_members_id="1"
        )
        matches_all = _matches_df(
            target_pharm=target_pharm,
            ligand_pharms=ligand_pharms,
            target_pdb_id=target_pdb_id,
            method=method,
        )
        matches_all["min_members"] = "1"
        matches_summary = _calculate_matches_summary(
            matches=matches_all,
            cluster_min_members_id="1"
        )
        matches_dfs = [matches_all]
        for percent in (min_members_percents or []):
            min_members = {
                feature_type: int((percent / 100) * feature_max_members)
                for feature_type, feature_max_members in max_members.items()
            }
            min_members_per_feature = target_pharm.features["type"].map(min_members)
            mask = target_pharm.features["n_members"] >= min_members_per_feature
            allowed_features = target_pharm.features[mask]
            pharm_summary.update(
                _pharm_summary_cnn(
                    features=allowed_features,
                    feature_types=feature_types,
                    cluster_min_members_id=f"{percent}%"
                )
            )
            subset_pharm = Pharmacophore(
                features=allowed_features,
                feature_types=target_pharm.feature_types,
            )
            matches_df = _matches_df(
                target_pharm=subset_pharm,
                ligand_pharms=ligand_pharms,
                target_pdb_id=target_pdb_id,
                method=method,
            )
            matches_df["min_members"] = f"{percent}%"
            matches_dfs.append(matches_df)
            matches_summary.update(
                _calculate_matches_summary(
                    matches=matches_df,
                    cluster_min_members_id=f"{percent}%"
                )
            )
        matches = pd.concat(matches_dfs, ignore_index=True)

    if filepath_matches:
        matches.to_parquet(
            path=Path(filepath_matches),
            engine="pyarrow",
            compression="zstd",
            compression_level=3,
            index=False,
        )

    pharm_summary["pdb_id"] = target_pdb_id
    summary = pharm_summary | matches_summary
    return summary, matches


def _pharm_summary_lp(
    features: pd.DataFrame,
    feature_types: list[str],
):
    summary = _pharm_summary_feature_counts(
        features=features,
        feature_types=feature_types,
    )
    if len(features) == 0:
        return summary
    available_feature_types = features["type"].unique()
    values = features["value"]
    summary.update(
        {
            "value-all_types-min": float(values.min()),
            "value-all_types-max": float(values.max()),
            "value-all_types-mean": float(values.mean()),
        }
    )
    for feature_type in available_feature_types:
        values = features[features["type"] == feature_type]["value"]
        summary.update(
            {
                f"value-{feature_type}-min": float(values.min()),
                f"value-{feature_type}-max": float(values.max()),
                f"value-{feature_type}-mean": float(values.mean()),
            }
        )
    return summary


def _pharm_summary_cnn(
    features: pd.DataFrame,
    feature_types: list[str],
    cluster_min_members_id: str,
):
    def calculate(
        values: pd.Series,
        radii: pd.Series,
        feature_type: str,
        center_type: str,
    ) -> dict[str, float]:
        return {
            f"value-{cluster_min_members_id}_member-{feature_type}-{center_type}-min": float(values.min()),
            f"value-{cluster_min_members_id}_member-{feature_type}-{center_type}-max": float(values.max()),
            f"value-{cluster_min_members_id}_member-{feature_type}-{center_type}-mean": float(values.mean()),
            f"radius-{cluster_min_members_id}_member-{feature_type}-{center_type}-min": float(radii.min()),
            f"radius-{cluster_min_members_id}_member-{feature_type}-{center_type}-max": float(radii.max()),
            f"radius-{cluster_min_members_id}_member-{feature_type}-{center_type}-mean": float(radii.mean()),
        }

    summary = _pharm_summary_feature_counts(
        features=features,
        feature_types=feature_types,
        cluster_min_members_id=cluster_min_members_id,
    )
    if len(features) == 0:
        return summary
    available_feature_types = features["type"].unique()
    for center_type in ("midpoint", "mean", "average"):
        values = features[f"value_{center_type}"]
        radii = features[f"radius_{center_type}_max"]
        summary.update(calculate(values, radii, feature_type="all_types", center_type=center_type))
        for feature_type in available_feature_types:
            type_mask = features["type"] == feature_type
            type_values = values[type_mask]
            type_radii = radii[type_mask]
            summary.update(calculate(type_values, type_radii, feature_type=feature_type, center_type=center_type))
    return summary


def _pharm_summary_feature_counts(
    features: pd.DataFrame,
    feature_types: list[str],
    cluster_min_members_id: str | None = None,
):
    def name(feature_type: str) -> str:
        return f"n-{cluster_min_members_id}_member-{feature_type}" if cluster_min_members_id else f"n-{feature_type}"
    return {name("all_types"): len(features)} | {
        name(feature_type): int(features["type"].eq(feature_type).sum())
        for feature_type in feature_types
    }


def _matches_df(
    target_pharm: Pharmacophore,
    ligand_pharms: dict[PDBID, Pharmacophore],
    target_pdb_id: PDBID,
    method: str,
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
        if method == "largest_peaks":
            matches = calculate(target_pharm=target_pharm, ligand_pharm=ligand_pharm)
        else:
            center_type_matches_df = []
            target_features = target_pharm.features.copy()
            for center_type in ("midpoint", "mean", "average"):
                target_features["center"] = target_features[f"center_{center_type}"]
                target_pharm_center = Pharmacophore(
                    features=target_features,
                    feature_types=target_pharm.feature_types,
                )
                matches_center = calculate(
                    target_pharm=target_pharm_center,
                    ligand_pharm=ligand_pharm
                )
                matches_center["center_type"] = center_type
                center_type_matches_df.append(matches_center)
            matches = pd.concat(center_type_matches_df, ignore_index=True)
        matches["ligand_pdb_id"] = ligand_pdb_id
        matches_dfs.append(matches)
    matches_df = pd.concat(matches_dfs, ignore_index=True)
    matches_df["target_pdb_id"] = target_pdb_id
    return matches_df


def _calculate_matches_summary(
    matches: pd.DataFrame,
    cluster_min_members_id: str | None = None,
) -> dict[str, Any]:
    matches_total = _calculate_match_summary(
        matches,
        ligand_type="all_ligands",
        cluster_min_members_id=cluster_min_members_id
    )
    matches_self = _calculate_match_summary(
        matches[matches["ligand_pdb_id"] == matches["target_pdb_id"]],
        ligand_type="self_ligand",
        cluster_min_members_id=cluster_min_members_id
    )
    return matches_total | matches_self


def _calculate_match_summary(
    matches: pd.DataFrame,
    ligand_type: str,
    cluster_min_members_id: str | None = None,
) -> dict[str, Any]:
    out = _calculate_feature_match_summary(
        matches,
        ligand_type=ligand_type,
        feature_type="all_types",
        cluster_min_members_id=cluster_min_members_id
    )
    for feature_type in matches["type"].unique():
        out_type = _calculate_feature_match_summary(
            matches,
            ligand_type=ligand_type,
            feature_type=feature_type,
            cluster_min_members_id=cluster_min_members_id
        )
        out.update(out_type)
    return out


def _calculate_feature_match_summary(
    matches: pd.DataFrame,
    ligand_type: str,
    feature_type: str,
    cluster_min_members_id: str | None = None,
) -> dict[str, Any]:
    n_ligand_features = len(matches)
    out = {f"match-{ligand_type}-{feature_type}-count": n_ligand_features}
    if "center_type" in matches:
        for center_type in matches["center_type"].unique():
            summary = _calculate_ligand_feature_center_match_summary(
                matches[matches["center_type"] == center_type],
                ligand_type=ligand_type,
                feature_type=feature_type,
                center_type=f"{center_type}_center",
                cluster_min_members_id=cluster_min_members_id,
            )
            out.update(summary)
    else:
        summary = _calculate_ligand_feature_center_match_summary(
            matches=matches,
            ligand_type=ligand_type,
            feature_type=feature_type,
            cluster_min_members_id=cluster_min_members_id,
        )
        out.update(summary)
    return out


def _calculate_ligand_feature_center_match_summary(
    matches: pd.DataFrame,
    ligand_type: str,
    feature_type: str,
    center_type: str | None = None,
    cluster_min_members_id: str | None = None,
) -> dict[str, Any]:
    def name(dist_type: str) -> str:
        out = "match"
        if cluster_min_members_id:
            out += f"-{cluster_min_members_id}_min_members"
        out += f"-{ligand_type}-{feature_type}"
        if center_type:
            out += f"-{center_type}"
        out += f"-dist-{dist_type}"
        return out
    n_ligand_features = len(matches)
    dists = matches["distance"]
    out = {
        name("min"): float(dists.min()),
        name("max"): float(dists.max()),
        name("mean"): float(dists.mean()) if dists.notna().any() else float("nan"),
        name("median"): float(dists.median()) if dists.notna().any() else float("nan"),
        name("inf"): float(dists.isna().sum()) / n_ligand_features,
        name("lt1"): float((dists < 1.0).sum()) / n_ligand_features,
        name("lt2"): float((dists < 2.0).sum()) / n_ligand_features,
        name("lt3"): float((dists < 3.0).sum()) / n_ligand_features,
    }
    return out
