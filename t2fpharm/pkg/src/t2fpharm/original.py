from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd


import scifile
import scids

from t2fpharm.input import T2FInput, CNNClusteringConfig


class Input(T2FInput):
    pdb_id: str | None = None
    pdb_file_raw: Path | bytes | str | scifile.pdb.PDBFile | None = None
    pdb_file_fixed: Path | bytes | str | scifile.pdb.PDBFile | None = None
    pdb_file_apo: Path | bytes | str | scifile.pdb.PDBFile | None = None


    clustering: CNNClusteringConfig = CNNClusteringConfig(
        max_distance=1.21,
        min_neighbors=(6, 12, 16),
        min_members=15,
        max_members=80,
    )


class T2FPharm:
    def __init__(self, input: Input):

        self._input = input

        self._pdb_file_raw = None
        self._pdb_file_fixed = None
        self._pdb_file_apo = None

        self._mask_pocket: scids.field.Field = None
        self._mask_energy: scids.field.Field = None
        self._mask_final: scids.field.Field = None
        self._feature_grid_point_position: dict[str, jax.Array] = {}
        self._feature_grid_point_label: dict[str, jax.Array] = {}
        self._pharmacophore: pd.DataFrame = None
        return

    @property
    def pharmacophore(self) -> pd.DataFrame:
        """Final pharmacophore features."""
        if self._pharmacophore is None:
            all_positions = self.feature_grid_point_position
            all_labels = self.feature_grid_point_label
            rows = []
            for feature_type, feature_pointcloud in all_positions.items():
                cluster_labels = all_labels[feature_type]
                cluster_sizes = jnp.bincount(cluster_labels)[1:]  # Exclude background label (0)
                cluster_count = cluster_sizes.size
                for cluster_label in range(1, cluster_count + 1):
                    point_coordinates = feature_pointcloud.points[cluster_labels == cluster_label]
                    rows.append(
                        {
                            "type": feature_type,
                            "n_points": cluster_sizes[cluster_label - 1],
                            "center": point_coordinates.mean(axis=0),
                            "label": cluster_label,
                            "points": point_coordinates,
                        }
                    )
            self._pharmacophore = pd.DataFrame(rows)
            self._pharmacophore.set_index(["type", "label"], inplace=True)
        return self._pharmacophore

    @property
    def feature_grid_point_label(self) -> dict[str, jax.Array]:
        """Cluster labels for pharmacophore feature points."""
        if not self._feature_grid_point_label:
            position = self.feature_grid_point_position
            cnn_config = self._input.clustering.model_dump()
            for feature_type, feature_pointcloud in position.items():
                labels = jnp.asarray(feature_pointcloud.cluster_cnn(**cnn_config))
                self._feature_grid_point_label[feature_type] = labels
        return self._feature_grid_point_label

    @property
    def feature_grid_point_position(self) -> dict[str, scids.pointcloud.PointCloud]:
        """Grid point positions for pharmacophore features."""
        if not self._feature_grid_point_position:
            final_mask = self.mask_final
            grid = self.grid
            feature = self.feature
            for feature_type in feature["type"].values:
                feature_mask = final_mask(feature=feature_type)
                feature_pointcloud = scids.pointcloud.from_array(grid.coordinates[feature_mask])
                self._feature_grid_point_position[feature_type] = feature_pointcloud
        return self._feature_grid_point_position

    @property
    def mask_final(self) -> scids.field.Field:
        """Final mask for pharmacophore features."""
        if self._mask_final is None:
            energy_mask = self.mask_energy
            pocket_mask = self.mask_pocket
            final_mask = jnp.logical_and(energy_mask.tensor, pocket_mask.tensor)
            self._mask_final = scids.field.Field(
                tensor=final_mask,
                grid=self.grid,
                batch=[(k, v) for k, v in energy_mask.batch_instance_labels.items()],
            )
        return self._mask_final

    @property
    def mask_energy(self) -> scids.field.Field:
        """Energy mask for pharmacophore features."""
        if self._mask_energy is None:
            energy = self.energy
            feature_data = self.feature
            max_energies = [
                feature_data.loc[feature_data['id'] == fid, 'max_energy'].iat[0]
                for fid in energy.batch_instance_labels["feature"]
            ]
            energy_mask = jnp.less_equal(energy.tensor, jnp.array(max_energies).reshape(-1, 1, 1, 1))
            self._mask_energy = scids.field.Field(
                tensor=energy_mask,
                grid=self.grid,
                batch=[(k, v) for k, v in energy.batch_instance_labels.items()],
            )
        return self._mask_energy

    @property
    def mask_pocket(self) -> scids.field.Field:
        """Pocket mask for pharmacophore features."""

        return

    @property
    def energy(self) -> scids.field.Field:
        """Energy field for pharmacophore features."""
        return

    @property
    def feature(self) -> pd.DataFrame:
        """Pharmacophore feature types."""
        return

    @property
    def grid(self) -> scids.grid.Grid:
        """Grid for pharmacophore features."""
        return
