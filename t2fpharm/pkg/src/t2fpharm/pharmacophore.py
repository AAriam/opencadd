
from typing import Any, Sequence, Self



import numpy as np
import pandas as pd
import scishow

from t2fpharm.field import Field
from t2fpharm.pocket import Pocket
from t2fpharm.ligand import LigandPharmacophore


class Pharmacophore:

    def __init__(self):
        self._feature_colors = {
            "HD": (0, 0.6, 0),
            "OA": (0.6, 0, 0),
            "C": (1.0, 1.0, 0),
            "e+": (0, 0, 1.0),
            "e-": (1.0, 0, 0),
        }
        return

    def set_feature_color(self, **kwargs: tuple[int, int, int] | tuple[float, float, float]) -> None:
        """Set custom colors for pharmacophore features.

        Parameters
        ----------
        **kwargs
            Feature types as keys and RGB color tuples as values.
            Each color can be a tuple of three integers (0-255) or floats (0.0-1.0).
            Example: `set_feature_color(HD=(0, 255, 0), OA=(255, 0, 0))`
        """
        for feature, color in kwargs.items():
            if isinstance(color, Sequence) and len(color) == 3:
                if all(isinstance(c, (int, float)) for c in color):
                    self._feature_colors[feature] = tuple(color)
                else:
                    raise ValueError(f"Invalid color format for feature '{feature}': {color}")
            else:
                raise ValueError(f"Color must be a tuple of three values for feature '{feature}'")
        return



class ReceptorPharmacophore(Pharmacophore):
    def __init__(
        self,
        features: pd.DataFrame,
        pocket: Pocket,
        field: Field,
        args,
        extra: dict[str, Any] | None = None
    ):
        self.features = features
        self.pocket = pocket
        self.field = field
        self.args = args
        self.extra = extra or {}
        super().__init__()
        return

    def match_spherical(self, ligand: LigandPharmacophore) -> pd.DataFrame:
        if not isinstance(ligand, LigandPharmacophore):
            raise TypeError(
                f"Expected LigandPharmacophore, got {type(ligand)}"
            )
        ligand = ligand.features
        ligand_features = set(ligand['type'])
        self_features = set(self.field.batch_instance_labels["feature"])
        if (invalid_features := ligand_features - self_features):
            raise ValueError(
                f"Invalid feature values: {sorted(invalid_features)}. "
                f"Allowed: {sorted(self_features)}"
            )
        ligand = ligand.reset_index().rename(columns={'index': 'ligand_idx'})

        # Get all unique instances
        instances = pd.DataFrame({'instance': self.features['instance'].unique()})
        # Cross-join ligand × instance
        ligand['_key'] = 1
        instances['_key'] = 1
        cross = ligand.merge(instances, on='_key').drop(columns=['_key'])
        # Merge with features on instance & type
        feat = self.features.rename(columns={'radius': 'radius_feat'})
        merged = cross.merge(
            feat[['instance', 'type', 'label', 'center', 'radius_feat']],
            on=['instance', 'type'],
            how='left'
        ).rename(columns={'radius': 'radius_lig'})
        # Compute distances where feature exists
        mask = merged['center'].notna()
        if mask.any():
            pos_arr = np.stack(merged.loc[mask, 'position'].values)
            cen_arr = np.stack(merged.loc[mask, 'center'].values)
            merged.loc[mask, 'distance'] = np.linalg.norm(pos_arr - cen_arr, axis=1)
            merged.loc[mask, 'max_distance'] = (
                merged.loc[mask, 'radius_lig'] + merged.loc[mask, 'radius_feat']
            )
            merged.loc[mask, 'match'] = merged.loc[mask, 'distance'] < merged.loc[mask, 'max_distance']
        # Defaults for missing-feature cases
        merged['distance'] = merged['distance'].astype(float)
        merged['max_distance'] = merged['max_distance'].astype(float)
        merged['match'] = merged['match'].fillna(False)
        # Pick minimum-distance feature per ligand_idx×instance
        # Treat NaN distances as +inf so real distances sort first
        merged['dist_sort'] = merged['distance'].fillna(np.inf)
        best = (
            merged
            .sort_values(['ligand_idx', 'instance', 'dist_sort'])
            .drop_duplicates(['ligand_idx', 'instance'], keep='first')
            .drop(columns=['dist_sort', 'position', 'center', 'radius_lig', 'radius_feat'])
        )
        # Reorder & return
        final_cols = ['ligand_idx', 'instance', 'label', 'type', 'match', 'distance', 'max_distance']
        return best[final_cols].reset_index(drop=True)

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        receptor: Any | None = None,
        show_box: bool = True,
        show_pocket: bool = True,
        show_fields: bool = False,
        show_feature_points: bool = False,
        show_feature_centers: bool = True,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
    ):
        def feature_color(feature_id: str) -> tuple[float, float, float] | tuple[int, int, int]:
            """Get color for a feature type, defaulting to gray if not set."""
            return feature_colors.get(
                feature_id,
                self._feature_colors.get(feature_id, (0.5, 0.5, 0.5))
            )

        nv = nglwidget or scishow.nglview.NGLWidget()
        if receptor:
            nv.add_trajectory(receptor)
        self.pocket.display(
            nglwidget=nv,
            show_box=show_box,
            visible=show_pocket,
        )
        feature_colors = feature_colors or {}
        for feature_id in self.field.batch_instance_labels["feature"]:
            nv.add_volume(
                data=self.field(feature=feature_id),
                basis=self.field.grid.unit_vectors,
                origin=self.field.grid.lower_bounds,
                name=f"{feature_id.upper()} Field",
                representation_params=scishow.nglview.SurfaceRepresentationParameters(
                    isolevel=0,
                    isolevel_type="value",
                    contour=False,
                    wireframe=True,
                    color=feature_color(feature_id),
                    visible=show_fields,
                )
            )
        for _, feature in self.features.iterrows():
            nv.add_spheres(
                coords=feature["points"],
                radii=self.field.grid.spacings[0] / 2,
                name=f"{feature['type'].upper()}_{feature['label']} Points",
                colors=feature_color(feature["type"]),
                representation_params=scishow.nglview.RepresentationParameters(
                    visible=show_feature_points,
                )
            )
            nv.add_spheres(
                coords=[feature["center"]],
                radii=feature["radius"],
                name=f"{feature['type'].upper()}_{feature['label']} Center",
                colors=feature_color(feature["type"]),
                representation_params=scishow.nglview.RepresentationParameters(
                    opacity=0.8,
                    visible=show_feature_centers,
                    lazy=True,
                )
            )
        return nv.display(gui=True)
