from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import scipy as sp
import pandas as pd

import caddpy
import scids
import scishow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal
    from scids.grid import Grid
    from caddpy.pocket.pocket import Pocket
    from caddpy.typing import JAXArray, ArrayLike
    from caddpy.chemsys import ChemicalSystem


class Pockets:
    def __init__(
        self,
        grid: Grid,
        pocket_labels: JAXArray,
        subpocket_labels: JAXArray | None = None,
        subpocket_parent_labels: dict[int, int] | None = None,
        batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
        external_data: pd.DataFrame | None = None,
        receptor: ChemicalSystem | None = None,
    ):
        def process_labels(labels: JAXArray, parent_labels: dict[int, int] | None = None) -> pd.DataFrame:
            label_set = jnp.unique(labels)
            num_points_per_label = jnp.bincount(labels.ravel())
            if label_set[0] == 0:
                label_set = label_set[1:]
                num_points_per_label = num_points_per_label[1:]
            num_points_per_label = num_points_per_label[num_points_per_label > 0]
            volumes = num_points_per_label * grid.point_volume
            slices = sp.ndimage.find_objects(labels)
            pockets = []
            parents = []
            for label in label_set:
                pocket_slices = slices[label - 1]
                pocket_tensor = labels[pocket_slices] == label
                pocket_lower_bound_idx = tuple(s.start for s in pocket_slices)
                pocket_lower_bound = grid.coordinates[pocket_lower_bound_idx]
                pocket_grid = scids.grid.from_anchor_shape_spacing(
                    shape=pocket_tensor.shape,
                    spacing=grid.spacings,
                    anchor_type="lower",
                    anchor_coord=pocket_lower_bound,
                )
                pocket = caddpy.pocket.Pocket(
                    tensor=pocket_tensor,
                    grid=pocket_grid,
                    batch=batch,
                    receptor=receptor,
                )
                pockets.append(pocket)
                parents.append(
                    parent_labels[int(label)] if parent_labels is not None else label
                )
            return pd.DataFrame(
                {
                    "label": label_set,
                    "volume": volumes,
                    "point_count": num_points_per_label,
                    "is_subpocket": parent_labels is not None,
                    "parent_label": parents,
                    "pocket": pockets,
                }
            )

        self._grid = grid
        self._label_pockets = pocket_labels
        self._label_subpockets = subpocket_labels
        self._receptor = receptor
        self._external_data = external_data

        self._pockets = process_labels(pocket_labels)
        if subpocket_labels is not None:
            if subpocket_parent_labels is None:
                raise ValueError(
                    "Subpocket parent labels must be provided if subpocket labels are given."
                )
            subpockets = process_labels(subpocket_labels, subpocket_parent_labels)
            self._pockets = pd.concat([self._pockets, subpockets], ignore_index=True)
        self._pockets.set_index("label", inplace=True, drop=False)
        return

    @property
    def pockets(self) -> pd.DataFrame:
        """The labels of the pockets."""
        return self._pockets

    @property
    def grid(self) -> Grid:
        """The grid associated with the pockets."""
        return self._grid

    @property
    def labels_pocket(self) -> JAXArray:
        """Labels array of the main pockets."""
        return self._label_pockets

    @property
    def labels_subpocket(self) -> JAXArray | None:
        """Labels array of the subpockets, if any."""
        return self._label_subpockets

    @property
    def external_data(self) -> pd.DataFrame | None:
        """External data associated with the pockets, if any."""
        return self._external_data

    def point_coverage(self, points: ArrayLike):
        cols = {}
        for label, row in self._pockets.iterrows():
            pocket = row["pocket"]
            coverage = pocket.point_coverage(points)
            cols[label] = coverage
        return pd.DataFrame(cols)

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        show_box: bool = False,
        name_prefix: str = "P ",
        box_name_prefix: str = "BBox ",
        contour: bool = False,
        wireframe: bool = True,
        visible: bool = True,
        color: tuple[float, float, float] | Sequence[tuple[float, float, float]] | None = None,
        receptor: Any | Literal[False] | None = None,
    ):
        """Display the pockets in an NGLWidget."""
        nv = nglwidget or scishow.nglview.NGLWidget()
        if receptor is not False:
            if receptor is not None:
                nv.add_trajectory(receptor)
            elif self._receptor is not None:
                nv.add_trajectory(self._receptor)
        if color is None:
            color = np.random.rand(len(self._pockets), 3).tolist()
        elif isinstance(color, tuple):
            color = [color] * len(self._pockets)
        for idx, (_, pocket) in enumerate(self._pockets.iterrows()):
            pocket.pocket.display(
                nglwidget=nv,
                show_box=show_box,
                name=f"{name_prefix}{pocket.label}",
                box_name=f"{box_name_prefix}{pocket.label}",
                contour=contour,
                wireframe=wireframe,
                visible=visible,
                color=color[idx],
                receptor=False,
            )
        return nv
