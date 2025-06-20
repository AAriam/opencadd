import numpy as np
import pandas as pd

import sciapi
import caddpy
import scids
import scishow


class T2FPharm:
    def __init__(
        self,
        pdb_id: str,
        ligand_id: str,
        grid_length: float = 16,
        grid_spacing: float = 0.6,
        psp_count_min: int = 4,
        max_energy_vacancy: float = 0.6,
        max_energy_hd: float = -0.35,
        max_energy_oa: float = -0.6,
        max_energy_c: float = -0.4,
        max_energy_pi: float = -1,
        max_energy_ni: float = -1,
        cnn_max_distance: float = 1.21,
        cnn_min_neighbors: tuple[int, ...] = (6, 12, 16, 20),
        cnn_min_members: int = 15,
        cnn_max_members: int = 80,
    ):
        self.pdb_file_raw: str = sciapi.pdb.file.entry(pdb_id, "pdb").decode("utf-8")
        self.pdb_file_fixed: str = sciapi.proteinsplus().protoss(pdb_id=pdb_id).protein

        self.complex = caddpy.chemsys.from_pdb(self.pdb_file_fixed)
        atoms = self.complex.composition.atoms
        self.ligand_atom_selection = (atoms["res_name"] == ligand_id).to_numpy()
        self.ligand_atom_coordinates = self.complex.trajectory.points[self.ligand_atom_selection]
        self.pocket_center = self.ligand_atom_coordinates.mean(axis=0)
        self.grid = scids.grid.from_size_spacing_anchor(
            size=(grid_length, grid_length, grid_length),
            spacings=grid_spacing,
            anchor="center",
            anchor_coord=self.pocket_center,
        )

        self.receptor = self.complex.select(atoms["res_poly"])

        energy_field = caddpy.mif.autogrid.from_chemsys(
            system=self.receptor,
            grid=self.grid,
            ligand_types=("HD", "OA", "C"),
            include_dsolvmap=False,
        )
        energies = np.empty((5, *self.grid.shape), dtype=np.float32)
        energies[0:4] = energy_field.tensor
        energies[4] = -energy_field.tensor[3]

        self.probes = {
            "hd": {"cutoff": max_energy_hd, "color": (0, 0.6, 0)},
            "oa": {"cutoff": max_energy_oa, "color": (0.6, 0, 0)},
            "c": {"cutoff": max_energy_c, "color": (1.0, 1.0, 0)},
            "pi": {"cutoff": max_energy_pi, "color": (0, 0, 1.0)},
            "ni": {"cutoff": max_energy_ni, "color": (1.0, 0, 0)},
        }
        self.energy_field = scids.field.from_tensor(
            tensor=energies,
            grid=self.grid,
            batch=(("feature", list(self.probes.keys())),),
        )
        self.energy_mask = np.less_equal(
            self.energy_field.tensor,
            np.array([probe["cutoff"] for probe in self.probes.values()], dtype=np.float32).reshape((-1, 1, 1, 1)),
        )

        self.pocket_vacancy = self.energy_field(feature="oa") <= max_energy_vacancy
        self.ligsite = caddpy.pocket.ligsite.LigSite(
            field=scids.field.from_tensor(
                tensor=self.pocket_vacancy,
                grid=self.grid,
                batch=0
            ),
            directions=(1, 3)
        )
        self.pocket_buriedness = self.ligsite.psp_count >= psp_count_min
        self.pocket_mask = np.logical_and(self.pocket_vacancy, self.pocket_buriedness)

        self.final_mask = np.logical_and(
            self.energy_mask,
            self.pocket_mask
        )
        self.feature_points = {}
        self.feature_labels = {}
        self.features = []
        voxel_volume = np.prod(self.grid.spacings)
        for probe_idx, probe_id in enumerate(self.probes.keys()):
            feature_mask = self.final_mask[probe_idx]
            feature_pointcloud = scids.pointcloud.from_array(self.grid.coordinates[feature_mask])
            self.feature_points[probe_id] = feature_pointcloud
            labels = feature_pointcloud.cluster_cnn(
                max_distance=cnn_max_distance,
                min_neighbors=cnn_min_neighbors,
                min_members=cnn_min_members,
                max_members=cnn_max_members,
            )
            self.feature_labels[probe_id] = labels
            cluster_sizes = np.bincount(labels)[1:]  # Exclude background label (0)
            cluster_count = cluster_sizes.size
            for cluster_label in range(1, cluster_count + 1):
                point_coordinates = feature_pointcloud.points[labels == cluster_label]
                n_points = cluster_sizes[cluster_label - 1]
                volume = n_points * voxel_volume
                self.features.append(
                    {
                        "type": probe_id,
                        "n_points": n_points,
                        "volume": volume,
                        "radius": (volume / ((4/3) * np.pi)) ** (1/3),
                        "center": point_coordinates.mean(axis=0),
                        "label": cluster_label,
                        "points": point_coordinates,
                    }
                )
        self.pharmacophore = pd.DataFrame(self.features)
        self.pharmacophore.set_index(["type", "label"], inplace=True, drop=False)
        self.plip = caddpy.interaction.from_pdb(self.pdb_file_fixed, ligands=[(ligand_id,)])
        self.match = self._calculate_match()
        return

    def display(
        self,
        nglwidget=None,
        plip_vis=("ligand", "water")
    ):
        nv = nglwidget or scishow.nglview.NGLWidget()
        nv.add_trajectory(self.complex)
        nv.add_box(self.grid.lower_bounds, self.grid.upper_bounds, name="Grid Box")
        nv.add_volume(
            data=self.pocket_vacancy.astype(int),
            basis=self.grid.unit_vectors,
            origin=self.grid.lower_bounds,
            name="Vacancy",
            representation_params=scishow.nglview.SurfaceRepresentationParameters(
                isolevel=0.5,
                isolevel_type="value",
                contour=True,
                color=(0.5, 0.5, 0.5),
                visible=False,
            )
        )
        nv.add_volume(
            data=self.pocket_buriedness.astype(int),
            basis=self.grid.unit_vectors,
            origin=self.grid.lower_bounds,
            name="Buriedness",
            representation_params=scishow.nglview.SurfaceRepresentationParameters(
                isolevel=0.5,
                isolevel_type="value",
                contour=True,
                color=(0.2, 0.8, 0.2),
                visible=False,
            )
        )
        nv.add_volume(
            data=self.pocket_mask.astype(int),
            basis=self.grid.unit_vectors,
            origin=self.grid.lower_bounds,
            name="Pocket",
            representation_params=scishow.nglview.SurfaceRepresentationParameters(
                isolevel=0.5,
                isolevel_type="value",
                contour=True,
                color=(0.8, 0.2, 0.2),
                visible=False,
            )
        )
        for probe_id, probe_data in self.probes.items():
            nv.add_volume(
                data=self.energy_field(feature=probe_id),
                basis=self.grid.unit_vectors,
                origin=self.grid.lower_bounds,
                name=f"{probe_id.upper()} Field",
                representation_params=scishow.nglview.SurfaceRepresentationParameters(
                    isolevel=probe_data["cutoff"],
                    isolevel_type="value",
                    contour=True,
                    color=probe_data["color"],
                    visible=False,
                )
            )
            nv.add_spheres(
                coords=self.feature_points[probe_id].points,
                radii=self.grid.spacings[0] / 2,
                name=f"{probe_id.upper()} Points",
                representation_params=scishow.nglview.RepresentationParameters(
                    visible=False,
                )
            )
        for feature in self.features:
            nv.add_spheres(
                coords=feature["points"],
                radii=self.grid.spacings[0] / 2,
                name=f"{feature['type'].upper()}{feature['label']} Points",
                colors=self.probes[feature["type"]]["color"],
                representation_params=scishow.nglview.RepresentationParameters(
                    visible=False,
                )
            )
            nv.add_spheres(
                coords=[feature["center"]],
                radii=feature["radius"],
                name=f"{feature['type'].upper()}{feature['label']} Center",
                colors=self.probes[feature["type"]]["color"],
                representation_params=scishow.nglview.RepresentationParameters(
                    opacity=0.8,
                )
            )
        self.plip.display(nv, vis=plip_vis)
        return nv.display(gui=True)

    def _calculate_match(self):
        out = []
        pharm = self.pharmacophore
        for row_idx, row in self.plip.all.iterrows():
            position_col = "l_position"
            match row["interaction_type"]:
                case "hbond":
                    if row["r_is_d"]:
                        probe = "oa"
                    else:
                        probe = "hd"
                        position_col = "h_position"
                case "water_bridge":
                    position_col = "w_position"
                    probe = "oa" if row["r_is_d"] else "hd"
                case "salt_bridge":
                    probe = "ni" if row["r_is_cation"] else "pi"
                case "hydrophobic":
                    probe = "c"
                case _:
                    continue
            feats = pharm[pharm["type"] == probe]
            if not len(feats):
                out.append({"match": False, "feature_type":probe})
                continue
            interaction_position = row[position_col]
            feature_centers = np.stack(feats["center"])
            distances = np.linalg.norm(feature_centers - interaction_position, axis=1)
            min_dist = distances.min()
            out.append({"match": min_dist < 2, "feature_type":probe, "feature_label": feats["label"].iloc[distances.argmin()], "dist": min_dist})
        return pd.DataFrame(out).convert_dtypes()



