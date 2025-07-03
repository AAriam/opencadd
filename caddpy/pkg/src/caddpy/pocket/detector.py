"""Grid-based binding pocket detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

import arrayer
from scids.field import Field

from caddpy.chemsys import ChemicalSystem
from caddpy.pocket.ligsite import LigSite
from caddpy.pocket.default import Default
from caddpy.pocket.pockets import Pockets
from caddpy import exception

if TYPE_CHECKING:
    from typing import Literal


class Detector:
    def __init__(self, receptor: ChemicalSystem, field: Field):
        self._receptor = receptor
        self._field = field

        self._grid_axis_indices = tuple(range(self.field.batch_ndim, self.field.tensor.ndim))
        self._original_volume_receptor = self.field.tensor.astype(bool)
        self._original_volume_solvent = jnp.logical_not(self._original_volume_receptor)
        self._ligsite = LigSite(field=self.field, directions=(1, 2, 3))
        self._receptor_volume = self._original_volume_receptor
        self._mask_morphology = self._original_volume_solvent
        self._mask_ligsite: np.ndarray | None = None
        self._mask_custom: np.ndarray | None = None
        self._gui = None
        return

    def extract_pockets(
        self,
        opening: bool = Default.EXTRACT_OPEN,
        opening_structure: float | np.ndarray = Default.EXTRACT_OPEN_STRUCT_RADIUS,
        opening_iterations: int = Default.EXTRACT_OPEN_ITER,
        opening_mask: np.ndarray | None = None,
        opening_brute_force: bool = False,
    ):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html
        mask_opened = sp.ndimage.binary_opening(
            input=self.mask,
            structure=self._create_structuring_element(opening_structure),
            iterations=opening_iterations,
            mask=opening_mask,
            brute_force=opening_brute_force,
            border_value=0,
            axes=self._grid_axis_indices,
        ) if opening else self.mask

        # if open:
        #     structure = self._create_structuring_element(opening_structure)
        #     eroded = sp.ndimage.binary_erosion(
        #         input=self.mask,
        #         structure=structure,
        #         iterations=opening_iterations,
        #         mask=opening_mask,
        #         border_value=opening_border_value,
        #         axes=self._grid_axis_indices,
        #     )
        #     mask_opened = sp.ndimage.binary_propagation(
        #         input=eroded,
        #         structure=structure,
        #         mask=self.mask,
        #         border_value=opening_border_value,
        #         axes=self._grid_axis_indices,
        #     )


        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        label_tensor, num_features = sp.ndimage.label(mask_opened)
        dtype = arrayer.dtype.smallest_integer(minimum=0, maximum=num_features)
        return Pockets(
            grid=self.field.grid,
            pocket_labels=jnp.asarray(label_tensor, dtype=dtype),
        )

    def set_mask_morphology(
        self,
        close: bool = Default.MORPH_CLOSE,
        fill: bool = Default.MORPH_FILL,
        closing_structure: float | np.ndarray = Default.MORPH_CLOSE_STRUCT_RADIUS,
        closing_iterations: int = Default.MORPH_CLOSE_ITER,
        closing_mask: np.ndarray | None = None,
        closing_brute_force: bool = False,
    ):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_closing.html
        volume_closed = sp.ndimage.binary_closing(
            input=self.field.tensor,
            structure=self._create_structuring_element(closing_structure),
            iterations=closing_iterations,
            mask=closing_mask,
            brute_force=closing_brute_force,
            border_value=0,
            axes=self._grid_axis_indices,
        ) if close else self.field.tensor

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
        volume_closed_and_filled = sp.ndimage.binary_fill_holes(
            input=volume_closed,
            axes=self._grid_axis_indices,
        ) if fill else volume_closed
        self._receptor_volume = volume_closed_and_filled.astype(bool)
        self._mask_morphology = jnp.logical_not(volume_closed_and_filled)
        return self._mask_morphology

    def set_mask_ligsite(
        self,
        count_lower: int | None = Default.LIGSITE_COUNT_LOWER,
        count_upper: int | None = Default.LIGSITE_COUNT_UPPER,
        dist_lower: float | None = Default.LIGSITE_DIST_LOWER,
        dist_upper: float | None = Default.LIGSITE_DIST_UPPER,
        dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = Default.LIGSITE_DIST_LOWER_MODE,
        dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = Default.LIGSITE_DIST_UPPER_MODE,
    ):
        self._mask_ligsite = self._ligsite.psp_mask(
            count_lower=count_lower,
            count_upper=count_upper,
            dist_lower=dist_lower,
            dist_upper=dist_upper,
            dist_lower_mode=dist_lower_mode,
            dist_upper_mode=dist_upper_mode,
        )
        return self._mask_ligsite

    def set_mask_custom(self, mask: np.ndarray):
        mask = jnp.asarray(mask)
        if mask.shape != self.field.tensor.shape:
            raise exception.InputError(
                name="mask",
                message=f"Mask shape {mask.shape} does not match field shape {self.field.tensor.shape}."
            )
        self._mask_custom = mask
        return self._mask_custom

    def unset_mask(self, *args: Literal["morphology", "ligsite", "custom"]) -> None:
        args = set(args or ("morphology", "ligsite", "custom"))
        if "morphology" in args:
            self.set_mask_morphology(close=False, fill=False)
        if "ligsite" in args:
            self._mask_ligsite = None
        if "custom" in args:
            self._mask_custom = None
        return

    @property
    def mask(self) -> jax.Array:
        masks = [self._mask_morphology]
        if self._mask_ligsite is not None:
            masks.append(self._mask_ligsite)
        if self._mask_custom is not None:
            masks.append(self._mask_custom)
        return jnp.logical_and.reduce(jnp.array(masks))

    @property
    def mask_morphology(self) -> jax.Array:
        return self._mask_morphology

    @property
    def mask_ligsite(self) -> jax.Array | None:
        return self._mask_ligsite

    @property
    def mask_custom(self) -> jax.Array | None:
        return self._mask_custom

    @property
    def receptor_volume(self) -> jax.Array:
        """The receptor volume tensor after morphological transformation.

        This is the inverse of `mask_morphology`.
        """
        return self._receptor_volume

    @property
    def receptor_volume_added(self) -> jax.Array:
        """The volume added to the receptor after morphological transformation."""
        return jnp.logical_and(self.receptor_volume, self._original_volume_solvent)

    @property
    def receptor_volume_removed(self) -> jax.Array:
        """The volume removed from the receptor after morphological transformation."""
        return jnp.logical_and(self.mask_morphology, self._original_volume_receptor)

    @property
    def ligsite(self) -> LigSite | None:
        return self._ligsite

    @property
    def field(self) -> Field:
        return self._field

    @property
    def receptor(self) -> ChemicalSystem:
        return self._receptor

    def _create_structuring_element(self, structure: float | np.ndarray) -> np.ndarray:
        """Return a structuring element.

        Parameters
        ----------
        structure
            Either a user-defined structuring element,
            or the radius of a sphere in the same units as the Grid's `spacing`.

        Returns
        -------
        If `structure` is an array, it is returned as-is.
        If `structure` is a number, a centrosymmetric 3D boolean array representing a filled sphere.
        """
        if not isinstance(structure, int | float):
            return structure
        return self.field.grid.footprint_spherical(radius=structure)
