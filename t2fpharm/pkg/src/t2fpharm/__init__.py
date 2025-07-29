"""T2FPharm: Truly Target Focused Pharmacophore Modeler."""

from typing import Any

from t2fpharm import field, grid, pharm, pocket, system
from t2fpharm.field import Field
from t2fpharm.grid import Grid
from t2fpharm.modeler import Modeler
from t2fpharm.pharm import Pharmacophore
from t2fpharm.pocket import Pocket
from t2fpharm.system import System


__all__ = [
    "field",
    "grid",
    "pharm",
    "pocket",
    "system",
    "Field",
    "Grid",
    "Modeler",
    "Pharmacophore",
    "Pocket",
    "System",
]


def modeler(
    field: Field,
    pocket: Pocket | None = None,
    system: Any | None = None,
) -> Modeler:
    """Create a target-focused pharmacophore modeler.

    Parameters
    ----------
    field
        Field containing energy values or other metrics
        corresponding to pharmacophore features.
    pocket
        Optional pocket mask for the field.
    system
        Optional chemical system associated with the pharmacophore.
        This is not used by the modeler itself.
        If provided, it is only used by the `display()` method
        of the generated Pharmacophore to visualize the pharmacophore
        in the context of the chemical structure.
        This can be any object that can be visualized by NGLView
        using its `add_trajectory()` method.
    """
    return Modeler(field=field, pocket=pocket, system=system)
