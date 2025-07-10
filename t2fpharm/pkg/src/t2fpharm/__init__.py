"""T2FPharm: Truly Target Focused Pharmacophore Modeler."""

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
    receptor: System | None = None,
) -> Modeler:
    """Create a target-focused pharmacophore modeler.

    Parameters
    ----------
    field
        Field containing energy values or other metrics
        corresponding to pharmacophore features.
    pocket
        Optional pocket mask for the field.
    receptor
        Receptor associated with the pharmacophore modeler.
    """
    return Modeler(field=field, pocket=pocket, receptor=receptor)
