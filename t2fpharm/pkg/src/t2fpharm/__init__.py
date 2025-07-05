"""T2FPharm: Truly Target Focused Pharmacophore Modeler."""

from t2fpharm import field, grid, ligand, pocket, receptor
from t2fpharm.modeler import Modeler

__all__ = [
    "field",
    "grid",
    "ligand",
    "pocket",
    "receptor",
]


def modeler(
    field: field.Field,
    pocket: pocket.Pocket | None = None,
    receptor: receptor.Receptor | None = None,
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
