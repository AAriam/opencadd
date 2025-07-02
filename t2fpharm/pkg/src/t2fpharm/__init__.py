from t2fpharm import field, grid, ligand, pocket, receptor
from t2fpharm.modeler import Modeler

__all__ = [
    "field",
    "grid",
    "ligand",
    "pocket",
    "receptor",
]


def load(
    field: field.Field,
    pocket: pocket.Pocket | None = None,
) -> Modeler:
    """Create a target-based pharmacophore modeler.

    Parameters
    ----------
    field
        Field containing energy (or similar) values
        for pharmacophore matching.
    pocket
        Additional pocket mask for the field.
    """
    return Modeler(pocket=pocket, field=field)
