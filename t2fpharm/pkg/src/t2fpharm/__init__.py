from t2fpharm import field, grid, ligand, pocket, receptor, modeler

__all__ = [
    "field",
    "grid",
    "ligand",
    "pocket",
    "receptor",
]


def load(
    pocket: pocket.Pocket,
    field: field.Field,
):
    return modeler.Modeler(
        pocket=pocket,
        field=field,
    )
