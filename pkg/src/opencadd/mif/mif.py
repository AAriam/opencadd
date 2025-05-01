"""
Abstract base clases used in the package.
"""

from opencadd import chem, spacetime


class MolecularInteractionField:
    """
    Intramolecular interaction field.
    """

    def __init__(
            self,
            ensemble: chem.ensemble.ChemicalEnsemble,
            field: spacetime.field.ToxelField,
    ):
        self._ensemble = ensemble
        self._field = field
        return

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def field(self):
        return self._field
