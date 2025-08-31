from sciapi.exception.base import SciAPIError


class PDBeError(SciAPIError):
    """Base class for all PDBe exceptions."""
    pass


class PDBeSIFTSError(PDBeError):
    """Base class for all PDBe SIFTS exceptions."""
    pass


class PDBeSIFTSMappingExpansionError(PDBeSIFTSError):
    """Exception raised when there is an error in mapping between PDB and UniProt."""
    pass