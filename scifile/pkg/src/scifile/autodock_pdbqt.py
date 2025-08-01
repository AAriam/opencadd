"""Read and write [AutoDock](https://autodock.scripps.edu/) PDBQT files.

References
----------
- [AutoDock Version 4.2.6 User Guide](https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/AutoDock4.2.6_UserGuide.pdf),
  Appendix I: AutoDock File Formats, Page 27: PDBQT format for coordinate files
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import scifile

if TYPE_CHECKING:
    from collections.abc import Sequence, Literal
    from scifile.pdb import PDBFileRecords, PDBFileSections, PDBFile


def read(
    file: str | bytes | Path,
    parse_only: Sequence[PDBFileRecords | PDBFileSections | str] | None = None,
    strictness: Literal[0, 1, 2, 3] = 0,
) -> PDBFile:
    """Read a PDBQT file.

    Parameters
    ----------
    file
        PDBQT file content or path.
        If a string, it is treated as the content of the file.
        If bytes, it is decoded to UTF-8.
        If a Path, the file is read as text.
    parse_only
        List of records or sections to parse.
        If None, all records are parsed.
    strictness
        Level of strictness for raising exceptions and warnings
        when encountering mistakes in the file:
        - 0: Raise only fatal errors and don't show any warnings.
        - 1: Raise only fatal errors. All other errors are reported as warnings.
        - 2: Raise fatal errors and mistakes resulting in ambiguous data.
             Inconsequential mistakes are reported as warnings.
        - 3: Completely validate the PDB file and raise all errors.
    """
    return scifile.pdb.read(
        file=file,
        variant="pdbqt",
        parse_only=parse_only,
        strictness=strictness
    )
