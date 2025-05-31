"""Exceptions raised by the package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class SciCoDaError(Exception):
    """Base class for all exceptions raised by this package."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        return


class DataFileNotFoundError(SciCoDaError):
    """Raised when a requested data file is not found.

    Parameters
    ----------
    path_relative
        Path to the file relative to the package's data directory.
    path_absolute
        Absolute path to the file.
    """

    def __init__(
        self,
        category: str,
        name: str,
        filepath: Path,
    ):
        self.category = category
        self.name = name
        self.filepath = filepath
        message = (
            f"Could not find the requested package data file "
            f"'{name}' in category '{category}' at filepath '{filepath}'."
        )
        super().__init__(message)
        return
