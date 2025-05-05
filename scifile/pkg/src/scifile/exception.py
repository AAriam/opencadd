"""Exceptions raised by SciFile."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class SciFileError(Exception):
    """Base class for all exceptions raised by SciFile."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SciFileReadError(SciFileError):
    """Exception raised when a file cannot be read."""

    def __init__(
        self,
        filetype: str,
        message: str,
        filepath: Path | None = None,
        content: str | bytes | None = None,
        line_idx: int | None = None,
        col_idx: int | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(message)
        self.filetype = filetype
        self.filepath = filepath
        self.content = content
        self.line_idx = line_idx
        self.col_idx = col_idx
        self.token = token
        return


class SciFileValidationError(SciFileError):
    """Exception raised when a file is not valid."""

    def __init__(
        self,
        filetype: str,
        message: str,
        filepath: Path | None = None,
        content: str | bytes | None = None,
        line_idx: int | None = None,
        col_idx: int | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(message)
        self.filetype = filetype
        self.filepath = filepath
        self.content = content
        self.line_idx = line_idx
        self.col_idx = col_idx
        self.token = token
        return


def raise_or_warn(
    exception: Exception,
    *,
    strict: bool,
    critical: bool = False,
) -> None:
    """Raise an error or warn based on the strictness level and criticality."""
    if critical or strict:
        raise exception
    print(f"Warning: {exception.message}")
    return
