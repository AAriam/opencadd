"""Exceptions raised by SciDS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ScidsError(Exception):
    """Base class for all SciDS exceptions."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        return


class ScidsReadError(ScidsError):
    """Exception raised when a file cannot be read."""
    def __init__(
        self,
        file_type: str,
        message: str,
        filepath: Path | None = None,
        content: str | bytes | None = None,
        line_idx: int | None = None,
        col_idx: int | None = None,
        token: str | None = None,
    ):
        super().__init__(file_type, message)
        self.file_type = file_type
        self.filepath = filepath
        self.content = content
        self.line_idx = line_idx
        self.col_idx = col_idx
        self.token = token
        return


class InputError(ScidsError):
    """Exception raised when an input is invalid."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name
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
