"""Exceptions raised by the CADDpy package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class CADDpyError(Exception):
    """Base class for all exceptions raised by the CADDpy package."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        return


class MissingDependencyError(CADDpyError):
    """Exception raised when a required dependency is missing."""

    def __init__(self, name: str) -> None:
        message = f"Missing required dependency: {name}"
        super().__init__(message)
        self.name = name
        return


class SubprocessError(CADDpyError):
    """Exception raised when a subprocess fails."""

    def __init__(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        code: int,
        stdout: str | bytes | None,
        stderr: str | bytes | None,

    ) -> None:
        message = f"{name} failed with exit code {code}."
        super().__init__(message)
        self.name = name
        self.command = command
        self.cwd = cwd
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        return


class InputError(CADDpyError):
    """Exception raised when an input is invalid."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name
        return