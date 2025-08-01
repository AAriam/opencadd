from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

if TYPE_CHECKING:
    from sciapi.typing import FileContentLike, FileLike, PathLike


def filelike_to_filepath(file: FileLike) -> Path:
    """
    Turn a file-like object to IO.

    Parameters
    ----------
    file

    Returns
    -------

    """
    # if isinstance(file, (io.BufferedIOBase, io.BytesIO)):
    #     return file
    # if isinstance(file, (io.TextIOBase, io.StringIO)):
    #     return file.buffer
    #
    # if isinstance(file, str):
    #     filepath = Path(file)
    #     try:
    #         if filepath.is_file():
    #             with open(filepath, "rb") as f:
    #                 return f
    return


def filelike_to_data_string(file):
    if isinstance(file, Path):
        with open(file) as f:
            return f.read()
    if isinstance(file, str):
        possible_path = Path(file)
        if possible_path.is_file():
            with open(possible_path) as f:
                return f.read()
        return file
    if isinstance(file, bytes):
        return file.decode()
    return None


def write_to_file(
    content: FileContentLike,
    filename: str,
    extension: str | None = None,
    path: PathLike | None = None
) -> Path:
    if path is None:
        dir_path = Path.cwd()
    else:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
    fullpath = (dir_path / filename).resolve()
    if extension is not None:
        ext = f".{extension.removeprefix(".")}" if extension else ""
        fullpath = fullpath.with_suffix(ext)
    mode = "xb" if isinstance(content, bytes) else "xt"
    with open(fullpath, mode) as f:
        f.write(content)
    return fullpath


def recursive_type_cast(
    obj: dict | list | tuple | str
) -> dict | list | tuple | str | int | float | bool | None:
    """Recursively cast all string values to their appropriate types.

    Parameters
    ----------
    obj
        The object to be recursively casted.
    """
    if isinstance(obj, dict):
        return {k: recursive_type_cast(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(recursive_type_cast(v) for v in obj)
    if not isinstance(obj, str):
        return obj
    value = obj.strip()
    if not value:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
