from __future__ import annotations

from typing import TYPE_CHECKING

import pkgdata
import pyserials

from scicoda import exception

if TYPE_CHECKING:
    from pathlib import Path


_data_dir = pkgdata.get_package_path_from_caller(top_level=True) / "data"
_cache: dict[str, dict] = {}


def get_data(category: str, name: str, cache: bool = True) -> dict | list:
    return get(category, name, cache=cache)["data"]


def get_schema(category: str, name: str, cache: bool = True) -> dict | list:
    return get(category, name, cache=cache)["schema"]


def get_filepath(category: str, name: str) -> Path:
    """Get the absolute path to a data file.

    Parameters
    ----------
    category
        Category of the data file.
        This corresponds to the module name where the data can be accessed.
    name
        Name of the data file.
        This corresponds to the function name that returns the data.
    """
    filepath = _data_dir / category / f"{name}.yaml"
    if not filepath.is_file():
        raise exception.DataFileNotFoundError(
            category=category,
            name=name,
            filepath=filepath,
        )
    return filepath


def get(category: str, name: str, cache: bool = True) -> dict | list:
    if cached := _cache.get(category, {}).get(name):
        return cached
    filepath = get_filepath(category, name)
    file = pyserials.read.yaml_from_file(filepath)
    pyserials.validate.jsonschema(
        data=file["data"],
        schema=file["schema"],
        fill_defaults=True,
    )
    if cache:
        _cache.setdefault(category, {})[name] = file
    return file
