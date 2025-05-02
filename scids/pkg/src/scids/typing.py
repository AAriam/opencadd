from typing import IO, TypeAlias
from pathlib import Path

PathLike: TypeAlias = str | Path
FileContentLike: TypeAlias = str | bytes | IO
