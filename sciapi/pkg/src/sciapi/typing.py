from collections.abc import Sequence
from pathlib import Path
from typing import IO, TypeAlias


PathLike: TypeAlias = str | Path
FileContentLike: TypeAlias = str | bytes | IO
FileLike: TypeAlias = PathLike | FileContentLike
