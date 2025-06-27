

class Field:
    def __init__(
        self,
        tensor,
        ids,
    ):
        self._tensor = tensor
        self._ids = ids
        return

    @property
    def tensor(self):
        return self._tensor

    @property
    def ids(self):
        return self._ids

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.tensor.shape[1:-3]
