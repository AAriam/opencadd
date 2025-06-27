

class Pocket:
    def __init__(
        self,
        origin,
        spacing,
        voxels,
    ):
        self._origin = origin
        self._spacing = spacing
        self._voxels = voxels
        self._voxel_volume = self.spacing ** 3
        return

    @property
    def origin(self):
        return self._origin

    @property
    def spacing(self):
        return self._spacing

    @property
    def voxels(self):
        return self._voxels

    @property
    def voxel_volume(self):
        """Calculate the volume of a single voxel."""
        return self._voxel_volume
