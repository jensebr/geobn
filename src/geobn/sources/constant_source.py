from __future__ import annotations

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


class ConstantSource(DataSource):
    """Broadcast a single scalar value across the entire domain.

    The 1×1 sentinel array is recognised by *align_to_grid* and expanded
    to the reference grid shape, so no explicit spatial metadata is needed.
    """

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        return RasterData(
            array=np.array([[self._value]], dtype=np.float32),
            crs=None,
            transform=None,
        )
