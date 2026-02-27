from __future__ import annotations

from pathlib import Path

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

# rasterio is imported lazily inside fetch() so the entire library works
# even when rasterio is not installed.


class RasterSource(DataSource):
    """Read a local GeoTIFF file.

    rasterio is used only to open the file and is discarded immediately;
    the returned RasterData contains only plain numpy/affine objects.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        try:
            import rasterio  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for RasterSource. "
                "Install it with: pip install geobn[io]"
            ) from exc

        with rasterio.open(self._path) as src:
            array = src.read(1).astype(np.float32)
            crs = src.crs.to_string()
            transform = src.transform  # affine.Affine — safe to keep

        return RasterData(array=array, crs=crs, transform=transform)
