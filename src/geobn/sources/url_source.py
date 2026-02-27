from __future__ import annotations

import numpy as np
import requests

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

# rasterio is imported lazily; URLSource therefore also requires geobn[io].


class URLSource(DataSource):
    """Fetch a GeoTIFF from an HTTP/HTTPS URL and return it as plain data.

    The file is streamed into an in-memory buffer so nothing is written to
    disk.  rasterio is the only component that sees the raw bytes; the
    returned RasterData contains only numpy / affine objects.
    """

    def __init__(self, url: str, timeout: int = 60) -> None:
        self._url = url
        self._timeout = timeout

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        try:
            import rasterio  # noqa: PLC0415
            from rasterio.io import MemoryFile  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for URLSource. "
                "Install it with: pip install geobn[io]"
            ) from exc

        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                array = src.read(1).astype(np.float32)
                crs = src.crs.to_string()
                transform = src.transform

        return RasterData(array=array, crs=crs, transform=transform)
