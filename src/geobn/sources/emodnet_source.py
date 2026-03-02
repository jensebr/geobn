"""EMODnet (European Marine Observation and Data Network) bathymetry source."""
from __future__ import annotations

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource
from .wcs_source import WCSSource

_WCS_BASE = "https://ows.emodnet-bathymetry.eu/wcs"


class EMODnetBathymetrySource(DataSource):
    """Fetch European seabed bathymetry from the EMODnet bathymetry WCS.

    Returns depth values as **negative floats** (e.g. −200 m means 200 m
    below sea level).  Pixels outside EMODnet's coverage (land areas) are
    filled with NaN.

    Requires ``rasterio`` (``pip install geobn[io]``).

    Parameters
    ----------
    layer:
        Coverage to fetch:

        * ``"emodnet:mean"`` — mean depth (default).
        * ``"emodnet:stdev"`` — depth uncertainty (standard deviation).
    timeout:
        HTTP request timeout in seconds.
    """

    _VALID_LAYERS = frozenset({"emodnet:mean", "emodnet:stdev"})

    def __init__(self, layer: str = "emodnet:mean", timeout: int = 60) -> None:
        if layer not in self._VALID_LAYERS:
            raise ValueError(
                f"Unknown EMODnet layer {layer!r}. "
                f"Valid options: {sorted(self._VALID_LAYERS)}"
            )
        self._layer = layer
        self._timeout = timeout
        self._wcs = WCSSource(
            url=_WCS_BASE,
            layer=layer,
            version="2.0.1",
            format="image/tiff",
            timeout=timeout,
        )

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        data = self._wcs.fetch(grid=grid)
        array = data.array.copy()

        # Replace sentinel nodata values with NaN
        # Depths > 9000 m or shallower than −15 000 m are clearly invalid
        array[(array > 9000) | (array < -15000)] = np.nan

        return RasterData(array=array, crs=data.crs, transform=data.transform)
