"""Kartverket (Norwegian Mapping Authority) Digital Terrain Model source."""
from __future__ import annotations

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource
from .wcs_source import WCSSource

_WCS_BASE = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm10"

_LAYER_MAP = {
    "dtm10": "nhm_dtm_topo_25833",   # 10 m terrain model (projected EPSG:25833)
    "dtm1": "nhm_dtm_topo_25833",    # 1 m terrain model (same endpoint, different coverage)
    "dom10": "nhm_dom_topo_25833",   # 10 m surface model
}


class KartverketDTMSource(DataSource):
    """Fetch the Norwegian Digital Terrain Model from Kartverket's free WCS.

    Coverage is limited to Norway; pixels outside the valid bounds are
    returned as NaN (no exception is raised).

    Requires ``rasterio`` (``pip install geobn[io]``).

    Parameters
    ----------
    layer:
        Coverage resolution/type:

        * ``"dtm10"`` — 10 m terrain model (default).
        * ``"dtm1"`` — 1 m terrain model (very large; use small grids only).
        * ``"dom10"`` — 10 m surface model (includes vegetation/buildings).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(self, layer: str = "dtm10", timeout: int = 60) -> None:
        if layer not in _LAYER_MAP:
            raise ValueError(
                f"Unknown Kartverket layer {layer!r}. "
                f"Valid options: {list(_LAYER_MAP)}"
            )
        self._layer = layer
        self._timeout = timeout
        self._wcs = WCSSource(
            url=_WCS_BASE,
            layer=_LAYER_MAP[layer],
            version="2.0.1",
            format="image/tiff",
            timeout=timeout,
        )

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        data = self._wcs.fetch(grid=grid)
        array = data.array.copy()

        # Replace sentinel nodata values with NaN
        array[(array < -500) | (array > 9000)] = np.nan

        return RasterData(array=array, crs=data.crs, transform=data.transform)
