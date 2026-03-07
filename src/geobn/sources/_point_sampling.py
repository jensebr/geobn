"""Abstract base class for point-sampling data sources.

Subclasses implement only ``_query_point(lat, lon) -> float``; the shared
fetch pipeline (grid validation, meshgrid construction, API loop, raster
assembly) lives here.
"""
from __future__ import annotations

import logging
import time
from abc import abstractmethod

import numpy as np
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_log = logging.getLogger(__name__)


class _PointSamplingSource(DataSource):
    """Mixin base for sources that sample a regular lat/lon grid of points.

    Subclasses must implement ``_query_point`` and may override ``__init__``
    with source-specific parameters, calling ``super().__init__`` to
    propagate *sample_points* and *timeout*.
    """

    requires_grid = True

    def __init__(self, sample_points: int = 5, timeout: int = 10) -> None:
        self._sample_points = max(1, sample_points)
        self._timeout = timeout

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                f"{type(self).__name__} requires a grid context to determine "
                "the spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        n = self._sample_points

        _log.info(
            "%s: sampling %d×%d grid (%d API calls)",
            type(self).__name__, n, n, n * n,
        )

        lats = np.linspace(lat_max, lat_min, n)  # north → south (row order)
        lons = np.linspace(lon_min, lon_max, n)
        lon_grid, lat_grid = np.meshgrid(lons, lats)  # (n, n) each

        values = np.full((n, n), np.nan, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                values[i, j] = self._query_point(
                    float(lat_grid[i, j]), float(lon_grid[i, j])
                )
                if n > 1:
                    time.sleep(0.05)  # be polite to free APIs

        valid_values = values[~np.isnan(values)]
        if valid_values.size > 0:
            _log.info(
                "%s: done — values range %.2f–%.2f",
                type(self).__name__, valid_values.min(), valid_values.max(),
            )
        else:
            _log.info("%s: done — all values are NaN", type(self).__name__)

        if n == 1:
            # Single-point result — broadcast like ConstantSource (1×1, no CRS)
            return RasterData(array=values, crs=None, transform=None)

        pixel_h = (lat_max - lat_min) / (n - 1)
        pixel_w = (lon_max - lon_min) / (n - 1)
        # Top-left corner shifted half a pixel outside the sample points so
        # that each sample sits at the centre of its cell.
        transform = Affine(pixel_w, 0, lon_min - pixel_w / 2, 0, -pixel_h, lat_max + pixel_h / 2)
        return RasterData(array=values, crs="EPSG:4326", transform=transform)

    @abstractmethod
    def _query_point(self, lat: float, lon: float) -> float:
        """Query the data source at a single WGS84 point.

        Returns the scalar value at (*lat*, *lon*), or ``float("nan")`` if the
        point is outside coverage or the API returns no data.
        """
