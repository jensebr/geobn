from __future__ import annotations

from typing import NamedTuple

import numpy as np
from affine import Affine


class RasterData(NamedTuple):
    """Internal raster representation — never exposes rasterio objects."""

    array: np.ndarray       # shape (H, W), float32
    crs: str | None         # EPSG string or WKT; None for ConstantSource
    transform: Affine | None  # pixel-to-world affine; None for ConstantSource
