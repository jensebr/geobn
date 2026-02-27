"""rasterio-backed I/O helpers.

All rasterio imports are confined to this module.  Functions here are only
called when the user explicitly writes output; the rest of the library
never touches rasterio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine


def write_geotiff(
    array: np.ndarray,
    crs: str,
    transform: Affine,
    path: str | Path,
    nodata: float = float("nan"),
) -> None:
    """Write a multi-band float32 GeoTIFF.

    Parameters
    ----------
    array:
        (bands, H, W) float32 array.
    crs:
        CRS as EPSG string or WKT.
    transform:
        Affine pixel-to-world transform.
    path:
        Output file path.
    nodata:
        NoData value written into the file metadata.
    """
    try:
        import rasterio  # noqa: PLC0415
        from rasterio.crs import CRS  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to write GeoTIFFs. "
            "Install it with: pip install geobn[io]"
        ) from exc

    path = Path(path)
    bands, H, W = array.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=bands,
        dtype=np.float32,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array.astype(np.float32))
