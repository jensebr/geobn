"""Grid specification and pure-numpy/pyproj reprojection engine.

No rasterio objects are used here.  The only spatial library is pyproj,
which is used solely for coordinate transformation between CRS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Transformer

from ._types import RasterData


@dataclass
class GridSpec:
    """Describes the spatial grid that all inputs are aligned to."""

    crs: str                  # EPSG string or WKT
    transform: Affine         # pixel-corner-to-world affine (GDAL convention)
    shape: tuple[int, int]    # (height, width)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_raster_data(cls, data: RasterData) -> GridSpec:
        if data.crs is None or data.transform is None:
            raise ValueError(
                "Cannot derive a GridSpec from a source with no CRS/transform "
                "(e.g. ConstantSource).  Call bn.set_grid() explicitly."
            )
        return cls(crs=data.crs, transform=data.transform, shape=data.array.shape[:2])

    @classmethod
    def from_params(
        cls,
        crs: str,
        resolution: float,
        extent: tuple[float, float, float, float],
    ) -> GridSpec:
        """Build a GridSpec from explicit parameters.

        Parameters
        ----------
        crs:
            Target CRS as EPSG string (e.g. "EPSG:32632") or WKT.
        resolution:
            Pixel size in the units of *crs*.
        extent:
            (xmin, ymin, xmax, ymax) in the units of *crs*.
        """
        xmin, ymin, xmax, ymax = extent
        width = max(1, round((xmax - xmin) / resolution))
        height = max(1, round((ymax - ymin) / resolution))
        transform = Affine(resolution, 0, xmin, 0, -resolution, ymax)
        return cls(crs=crs, transform=transform, shape=(height, width))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extent_wgs84(self) -> tuple[float, float, float, float]:
        """Return the bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max)."""
        H, W = self.shape
        t = self.transform
        # Four corners in the source CRS
        xs = [t.c, t.c + W * t.a + H * t.b]
        ys = [t.f + H * t.e + W * t.d, t.f]
        corner_x = [t.c, t.c + W * t.a, t.c + H * t.b, t.c + W * t.a + H * t.b]
        corner_y = [t.f, t.f + W * t.d, t.f + H * t.e, t.f + W * t.d + H * t.e]
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(corner_x, corner_y)
        return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_to_grid(data: RasterData, grid: GridSpec) -> np.ndarray:
    """Reproject and resample *data* to match *grid*.

    Returns a (H, W) float32 array.  Pixels that fall outside *data*'s extent
    are filled with NaN.

    A source with ``crs=None`` (ConstantSource) is broadcast directly.
    A source that already matches the target grid is returned as-is.
    """
    if data.crs is None:
        # ConstantSource — broadcast scalar to reference shape
        return np.full(grid.shape, float(data.array.flat[0]), dtype=np.float32)

    src_arr = data.array.astype(np.float32)

    if (
        data.crs == grid.crs
        and data.transform == grid.transform
        and data.array.shape == grid.shape
    ):
        return src_arr

    return _reproject(src_arr, data.crs, data.transform, grid.crs, grid.transform, grid.shape)


# ---------------------------------------------------------------------------
# Core reprojection (pure numpy + pyproj)
# ---------------------------------------------------------------------------


def _reproject(
    src: np.ndarray,
    src_crs: str,
    src_transform: Affine,
    dst_crs: str,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    H, W = dst_shape

    # Pixel-centre coordinates of every destination pixel
    col_idx = np.arange(W, dtype=np.float64)
    row_idx = np.arange(H, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    # Pixel centre = corner + 0.5
    cc = col_grid + 0.5
    rc = row_grid + 0.5

    # Destination pixel centres in destination CRS (world coords)
    dst_x = dst_transform.a * cc + dst_transform.b * rc + dst_transform.c
    dst_y = dst_transform.d * cc + dst_transform.e * rc + dst_transform.f

    # Transform from destination CRS to source CRS
    if dst_crs != src_crs:
        tr = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        src_x, src_y = tr.transform(dst_x.ravel(), dst_y.ravel())
        src_x = src_x.reshape(H, W)
        src_y = src_y.reshape(H, W)
    else:
        src_x, src_y = dst_x, dst_y

    # Invert the source affine to get fractional pixel-grid coordinates
    inv = ~src_transform
    src_col = inv.a * src_x + inv.b * src_y + inv.c
    src_row = inv.d * src_x + inv.e * src_y + inv.f

    return _bilinear_resample(src, src_row, src_col)


def _bilinear_resample(
    src: np.ndarray,
    row_pix: np.ndarray,
    col_pix: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation at fractional pixel-grid coordinates.

    *row_pix* and *col_pix* are in pixel-grid space where 0.5 is the centre
    of the first pixel (GDAL / affine-package convention).
    """
    src_H, src_W = src.shape

    # Shift so pixel-centre of pixel 0 maps to 0.0
    ra = row_pix - 0.5
    ca = col_pix - 0.5

    r0 = np.floor(ra).astype(np.int32)
    c0 = np.floor(ca).astype(np.int32)
    r1 = r0 + 1
    c1 = c0 + 1

    dr = (ra - r0).astype(np.float32)
    dc = (ca - c0).astype(np.float32)

    # Pixels outside source extent → NaN
    oob = (ra < -0.5) | (ra >= src_H - 0.5) | (ca < -0.5) | (ca >= src_W - 0.5)

    # Clamp indices for safe array access (oob pixels will be overwritten)
    r0c = np.clip(r0, 0, src_H - 1)
    r1c = np.clip(r1, 0, src_H - 1)
    c0c = np.clip(c0, 0, src_W - 1)
    c1c = np.clip(c1, 0, src_W - 1)

    v00 = src[r0c, c0c]
    v01 = src[r0c, c1c]
    v10 = src[r1c, c0c]
    v11 = src[r1c, c1c]

    result = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    result[oob] = np.nan
    return result
