"""Wildfire risk assessment — Calabria, Italy.

A real-data example for geobn using publicly available, no-registration data.

Data sources
------------
Copernicus DEM GLO-90
    90-metre digital elevation model from the Copernicus Land Monitoring Service.
    Hosted as Cloud-Optimised GeoTIFFs on AWS S3 (no authentication required).
    Four 1°×1° tiles are downloaded and cached on first run (~1.6 MB each).
    Tile URL pattern:
        https://copernicus-dem-90m.s3.amazonaws.com/
            Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM/
            Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM.tif

Open-Meteo Archive API
    Free historical weather data (https://open-meteo.com/), no API key required.
    Two variables are sampled on a 5×5 grid over the study area on a reference date.

Derived inputs
--------------
``slope``     — terrain slope in degrees, computed from the DEM with numpy
``elevation`` — raw DEM height in metres

Live inputs (Open-Meteo)
-------------------------
``temperature`` — daily mean air temperature (°C)
``wind_speed``  — daily maximum wind speed (km/h)

Bayesian network (wildfire_risk.bif)
-------------------------------------
    slope  ─┐
             ├─► terrain_flammability ─┐
    elevation─┘                        ├─► wildfire_risk
    temperature ─┐                     │
                 ├─► fire_weather ─────┘
    wind_speed  ─┘

Run from the repository root::

    uv run python examples/calabria_wildfire/run_example.py

DEM tiles are cached in examples/calabria_wildfire/data/ and reused on subsequent
runs.  Output GeoTIFFs are written to examples/calabria_wildfire/output/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests
import rasterio
from affine import Affine
from rasterio.merge import merge as rio_merge

import geobn

# ---------------------------------------------------------------------------
# Study area — central Calabria, southern Italy
# The Aspromonte Massif rises to ~1956 m, surrounded by steep coastal slopes
# covered in Mediterranean scrub (maquis).  Calabria has one of the highest
# wildfire frequencies in Europe.
# ---------------------------------------------------------------------------
WEST, SOUTH, EAST, NORTH = 15.5, 38.0, 16.6, 39.4
CRS = "EPSG:4326"

# Reference date: a hot July day typical of peak fire-risk season
REFERENCE_DATE = "2023-07-15"

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Copernicus DEM GLO-90 download helpers
# ---------------------------------------------------------------------------
_COPDEM_URL = (
    "https://copernicus-dem-90m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
)


def _tile_path(lat: int, lon: int) -> Path:
    return DATA_DIR / f"copdem90_N{lat:02d}_E{lon:03d}.tif"


def _download_tile(lat: int, lon: int) -> Path | None:
    """Download one DEM tile, return path or None if not available."""
    path = _tile_path(lat, lon)
    if path.exists():
        print(f"    cached  : {path.name}")
        return path

    url = _COPDEM_URL.format(lat=lat, lon=lon)
    print(f"    fetching: N{lat:02d}_E{lon:03d} ... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"SKIPPED ({exc})")
        return None

    path.write_bytes(resp.content)
    print(f"{len(resp.content) / 1e6:.1f} MB")
    return path


def load_dem() -> tuple[np.ndarray, Affine]:
    """Download (or load from cache) and mosaic the DEM tiles.

    Returns the elevation array (float32, NaN for sea/nodata) and its Affine
    transform, clipped to the study bounding box.
    """
    DATA_DIR.mkdir(exist_ok=True)

    # Which 1°×1° tiles cover the bounding box?
    import math
    lat_tiles = range(math.floor(SOUTH), math.ceil(NORTH))
    lon_tiles = range(math.floor(WEST), math.ceil(EAST))

    print(f"Copernicus DEM GLO-90 — {len(lat_tiles) * len(lon_tiles)} tiles:")
    tile_paths = [
        p
        for lat in lat_tiles
        for lon in lon_tiles
        if (p := _download_tile(lat, lon)) is not None
    ]

    if not tile_paths:
        sys.exit(
            "ERROR: No DEM tiles could be downloaded.  "
            "Check your internet connection and try again."
        )

    # Mosaic + crop to study area
    datasets = [rasterio.open(str(p)) for p in tile_paths]
    try:
        mosaic, transform = rio_merge(
            datasets,
            bounds=(WEST, SOUTH, EAST, NORTH),
            nodata=np.nan,
        )
    finally:
        for ds in datasets:
            ds.close()

    elevation = mosaic[0].astype(np.float32)
    # Belt-and-suspenders: mask any remaining fill values (ocean, voids)
    elevation[elevation < -500] = np.nan

    return elevation, transform


# ---------------------------------------------------------------------------
# Slope computation
# ---------------------------------------------------------------------------

def compute_slope(dem: np.ndarray, transform: Affine) -> np.ndarray:
    """Compute slope in degrees from a geographic (EPSG:4326) DEM.

    Converts the angular pixel spacing to approximate metric distances at the
    centre latitude so that ``np.gradient`` operates in consistent units.
    """
    lat_mid = (SOUTH + NORTH) / 2.0
    # Copernicus DEM GLO-90: 3 arc-second pixel spacing
    deg_per_pixel = abs(transform.a)                           # ≈ 0.000833°
    lat_m = deg_per_pixel * 111_320.0                          # ≈ 92.8 m N–S
    lon_m = deg_per_pixel * 111_320.0 * np.cos(np.radians(lat_mid))  # ≈ 72 m E–W

    # Temporarily fill NaN (ocean) with 0 so gradient doesn't propagate NaN
    dem_filled = np.where(np.isnan(dem), 0.0, dem)
    dz_dy, dz_dx = np.gradient(dem_filled, lat_m, lon_m)

    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)
    slope[np.isnan(dem)] = np.nan   # restore ocean mask
    return slope


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # ── 1. Load and process the DEM ───────────────────────────────────────
    elevation, transform = load_dem()
    slope = compute_slope(elevation, transform)

    H, W = elevation.shape
    valid = np.isfinite(elevation)
    print(f"\nGrid            : {H} × {W} pixels at 90 m resolution")
    print(f"Land pixels     : {valid.sum():,} / {H * W:,}  "
          f"({100 * valid.mean():.1f}% land)")
    print(f"Elevation range : {np.nanmin(elevation):.0f} – "
          f"{np.nanmax(elevation):.0f} m")
    print(f"Slope range     : {np.nanmin(slope):.1f} – "
          f"{np.nanmax(slope):.1f}°")

    # ── 2. Load BN ────────────────────────────────────────────────────────
    bif_path = Path(__file__).parent / "wildfire_risk.bif"
    bn = geobn.load(bif_path)

    bn.set_input("slope",       geobn.ArraySource(slope,     crs=CRS, transform=transform))
    bn.set_input("elevation",   geobn.ArraySource(elevation, crs=CRS, transform=transform))
    bn.set_input("temperature", geobn.OpenMeteoSource(
        "temperature_2m_mean", date=REFERENCE_DATE, sample_points=5))
    bn.set_input("wind_speed",  geobn.OpenMeteoSource(
        "wind_speed_10m_max", date=REFERENCE_DATE, sample_points=5))

    bn.set_discretization("slope",       [0, 15, 35, 90],    ["flat",    "moderate", "steep"])
    bn.set_discretization("elevation",   [0, 300, 900, 4000], ["lowland", "montane",  "alpine"])
    bn.set_discretization("temperature", [0, 20, 30, 55],    ["cool",    "warm",     "hot"])
    bn.set_discretization("wind_speed",  [0, 20, 40, 150],   ["calm",    "moderate", "strong"])

    # ── 3. Inference ─────────────────────────────────────────────────────
    print(f"\nFetching weather from Open-Meteo ({REFERENCE_DATE}) ...")
    result = bn.infer(query=["wildfire_risk"])

    probs = result.probabilities["wildfire_risk"]   # (H, W, 3): low / medium / high
    ent   = result.entropy("wildfire_risk")         # (H, W)

    p_high = probs[..., 2]
    print(f"\nP(wildfire_risk=high) — min / mean / max : "
          f"{np.nanmin(p_high):.3f} / {np.nanmean(p_high):.3f} / "
          f"{np.nanmax(p_high):.3f}")
    print(f"Entropy (bits)        — min / mean / max : "
          f"{np.nanmin(ent):.3f} / {np.nanmean(ent):.3f} / "
          f"{np.nanmax(ent):.3f}")

    # ── 4. xarray summary ─────────────────────────────────────────────────
    ds = result.to_xarray()
    print(f"\nxarray Dataset:\n{ds}")

    # ── 5. Interactive map ─────────────────────────────────────────────────
    try:
        html_path = result.show_map(
            OUT_DIR,
            extra_layers={
                "Slope (°)": slope,
                "Elevation (m)": elevation,
            },
        )
        print(f"\nInteractive map opened in browser → {html_path}")
        print("  Use the layer control (top-right) to switch overlays.")
    except ImportError as exc:
        print(f"\nSkipping interactive map ({exc})")
        print("  Install folium: pip install geobn[viz]")

    # ── 6. GeoTIFF export ─────────────────────────────────────────────────
    try:
        result.to_geotiff(OUT_DIR)
        out = OUT_DIR / "wildfire_risk.tif"
        print(f"\nGeoTIFF written to {out}")
        print("  Band 1: P(low)   Band 2: P(medium)   "
              "Band 3: P(high)   Band 4: entropy (bits)")
    except ImportError as exc:
        print(f"\nSkipping GeoTIFF export ({exc})")


if __name__ == "__main__":
    main()
