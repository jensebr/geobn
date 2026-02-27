"""Synthetic fire-risk example for geobn.

This script uses only in-memory arrays (no external files or API calls)
so it runs fully offline.  It demonstrates the complete pipeline:

  ArraySource → discretize → infer → to_geotiff / to_xarray

Run from the repository root:

    uv run python examples/synthetic_fire_risk/run_example.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine

import geobn

# ---------------------------------------------------------------------------
# 1.  Synthetic input data
# ---------------------------------------------------------------------------
# 20×30 pixel grid, 500 m resolution, UTM zone 32N
TRANSFORM = Affine(500, 0, 400_000, 0, -500, 5_100_000)
CRS = "EPSG:32632"
H, W = 20, 30

rng = np.random.default_rng(0)

# Slope: gradient from flat in the west to steep in the east (degrees)
slope = np.tile(np.linspace(2, 42, W), (H, 1)).astype(np.float32)
slope += rng.normal(0, 3, size=(H, W)).astype(np.float32)
slope = np.clip(slope, 0, 90)

# Rainfall: random, spatially varying (mm/day)
rainfall = rng.uniform(10, 180, (H, W)).astype(np.float32)

# ---------------------------------------------------------------------------
# 2.  Load BN and wire up sources
# ---------------------------------------------------------------------------
bif_path = Path(__file__).parent / "fire_risk.bif"
bn = geobn.load(bif_path)

bn.set_input("slope",    geobn.ArraySource(slope,    crs=CRS, transform=TRANSFORM))
bn.set_input("rainfall", geobn.ArraySource(rainfall, crs=CRS, transform=TRANSFORM))

bn.set_discretization("slope",    [0, 10, 30, 90],  ["flat", "moderate", "steep"])
bn.set_discretization("rainfall", [0, 25, 75, 200], ["low",  "medium",   "high"])

# ---------------------------------------------------------------------------
# 3.  Run inference
# ---------------------------------------------------------------------------
result = bn.infer(query=["fire_risk"])

probs = result.probabilities["fire_risk"]   # (H, W, 3)
ent   = result.entropy("fire_risk")         # (H, W)

print(f"Output grid : {H}×{W} pixels at {CRS}")
print(f"P(fire_risk=high) — min/mean/max : "
      f"{probs[...,2].min():.3f} / {probs[...,2].mean():.3f} / {probs[...,2].max():.3f}")
print(f"Entropy      — min/mean/max : "
      f"{ent.min():.3f} / {ent.mean():.3f} / {ent.max():.3f} bits")

# ---------------------------------------------------------------------------
# 4.  Export
# ---------------------------------------------------------------------------
out_dir = Path(__file__).parent / "output"

# xarray (works without rasterio)
ds = result.to_xarray()
print(f"\nxarray Dataset:\n{ds}")

# GeoTIFF (requires rasterio)
try:
    result.to_geotiff(out_dir)
    print(f"\nGeoTIFF written to {out_dir / 'fire_risk.tif'}")
    print("  Band 1: P(low)    Band 2: P(medium)    Band 3: P(high)    Band 4: entropy")
except ImportError as exc:
    print(f"\nSkipping GeoTIFF export ({exc})")
