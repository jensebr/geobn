"""Tests for data source classes."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

import geobn
from geobn._types import RasterData


def test_array_source_roundtrip(slope_array, reference_transform):
    source = geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform)
    data = source.fetch()
    assert isinstance(data, RasterData)
    np.testing.assert_array_equal(data.array, slope_array)
    assert data.crs == "EPSG:4326"
    assert data.transform == reference_transform


def test_array_source_converts_to_float32(reference_transform):
    arr = np.ones((5, 5), dtype=np.float64)
    source = geobn.ArraySource(arr, crs="EPSG:4326", transform=reference_transform)
    assert source.fetch().array.dtype == np.float32


def test_array_source_rejects_3d(reference_transform):
    arr = np.ones((5, 5, 3))
    with pytest.raises(ValueError, match="2-D"):
        geobn.ArraySource(arr, crs="EPSG:4326", transform=reference_transform)


def test_constant_source_returns_scalar():
    source = geobn.ConstantSource(0.6)
    data = source.fetch()
    assert data.crs is None
    assert data.transform is None
    assert data.array.shape == (1, 1)
    assert float(data.array[0, 0]) == pytest.approx(0.6)


def test_raster_source_requires_rasterio(tmp_path):
    """RasterSource.fetch() should raise ImportError if rasterio is missing."""
    import importlib
    import sys

    # Only run this test if rasterio is NOT installed
    if importlib.util.find_spec("rasterio") is not None:
        pytest.skip("rasterio is installed; skipping missing-rasterio test")

    source = geobn.RasterSource(tmp_path / "nonexistent.tif")
    with pytest.raises(ImportError, match="rasterio"):
        source.fetch()


def test_openmeteo_source_requires_grid():
    source = geobn.OpenMeteoSource(variable="precipitation_sum", date="2024-01-01")
    with pytest.raises(ValueError, match="grid context"):
        source.fetch(grid=None)
