"""Tests for GridSpec and alignment logic."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from geobn._types import RasterData
from geobn.grid import GridSpec, align_to_grid, _bilinear_resample


class TestGridSpec:
    def test_from_raster_data(self, slope_array, reference_transform):
        data = RasterData(array=slope_array, crs="EPSG:4326", transform=reference_transform)
        grid = GridSpec.from_raster_data(data)
        assert grid.shape == (10, 10)
        assert grid.crs == "EPSG:4326"
        assert grid.transform == reference_transform

    def test_from_raster_data_no_crs_raises(self, slope_array):
        data = RasterData(array=slope_array, crs=None, transform=None)
        with pytest.raises(ValueError, match="no CRS"):
            GridSpec.from_raster_data(data)

    def test_from_params(self):
        grid = GridSpec.from_params("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        assert grid.shape == (10, 10)
        assert grid.transform.a == pytest.approx(0.1)
        assert grid.transform.e == pytest.approx(-0.1)

    def test_extent_wgs84_identity(self, slope_array, reference_transform):
        """A grid in EPSG:4326 should return its own extent."""
        data = RasterData(array=slope_array, crs="EPSG:4326", transform=reference_transform)
        grid = GridSpec.from_raster_data(data)
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        assert lon_min == pytest.approx(0.0, abs=0.01)
        assert lat_max == pytest.approx(50.0, abs=0.01)


class TestAlignToGrid:
    def test_constant_source_broadcast(self, slope_array, reference_transform):
        data = RasterData(array=np.array([[7.5]], dtype=np.float32), crs=None, transform=None)
        grid = GridSpec(crs="EPSG:4326", transform=reference_transform, shape=(10, 10))
        result = align_to_grid(data, grid)
        assert result.shape == (10, 10)
        assert np.all(result == pytest.approx(7.5))

    def test_identity_passthrough(self, slope_array, reference_transform):
        data = RasterData(array=slope_array, crs="EPSG:4326", transform=reference_transform)
        grid = GridSpec(crs="EPSG:4326", transform=reference_transform, shape=(10, 10))
        result = align_to_grid(data, grid)
        np.testing.assert_array_almost_equal(result, slope_array)

    def test_same_crs_resample(self, reference_transform):
        """Upsample a 5×5 array to 10×10 in the same CRS."""
        src = np.arange(25, dtype=np.float32).reshape(5, 5)
        src_transform = Affine(0.2, 0, 0.0, 0, -0.2, 50.0)  # 0.2° pixels
        data = RasterData(array=src, crs="EPSG:4326", transform=src_transform)

        dst_transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)  # 0.1° pixels
        grid = GridSpec(crs="EPSG:4326", transform=dst_transform, shape=(10, 10))

        result = align_to_grid(data, grid)
        assert result.shape == (10, 10)
        # Centre of resampled grid should match centre of source grid
        assert not np.all(np.isnan(result))


class TestBilinearResample:
    def test_exact_pixel_centres(self):
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # Pixel centres at 0.5, 1.5
        rows = np.array([[0.5, 0.5], [1.5, 1.5]])
        cols = np.array([[0.5, 1.5], [0.5, 1.5]])
        result = _bilinear_resample(src, rows, cols)
        np.testing.assert_array_almost_equal(result, src)

    def test_midpoint_interpolation(self):
        src = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
        # Centre between columns 0 and 1 (col=1.0 in pixel-grid space)
        rows = np.array([[0.5]])
        cols = np.array([[1.0]])
        result = _bilinear_resample(src, rows, cols)
        assert result[0, 0] == pytest.approx(1.0)

    def test_out_of_bounds_is_nan(self):
        src = np.ones((3, 3), dtype=np.float32)
        rows = np.array([[10.0]])
        cols = np.array([[10.0]])
        result = _bilinear_resample(src, rows, cols)
        assert np.isnan(result[0, 0])
