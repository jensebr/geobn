"""Tests for the discretization module."""
from __future__ import annotations

import numpy as np
import pytest

from geobn.discretize import DiscretizationSpec, discretize_array


class TestDiscretizationSpec:
    def test_valid_spec(self):
        spec = DiscretizationSpec([0, 10, 30, 90], ["flat", "moderate", "steep"])
        assert len(spec.labels) == 3

    def test_wrong_label_count_raises(self):
        with pytest.raises(ValueError, match="labels"):
            DiscretizationSpec([0, 10, 30, 90], ["only_two", "labels_given"])

    def test_single_breakpoint_raises(self):
        with pytest.raises(ValueError, match="breakpoints"):
            DiscretizationSpec([0], ["flat"])


class TestDiscretizeArray:
    def setup_method(self):
        self.spec = DiscretizationSpec([0, 10, 30, 90], ["flat", "moderate", "steep"])

    def test_below_first_bin(self):
        arr = np.array([[5.0]])
        idx = discretize_array(arr, self.spec)
        assert idx[0, 0] == 0  # "flat"

    def test_at_lower_boundary(self):
        arr = np.array([[10.0]])
        idx = discretize_array(arr, self.spec)
        assert idx[0, 0] == 1  # "moderate"

    def test_above_last_bin(self):
        arr = np.array([[100.0]])
        idx = discretize_array(arr, self.spec)
        assert idx[0, 0] == 2  # "steep"

    def test_nan_maps_to_minus_one(self):
        arr = np.array([[np.nan]])
        idx = discretize_array(arr, self.spec)
        assert idx[0, 0] == -1

    def test_2d_array(self):
        arr = np.array([[5.0, 15.0, 50.0]])
        idx = discretize_array(arr, self.spec)
        np.testing.assert_array_equal(idx, [[0, 1, 2]])
