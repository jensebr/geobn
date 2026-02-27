"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def fire_risk_model() -> DiscreteBayesianNetwork:
    """Three-node fire-risk BN: slope + rainfall → fire_risk."""
    model = DiscreteBayesianNetwork([("slope", "fire_risk"), ("rainfall", "fire_risk")])

    cpd_slope = TabularCPD(
        "slope",
        3,
        [[0.5], [0.3], [0.2]],
        state_names={"slope": ["flat", "moderate", "steep"]},
    )
    cpd_rainfall = TabularCPD(
        "rainfall",
        3,
        [[0.3], [0.4], [0.3]],
        state_names={"rainfall": ["low", "medium", "high"]},
    )

    # CPT columns: slope × rainfall  (slope outer, rainfall inner)
    # (flat,low) (flat,med) (flat,hi) (mod,lo) (mod,med) (mod,hi) (steep,lo) (steep,med) (steep,hi)
    cpd_fire = TabularCPD(
        "fire_risk",
        3,
        [
            [0.70, 0.60, 0.40, 0.40, 0.30, 0.20, 0.20, 0.10, 0.05],  # low
            [0.20, 0.30, 0.40, 0.40, 0.40, 0.30, 0.40, 0.30, 0.25],  # medium
            [0.10, 0.10, 0.20, 0.20, 0.30, 0.50, 0.40, 0.60, 0.70],  # high
        ],
        evidence=["slope", "rainfall"],
        evidence_card=[3, 3],
        state_names={
            "fire_risk": ["low", "medium", "high"],
            "slope": ["flat", "moderate", "steep"],
            "rainfall": ["low", "medium", "high"],
        },
    )

    model.add_cpds(cpd_slope, cpd_rainfall, cpd_fire)
    assert model.check_model(), "Fixture BN failed check_model()"
    return model


@pytest.fixture
def reference_transform() -> Affine:
    """North-up affine: 0.1° pixels, origin at (0°E, 50°N)."""
    return Affine(0.1, 0, 0.0, 0, -0.1, 50.0)


@pytest.fixture
def slope_array() -> np.ndarray:
    """10×10 synthetic slope array (degrees, 0–45)."""
    return np.linspace(0.0, 45.0, 100).reshape(10, 10).astype(np.float32)


@pytest.fixture
def rainfall_array() -> np.ndarray:
    """10×10 synthetic rainfall array (mm/day, 0–200)."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 200, (10, 10)).astype(np.float32)
