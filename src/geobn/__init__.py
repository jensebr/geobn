"""geobn — Bayesian network inference over geographic space."""
from .network import GeoBayesianNetwork, load
from .result import InferenceResult
from .sources import (
    ArraySource,
    BarentswatchAISSource,
    ConstantSource,
    CopernicusMarineSource,
    EMODnetBathymetrySource,
    HubOceanSource,
    KartverketDTMSource,
    METLocationForecastSource,
    METOceanForecastSource,
    OpenMeteoSource,
    RasterSource,
    URLSource,
    WCSSource,
)

__all__ = [
    "load",
    "GeoBayesianNetwork",
    "InferenceResult",
    "ArraySource",
    "BarentswatchAISSource",
    "ConstantSource",
    "CopernicusMarineSource",
    "EMODnetBathymetrySource",
    "HubOceanSource",
    "KartverketDTMSource",
    "METLocationForecastSource",
    "METOceanForecastSource",
    "OpenMeteoSource",
    "RasterSource",
    "URLSource",
    "WCSSource",
]
