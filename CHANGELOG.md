# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2025

### Added
- `GeoBayesianNetwork` with `load()` factory, `set_input()`, `set_discretization()`, `infer()`.
- 11 built-in data sources: `ArraySource`, `ConstantSource`, `RasterSource`, `URLSource`,
  `OpenMeteoSource`, `WCSSource`, `KartverketDTMSource`, `EMODnetBathymetrySource`,
  `EMODnetShippingDensitySource`, `METOceanForecastSource`, `METLocationForecastSource`,
  `CopernicusMarineSource`, `BarentswatchAISSource`, `HubOceanSource`.
- Grid alignment via pure numpy + pyproj bilinear interpolation (no rasterio dependency on the hot path).
- Batched pixel-wise inference using unique-combination grouping.
- Real-time optimisation: `freeze()` (Tier 1) and `precompute()` (Tier 2) for repeated inference.
- `InferenceResult` with `.probabilities`, `.entropy()`, `.to_geotiff()`, `.to_xarray()`, `.show_map()`.
- Disk caching for static raster sources (`cache_dir` parameter).
- MkDocs + Material documentation site deployed to GitHub Pages.
- Two worked examples: Lyngen Alps avalanche risk, Norwegian maritime navigation risk.
