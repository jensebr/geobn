"""InferenceResult — the object returned by GeoBayesianNetwork.infer()."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine

from ._io import write_geotiff
from .inference import shannon_entropy


@dataclass
class InferenceResult:
    """Holds per-pixel probability distributions for one or more query nodes.

    Attributes
    ----------
    probabilities:
        Mapping from query node name to a (H, W, n_states) float32 array.
        NaN where any input was NoData.
    state_names:
        Mapping from query node name to its ordered list of state labels.
    crs:
        CRS of the output grid as an EPSG string or WKT.
    transform:
        Affine pixel-to-world transform of the output grid.
    """

    probabilities: dict[str, np.ndarray]   # node → (H, W, n_states)
    state_names: dict[str, list[str]]
    crs: str
    transform: Affine

    # ------------------------------------------------------------------
    # Derived property
    # ------------------------------------------------------------------

    def entropy(self, node: str) -> np.ndarray:
        """Shannon entropy (bits) for *node*, shape (H, W)."""
        return shannon_entropy(self.probabilities[node])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_geotiff(self, output_dir: str | Path) -> None:
        """Write one multi-band GeoTIFF per query node.

        Band layout (1-indexed):
          Bands 1…N  — P(state_i | evidence) for each state i
          Band N+1   — Shannon entropy

        Band descriptions contain the state label or "entropy".
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for node, probs in self.probabilities.items():
            H, W, n_states = probs.shape
            ent = self.entropy(node)[..., np.newaxis]  # (H, W, 1)
            cube = np.concatenate([probs, ent], axis=-1)  # (H, W, n_states+1)
            bands = cube.transpose(2, 0, 1)               # (bands, H, W)

            out_path = output_dir / f"{node}.tif"
            write_geotiff(bands, self.crs, self.transform, out_path)

    def to_xarray(self) -> xr.Dataset:
        """Return an xarray Dataset with spatial coordinates.

        Each query node becomes a DataArray with dimensions
        (state, y, x).  Entropy is added as a separate variable
        ``{node}_entropy`` with dimensions (y, x).
        """
        H, W = next(iter(self.probabilities.values())).shape[:2]
        transform = self.transform

        # Pixel-centre coordinates
        xs = transform.c + (np.arange(W) + 0.5) * transform.a
        ys = transform.f + (np.arange(H) + 0.5) * transform.e

        data_vars: dict[str, xr.DataArray] = {}

        for node, probs in self.probabilities.items():
            states = self.state_names[node]
            da = xr.DataArray(
                probs.transpose(2, 0, 1),  # (state, y, x)
                dims=["state", "y", "x"],
                coords={
                    "state": states,
                    "y": ys,
                    "x": xs,
                },
                name=node,
                attrs={"crs": self.crs},
            )
            data_vars[node] = da

            ent_da = xr.DataArray(
                self.entropy(node),
                dims=["y", "x"],
                coords={"y": ys, "x": xs},
                name=f"{node}_entropy",
                attrs={"crs": self.crs, "units": "bits"},
            )
            data_vars[f"{node}_entropy"] = ent_da

        return xr.Dataset(data_vars)
