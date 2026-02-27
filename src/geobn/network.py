"""GeoBayesianNetwork — the primary user-facing class."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .discretize import DiscretizationSpec, discretize_array
from .grid import GridSpec, align_to_grid
from .inference import run_inference, shannon_entropy
from .result import InferenceResult
from .sources._base import DataSource


class GeoBayesianNetwork:
    """A Bayesian network wired to geographic data sources.

    Typical usage::

        bn = geobn.load("model.bif")
        bn.set_input("slope",    geobn.RasterSource("slope.tif"))
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope",    [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])
        result = bn.infer(query=["fire_risk"])
        result.to_geotiff("output/")
    """

    def __init__(self, model: Any) -> None:
        """
        Parameters
        ----------
        model:
            A fitted ``pgmpy.models.BayesianNetwork``.
        """
        self._model = model
        self._inputs: dict[str, DataSource] = {}
        self._discretizations: dict[str, DiscretizationSpec] = {}
        self._grid: GridSpec | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_input(self, node: str, source: DataSource) -> None:
        """Attach a data source to a BN evidence node.

        Parameters
        ----------
        node:
            Name of a root node (no parents) in the BN.
        source:
            Any :class:`~geobn.sources.DataSource` subclass.
        """
        self._validate_node_exists(node)
        self._validate_is_root(node)
        self._inputs[node] = source

    def set_discretization(
        self,
        node: str,
        breakpoints: list[float],
        labels: list[str],
    ) -> None:
        """Define how continuous values for *node* are mapped to BN states.

        Parameters
        ----------
        node:
            Node name (must match an input node already registered via
            :meth:`set_input`).
        breakpoints:
            Monotonically increasing list of ``len(labels) + 1`` boundary
            values.  The first and last are the documented range limits; the
            interior values define the bin edges.
        labels:
            State names that **exactly** match the state names in the BN.
        """
        self._validate_node_exists(node)
        spec = DiscretizationSpec(breakpoints=list(breakpoints), labels=list(labels))
        self._validate_labels_match_bn(node, spec.labels)
        self._discretizations[node] = spec

    def set_grid(
        self,
        crs: str,
        resolution: float,
        extent: tuple[float, float, float, float],
    ) -> None:
        """Override the reference grid instead of deriving it from the first input.

        Parameters
        ----------
        crs:
            Target CRS as EPSG string (e.g. "EPSG:32632") or WKT.
        resolution:
            Pixel size in CRS units.
        extent:
            (xmin, ymin, xmax, ymax) in CRS units.
        """
        self._grid = GridSpec.from_params(crs, resolution, extent)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, query: list[str]) -> InferenceResult:
        """Run pixel-wise Bayesian inference and return probability rasters.

        Parameters
        ----------
        query:
            List of BN node names whose posterior distributions are requested.
            These nodes do not need to be root nodes.

        Returns
        -------
        InferenceResult
            Contains per-pixel probability arrays for each query node plus
            Shannon entropy.  Write to disk with ``.to_geotiff()`` or convert
            to xarray with ``.to_xarray()``.
        """
        if not self._inputs:
            raise RuntimeError("No inputs registered.  Call set_input() first.")

        for node in query:
            self._validate_node_exists(node)

        # ── 1. Determine the reference grid ───────────────────────────
        if self._grid is not None:
            ref_grid = self._grid
            pre_fetched: dict[str, Any] = {}
        else:
            first_node = next(iter(self._inputs))
            first_data = self._inputs[first_node].fetch(grid=None)
            if first_data.crs is None:
                raise ValueError(
                    f"The first registered input '{first_node}' has no CRS "
                    f"(it is a ConstantSource or similar).  "
                    f"Call bn.set_grid(crs, resolution, extent) explicitly "
                    f"or register a georeferenced source first."
                )
            ref_grid = GridSpec.from_raster_data(first_data)
            pre_fetched = {first_node: first_data}

        # ── 2. Validate discretizations are present for all inputs ─────
        for node in self._inputs:
            if node not in self._discretizations:
                raise ValueError(
                    f"No discretization set for input node '{node}'.  "
                    f"Call bn.set_discretization('{node}', breakpoints, labels)."
                )

        # ── 3. Fetch and align all inputs ──────────────────────────────
        aligned: dict[str, np.ndarray] = {}
        for node, source in self._inputs.items():
            data = pre_fetched[node] if node in pre_fetched else source.fetch(grid=ref_grid)
            aligned[node] = align_to_grid(data, ref_grid)

        # ── 4. Build NoData mask ───────────────────────────────────────
        nodata_mask = np.zeros(ref_grid.shape, dtype=bool)
        for arr in aligned.values():
            nodata_mask |= np.isnan(arr)

        # ── 5. Discretize ──────────────────────────────────────────────
        evidence_indices: dict[str, np.ndarray] = {}
        evidence_state_names: dict[str, list[str]] = {}
        for node, arr in aligned.items():
            spec = self._discretizations[node]
            idx = discretize_array(arr, spec)
            # Treat discretization NoData (-1) as spatial NoData
            nodata_mask |= idx < 0
            evidence_indices[node] = idx
            evidence_state_names[node] = spec.labels

        # ── 6. Collect query node state names from the BN ──────────────
        query_state_names: dict[str, list[str]] = {}
        for node in query:
            cpd = self._model.get_cpds(node)
            query_state_names[node] = list(cpd.state_names[node])

        # ── 7. Run inference ───────────────────────────────────────────
        probabilities = run_inference(
            model=self._model,
            evidence_indices=evidence_indices,
            evidence_state_names=evidence_state_names,
            query_nodes=query,
            query_state_names=query_state_names,
            nodata_mask=nodata_mask,
        )

        return InferenceResult(
            probabilities=probabilities,
            state_names=query_state_names,
            crs=ref_grid.crs,
            transform=ref_grid.transform,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_node_exists(self, node: str) -> None:
        if node not in self._model.nodes():
            raise ValueError(
                f"Node '{node}' does not exist in the BN.  "
                f"Available nodes: {sorted(self._model.nodes())}"
            )

    def _validate_is_root(self, node: str) -> None:
        parents = list(self._model.predecessors(node))
        if parents:
            raise ValueError(
                f"Node '{node}' has parents {parents} and is not a root node.  "
                f"Only root nodes (no parents) can be used as inputs."
            )

    def _validate_labels_match_bn(self, node: str, labels: list[str]) -> None:
        cpd = self._model.get_cpds(node)
        bn_states = list(cpd.state_names[node])
        if sorted(labels) != sorted(bn_states):
            raise ValueError(
                f"Discretization labels {labels} for node '{node}' do not "
                f"match the BN state names {bn_states}.  "
                f"Labels must exactly match (order-independent)."
            )


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def load(path: str | Path) -> GeoBayesianNetwork:
    """Load a Bayesian network from a BIF file.

    Parameters
    ----------
    path:
        Path to a ``.bif`` file.

    Returns
    -------
    GeoBayesianNetwork
        Ready to accept inputs via :meth:`~GeoBayesianNetwork.set_input`.
    """
    from pgmpy.readwrite import BIFReader  # noqa: PLC0415

    reader = BIFReader(str(Path(path)))
    model = reader.get_model()  # returns DiscreteBayesianNetwork in pgmpy >=1.0
    return GeoBayesianNetwork(model)
