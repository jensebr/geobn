"""Batched pixel-wise Bayesian network inference.

Strategy
--------
Rather than running pgmpy once per pixel (potentially millions of times),
we find all *unique combinations* of discretised input states and run
inference exactly once per combination.  For typical BNs with a small
number of discrete states per node the number of unique combinations is
tiny even for very large rasters.

Data flow
---------
evidence_indices  dict[node, (H, W) int16]   state index per pixel
                  (-1 = NoData)
nodata_mask       (H, W) bool                True where any input is NaN

Returns
-------
dict[node, (H, W, n_states) float32]         probability per pixel per state
"""
from __future__ import annotations

import numpy as np
from pgmpy.inference import VariableElimination


def run_inference(
    model,
    evidence_indices: dict[str, np.ndarray],
    evidence_state_names: dict[str, list[str]],
    query_nodes: list[str],
    query_state_names: dict[str, list[str]],
    nodata_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run batched pixel-wise inference.

    Parameters
    ----------
    model:
        A fitted pgmpy BayesianNetwork.
    evidence_indices:
        Mapping from evidence node name to (H, W) int16 array of state indices.
    evidence_state_names:
        Mapping from evidence node name to its ordered list of state labels.
    query_nodes:
        Nodes whose posterior distributions are requested.
    query_state_names:
        Mapping from query node name to its ordered list of state labels.
    nodata_mask:
        (H, W) boolean array; True where any input pixel is NoData.

    Returns
    -------
    Mapping from query node name to a (H, W, n_states) float32 array.
    """
    H, W = next(iter(evidence_indices.values())).shape
    node_list = list(evidence_indices.keys())

    valid = ~nodata_mask  # (H, W)
    n_valid = int(valid.sum())

    # Pre-allocate output arrays filled with NaN
    output: dict[str, np.ndarray] = {}
    for node in query_nodes:
        n_states = len(query_state_names[node])
        output[node] = np.full((H, W, n_states), np.nan, dtype=np.float32)

    if n_valid == 0:
        return output

    # Stack evidence indices for valid pixels → (n_valid, n_nodes)
    valid_stack = np.column_stack(
        [evidence_indices[n][valid].astype(np.int32) for n in node_list]
    )

    # Find unique evidence combinations
    unique_combos, inverse = np.unique(valid_stack, axis=0, return_inverse=True)
    # unique_combos: (n_unique, n_nodes)
    # inverse:       (n_valid,)  maps each valid pixel → unique combo index

    ve = VariableElimination(model)

    # Results per unique combo: dict[node] → list of probability arrays
    combo_probs: dict[str, list[np.ndarray]] = {node: [] for node in query_nodes}

    for combo in unique_combos:
        evidence = {
            node_list[i]: evidence_state_names[node_list[i]][combo[i]]
            for i in range(len(node_list))
        }
        for node in query_nodes:
            factor = ve.query([node], evidence=evidence, show_progress=False)
            combo_probs[node].append(factor.values.astype(np.float32))

    # Map results back to spatial arrays
    for node in query_nodes:
        probs_per_combo = np.stack(combo_probs[node], axis=0)  # (n_unique, n_states)
        flat_probs = probs_per_combo[inverse]                   # (n_valid, n_states)
        output[node][valid] = flat_probs

    return output


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute per-pixel Shannon entropy (bits) from a probability array.

    Parameters
    ----------
    probs:
        (..., n_states) array of probabilities.

    Returns
    -------
    (...) array of entropy values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_p = np.where(probs > 0, np.log2(probs), 0.0)
    return -np.sum(probs * log2_p, axis=-1)
