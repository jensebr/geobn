#!/usr/bin/env python3
"""Convert a GeNIe XDSL file to BIF format for use with pgmpy / geobn.

Usage
-----
    python tools/xdsl_to_bif.py <input.xdsl> [output.bif]

Notes
-----
- <cpt> nodes are converted directly (probability orderings are identical
  in XDSL and BIF: last parent varies fastest, child varies fastest).
- <noisymax> nodes are expanded to full CPTs using the NoisyMax CDF formula:
    P(Y ≤ k | parents) = CDF_leak(k) × ∏_i CDF_i(k | parent_i_state)
  Parameters in XDSL are grouped per (parent, parent_state_ordered_by_strength_ascending).
- <utility> and <mau> nodes are skipped with a warning — they are part of
  GeNIe's influence-diagram extension and have no equivalent in a plain BN.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_xdsl(path: Path) -> tuple[dict, dict]:
    """Parse XDSL and return (nodes, skipped).

    nodes   : dict[node_id → info_dict]
    skipped : dict[node_id → tag_name]
    """
    tree = ET.parse(path)
    root = tree.getroot()
    nodes_elem = root.find("nodes")

    nodes: dict = {}
    skipped: dict = {}

    for child in nodes_elem:
        tag = child.tag
        node_id = child.get("id")

        if tag in ("utility", "mau"):
            skipped[node_id] = tag
            continue

        if tag not in ("cpt", "noisymax"):
            skipped[node_id] = tag
            continue

        states = [s.get("id") for s in child.findall("state")]

        parents_elem = child.find("parents")
        parents = parents_elem.text.strip().split() if parents_elem is not None else []

        info: dict = {"type": tag, "states": states, "parents": parents}

        if tag == "cpt":
            probs_text = child.find("probabilities").text.strip()
            info["probabilities"] = list(map(float, probs_text.split()))

        else:  # noisymax
            strengths_text = child.find("strengths").text.strip()
            params_text = child.find("parameters").text.strip()
            info["strengths"] = list(map(int, strengths_text.split()))
            info["parameters"] = list(map(float, params_text.split()))

        nodes[node_id] = info

    return nodes, skipped


# ---------------------------------------------------------------------------
# NoisyMax expansion
# ---------------------------------------------------------------------------


def _cdf(probs: list[float], k: int) -> float:
    """CDF up to and including index k."""
    return sum(probs[: k + 1])


def expand_noisymax(node_id: str, info: dict, all_nodes: dict) -> list[float]:
    """Expand a NoisyMax node to a full CPT probability table.

    Returns a flat list in BIF ordering:
        for each parent-state combination (last parent fastest):
            P(child=state_0), P(child=state_1), ...

    NoisyMax CDF formula
    --------------------
    Parameters are grouped: for each parent, one sub-group per parent state
    sorted by strength ASCENDING (lowest strength = index 0).
    The last group is the leak distribution.

    P(Y ≤ k | x_1,...,x_n) = CDF_leak(k) × ∏_i CDF_i(k | rank_of(x_i))

    P(Y = k) = P(Y ≤ k) − P(Y ≤ k−1)
    """
    states = info["states"]
    parents = info["parents"]
    strengths = info["strengths"]
    parameters = info["parameters"]
    n_child = len(states)

    parent_state_counts = [len(all_nodes[p]["states"]) for p in parents]

    # ── Parse strengths ──────────────────────────────────────────────────────
    # strengths is a flat list: for each parent, the strength of each XML-ordered state.
    # state_to_rank[parent_i][xml_state_j] = rank in strength-ascending order
    strength_map: list[list[int]] = []
    offset = 0
    for n_pstates in parent_state_counts:
        strength_map.append(strengths[offset : offset + n_pstates])
        offset += n_pstates

    state_to_rank: list[list[int]] = []
    for strs in strength_map:
        # Build rank: sort by ascending strength value
        sorted_indices = sorted(range(len(strs)), key=lambda j: strs[j])
        rank = [0] * len(strs)
        for rank_pos, xml_idx in enumerate(sorted_indices):
            rank[xml_idx] = rank_pos
        state_to_rank.append(rank)

    # ── Parse parameters ─────────────────────────────────────────────────────
    # param_groups[parent_i][strength_rank] = [n_child probabilities]
    param_groups: list[list[list[float]]] = []
    param_offset = 0
    for n_pstates in parent_state_counts:
        group = []
        for _ in range(n_pstates):
            group.append(parameters[param_offset : param_offset + n_child])
            param_offset += n_child
        param_groups.append(group)

    leak = parameters[param_offset : param_offset + n_child]
    param_offset += n_child

    if param_offset != len(parameters):
        raise ValueError(
            f"NoisyMax '{node_id}': expected {param_offset} parameters, "
            f"got {len(parameters)}"
        )

    # ── Compute full CPT ─────────────────────────────────────────────────────
    result: list[float] = []
    for parent_states in product(*[range(n) for n in parent_state_counts]):
        # Combined CDF at each child-state index
        combined_cdf = []
        for k in range(n_child):
            cdf_k = _cdf(leak, k)
            for i, pstate in enumerate(parent_states):
                rank_i = state_to_rank[i][pstate]
                cdf_k *= _cdf(param_groups[i][rank_i], k)
            combined_cdf.append(cdf_k)

        # CDF → PMF
        pmf = [
            max(0.0, combined_cdf[k] - (combined_cdf[k - 1] if k > 0 else 0.0))
            for k in range(n_child)
        ]

        # Normalize for floating-point safety
        total = sum(pmf)
        if total > 1e-12:
            pmf = [p / total for p in pmf]
        else:
            pmf = [1.0 / n_child] * n_child

        result.extend(pmf)

    return result


# ---------------------------------------------------------------------------
# BIF generation
# ---------------------------------------------------------------------------


def to_bif(nodes: dict, network_name: str) -> str:
    lines: list[str] = []
    lines.append(f"network {network_name} {{")
    lines.append("}")
    lines.append("")

    # Variable declarations
    for node_id, info in nodes.items():
        states_str = ", ".join(info["states"])
        n = len(info["states"])
        lines.append(f"variable {node_id} {{")
        lines.append(f"  type discrete [ {n} ] {{ {states_str} }};")
        lines.append("}")
        lines.append("")

    # Probability tables
    for node_id, info in nodes.items():
        parents = info["parents"]
        n_child = len(info["states"])

        if info["type"] == "cpt":
            probs = info["probabilities"]
        else:
            try:
                probs = expand_noisymax(node_id, info, nodes)
            except Exception as exc:
                print(
                    f"  WARNING: Failed to expand noisymax '{node_id}': {exc}",
                    file=sys.stderr,
                )
                n_combos = 1
                for p in parents:
                    n_combos *= len(nodes[p]["states"])
                probs = [1.0 / n_child] * (n_combos * n_child)

        if not parents:
            probs_str = ", ".join(f"{p:.8f}" for p in probs)
            lines.append(f"probability ( {node_id} ) {{")
            lines.append(f"  table {probs_str};")
            lines.append("}")
        else:
            parents_str = ", ".join(parents)
            lines.append(f"probability ( {node_id} | {parents_str} ) {{")

            parent_states_list = [nodes[p]["states"] for p in parents]

            for parent_combo in product(*[range(len(ps)) for ps in parent_states_list]):
                # BIF ordering: last parent varies fastest
                combo_idx = 0
                stride = 1
                for i in reversed(range(len(parents))):
                    combo_idx += parent_combo[i] * stride
                    stride *= len(parent_states_list[i])

                chunk = probs[combo_idx * n_child : (combo_idx + 1) * n_child]
                combo_names = ", ".join(
                    parent_states_list[i][parent_combo[i]] for i in range(len(parents))
                )
                probs_str = ", ".join(f"{p:.8f}" for p in chunk)
                lines.append(f"  ({combo_names}) {probs_str};")

            lines.append("}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.xdsl> [output.bif]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".bif")
    )

    print(f"Reading {input_path} ...", file=sys.stderr)
    nodes, skipped = parse_xdsl(input_path)

    if skipped:
        by_type: dict[str, list[str]] = {}
        for nid, ntype in skipped.items():
            by_type.setdefault(ntype, []).append(nid)
        for ntype, nids in by_type.items():
            print(
                f"  Skipping {len(nids)} <{ntype}> node(s): {', '.join(nids)}",
                file=sys.stderr,
            )

    noisymax_nodes = [nid for nid, info in nodes.items() if info["type"] == "noisymax"]
    if noisymax_nodes:
        print(
            f"  Expanding {len(noisymax_nodes)} <noisymax> node(s) to full CPT: "
            f"{', '.join(noisymax_nodes)}",
            file=sys.stderr,
        )

    network_name = input_path.stem.replace("-", "_").replace(" ", "_")
    bif_content = to_bif(nodes, network_name)

    output_path.write_text(bif_content, encoding="utf-8")
    print(f"Written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()