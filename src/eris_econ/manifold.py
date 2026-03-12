# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Economic Decision Complex (E): the weighted graph on which agents pathfind.

Vertices are economic states (points in R^9).
Edges are available actions/transactions.
Edge weights combine Mahalanobis distance (attribute-vector change)
with boundary penalties (moral rules that impose discontinuous costs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from eris_econ.dimensions import N_DIMS, EconomicState
from eris_econ.metrics import edge_weight


@dataclass
class Edge:
    """A directed edge (action/transaction) in the decision complex."""

    source: str  # vertex id
    target: str  # vertex id
    label: str = ""  # human-readable action name
    weight: float = 0.0  # computed total cost


@dataclass
class Vertex:
    """A vertex (economic state) in the decision complex."""

    id: str
    state: EconomicState
    label: str = ""


class EconomicDecisionComplex:
    """Weighted directed graph representing an agent's decision space.

    The decision complex E = (V, E, w) where:
    - V: set of economic states (vertices)
    - E: set of available actions (directed edges)
    - w: edge weight function (Mahalanobis + boundary penalties)

    Usage:
        E = EconomicDecisionComplex(sigma=np.eye(9))
        E.add_vertex("start", EconomicState((100, 1, 0.5, ...)))
        E.add_vertex("trade", EconomicState((80, 1, 0.7, ...)))
        E.add_edge("start", "trade", label="sell widget")
        E.compute_weights()
    """

    def __init__(
        self,
        sigma: np.ndarray,
        boundaries: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            sigma: [9, 9] covariance matrix for Mahalanobis distance.
                   Encodes how dimensions interact and their relative importance.
            boundaries: dict mapping boundary name → penalty β_k.
                        Use np.inf for sacred-value boundaries.
        """
        if sigma.shape != (N_DIMS, N_DIMS):
            raise ValueError(f"sigma must be ({N_DIMS}, {N_DIMS}), got {sigma.shape}")
        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))
        self.boundaries = boundaries or {}

        self.vertices: Dict[str, Vertex] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[str, List[Edge]] = {}

    def add_vertex(self, vid: str, state: EconomicState, label: str = "") -> None:
        """Add a vertex (economic state) to the complex."""
        self.vertices[vid] = Vertex(id=vid, state=state, label=label or vid)
        if vid not in self._adjacency:
            self._adjacency[vid] = []

    def add_edge(self, source: str, target: str, label: str = "") -> Edge:
        """Add a directed edge (available action) between vertices."""
        if source not in self.vertices or target not in self.vertices:
            raise KeyError(f"Both vertices must exist: {source}, {target}")
        e = Edge(source=source, target=target, label=label)
        self.edges.append(e)
        self._adjacency[source].append(e)
        return e

    def add_bidirectional(self, v1: str, v2: str, label: str = "") -> Tuple[Edge, Edge]:
        """Add edges in both directions."""
        e1 = self.add_edge(v1, v2, label=f"{label} (→)")
        e2 = self.add_edge(v2, v1, label=f"{label} (←)")
        return e1, e2

    def compute_weights(self) -> None:
        """Compute edge weights for all edges using Mahalanobis + boundaries."""
        for e in self.edges:
            s = self.vertices[e.source].state
            t = self.vertices[e.target].state
            e.weight = edge_weight(
                np.array(s.values),
                np.array(t.values),
                self.sigma_inv,
                self.boundaries,
            )

    def neighbors(self, vid: str) -> List[Edge]:
        """Return outgoing edges from a vertex."""
        return self._adjacency.get(vid, [])

    def get_state(self, vid: str) -> EconomicState:
        """Get the economic state of a vertex."""
        return self.vertices[vid].state

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)
