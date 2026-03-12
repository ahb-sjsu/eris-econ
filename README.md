# eris-econ

[![CI](https://github.com/ahb-sjsu/eris-econ/actions/workflows/ci.yml/badge.svg)](https://github.com/ahb-sjsu/eris-econ/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/eris-econ)](https://pypi.org/project/eris-econ/)
[![Python](https://img.shields.io/pypi/pyversions/eris-econ)](https://pypi.org/project/eris-econ/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Geometric economics: multi-dimensional decision manifolds, A\* pathfinding, and Bond Geodesic Equilibrium.

## Overview

Classical economics reduces human decisions to scalar utility maximization. This library implements the **geometric economics** framework from [Bond (2026)](https://doi.org/10.2139/ssrn.XXXXX), which models economic decisions as **pathfinding on a 9-dimensional decision manifold**.

The core insight: *Homo economicus* is not wrong because humans are irrational — it's incomplete because it computes on a projected subspace of the actual decision manifold. Projecting from 9 dimensions to 1 (monetary utility) destroys information that is mathematically irrecoverable.

## The Nine Dimensions

Every economic state is a point in R^9:

| Dim | Name | Examples |
|-----|------|----------|
| d₁ | **Consequences** | monetary cost, material outcome, expected value |
| d₂ | **Rights** | property rights, contractual obligations |
| d₃ | **Fairness** | distributional justice, reciprocity |
| d₄ | **Autonomy** | freedom of choice, voluntariness |
| d₅ | **Privacy/Trust** | information asymmetry, fiduciary duty |
| d₆ | **Social Impact** | externalities, reputation |
| d₇ | **Virtue/Identity** | self-image, moral identity |
| d₈ | **Legitimacy** | institutional trust, rule compliance |
| d₉ | **Epistemic** | information quality, confidence |

Dimensions d₁–d₄ are *transferable* (conserved in bilateral exchange). Dimensions d₅–d₉ are *evaluative* (not conserved, enabling mutual gains from trade).

## Key Concepts

### Bond Geodesic
The optimal path on the decision manifold from current state to goal, minimizing Mahalanobis distance + boundary penalties. This is computed via A\* search, where:
- **g(n)** = accumulated cost (System 2: deliberate calculation)
- **h(n)** = heuristic estimate (System 1: moral intuition)

### Bond Geodesic Equilibrium (BGE)
Generalization of Nash equilibrium to multi-dimensional manifolds. Each agent minimizes behavioral friction on their own decision complex. Reduces to Nash when all non-monetary dimensions vanish.

### Emergent Behavioral Phenomena
These are not ad-hoc biases — they emerge geometrically:
- **Loss aversion** (λ ≈ 2.25): losses traverse more dimensions than gains
- **Reference dependence**: distance measured from current state on manifold
- **Endowment effect**: ownership activates rights + identity dimensions
- **Framing effects**: gauge transformations on the description basis

## Installation

```bash
pip install eris-econ
```

Or for development:

```bash
git clone https://github.com/ahb-sjsu/eris-econ.git
cd eris-econ
pip install -e ".[dev]"
```

## Quick Start

```python
from eris_econ.games import ultimatum_game
from eris_econ.pathfinding import astar

# Build the proposer's decision complex
E = ultimatum_game(stake=10.0)

# Find the optimal offer (Bond Geodesic)
goals = {f"offer_{p}" for p in [0, 10, 20, 30, 40, 50]}
result = astar(E, "start", goals)

print(f"Optimal choice: {result.path[-1]}")  # ~offer_40 (not offer_0!)
print(f"Total behavioral friction: {result.total_cost:.3f}")
```

```python
from eris_econ.behavioral import compute_loss_aversion
import numpy as np

# Compute emergent loss aversion from metric structure
sigma = np.eye(9)
sigma[0, 6] = 0.3  # consequences-identity coupling
sigma[6, 0] = 0.3
lambda_ratio = compute_loss_aversion(sigma)
print(f"Loss aversion λ = {lambda_ratio:.2f}")  # > 1.0
```

## Modules

| Module | Description |
|--------|-------------|
| `dimensions.py` | The 9 economic dimensions, transferable vs evaluative classification |
| `manifold.py` | Economic Decision Complex — weighted directed graph on R^9 |
| `metrics.py` | Mahalanobis distance, boundary penalties, edge weights |
| `pathfinding.py` | A\* search for Bond Geodesic computation |
| `equilibrium.py` | Bond Geodesic Equilibrium via iterated best response |
| `games.py` | Standard games: ultimatum, dictator, prisoner's dilemma, public goods |
| `behavioral.py` | Loss aversion, reference dependence, endowment effect, framing |
| `calibration.py` | Parameter estimation for Σ and β from behavioral data |
| `welfare.py` | Multi-dimensional Pareto optimality and social welfare |

## Citation

If you use this library in academic work, please cite:

```bibtex
@article{bond2026geometric,
  title={Geometric Economics: Multi-Dimensional Decision Manifolds and Bond Geodesic Equilibrium},
  author={Bond, Andrew H.},
  journal={Journal of Economic Theory},
  year={2026}
}
```

## License

MIT
