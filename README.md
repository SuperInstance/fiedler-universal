# fiedler-universal

**Fiedler vector partitioning across every graph family — benchmark spectral clustering against k-means, spectral clustering, Louvain, and label propagation.**

Systematic benchmark of the Fiedler vector (eigenvector for λ₂ of the normalized Laplacian) as a universal partition method. Tests on Erdős-Rényi, stochastic block model, geometric, Watts-Strogatz, Barabási-Albert, planted partition, ring of cliques, and ladder graphs. Compares against 4 alternative methods using ARI and NMI.

## What This Gives You

- **Fiedler partition** — spectral bipartitioning via algebraic connectivity eigenvector
- **Multi-k extension** — use eigenvectors 1..k for k-way partitioning
- **8 graph families** — ER, SBM, geometric, WS, BA, planted partition, ring of cliques, ladder
- **5 methods compared** — Fiedler, k-means on Fiedler, full spectral clustering, Louvain, label propagation
- **Metrics** — Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- **Configurable** — node counts, cluster sizes, edge probabilities, trials

## Quick Start

```bash
pip install numpy scipy scikit-learn
python benchmark.py
```

Outputs results table to console and saves summary JSON.

## How It Fits

Part of the SuperInstance ecosystem:

- **[graph-neural](https://github.com/SuperInstance/graph-neural)** — Spectral GNN primitives (uses Fiedler vector)
- **fiedler-universal** — Systematic Fiedler benchmarking (this repo)

## License

MIT
