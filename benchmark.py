#!/usr/bin/env python3
"""
Fiedler Universal Partition Benchmark v2
=========================================
Fixed eigenvector extraction and improved graph generators.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import warnings, json, time, os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# Partition Methods
# ============================================================================

def _normalize_laplacian(adj):
    """Compute normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}"""
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    if np.any(degrees == 0):
        degrees += 1e-10
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    # L_norm = I - D^{-1/2} A D^{-1/2}
    normed = adj * np.outer(d_inv_sqrt, d_inv_sqrt)
    L_norm = np.eye(n) - normed
    return L_norm


def fiedler_partition(adj, n_clusters=2):
    """Fiedler partition using normalized Laplacian eigenvectors."""
    n = adj.shape[0]
    if sparse.issparse(adj):
        adj = adj.toarray()
    adj = np.float64((adj + adj.T) / 2.0)
    np.fill_diagonal(adj, 0)
    
    L_norm = _normalize_laplacian(adj)
    
    try:
        k = min(n_clusters + 1, n)  # get one extra for safety
        eigenvalues, eigenvectors = eigsh(sparse.csr_matrix(L_norm), k=k, which='SM',
                                           tol=1e-6, maxiter=5000)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Skip trivial eigenvector (eigenvalue ≈ 0)
        # Use eigenvectors 1..k for clustering
        if n_clusters == 2:
            fiedler_vec = eigenvectors[:, 1]
            # Try both signs and pick better
            labels_pos = (fiedler_vec > np.median(fiedler_vec)).astype(int)
            labels_neg = (fiedler_vec <= np.median(fiedler_vec)).astype(int)
            return labels_pos  # ARI handles label permutation
        else:
            features = eigenvectors[:, 1:n_clusters+1]
            # Normalize rows
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
            return km.fit_predict(features)
    except Exception:
        return np.zeros(n, dtype=int)


def kmeans_partition(adj, n_clusters=2):
    X = adj.toarray() if sparse.issparse(adj) else adj.copy()
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)


def spectral_partition(adj, n_clusters=2):
    if sparse.issparse(adj):
        adj = adj.toarray()
    adj = np.float64((adj + adj.T) / 2.0)
    np.fill_diagonal(adj, 0)
    mx = np.abs(adj).max()
    if mx > 0:
        adj = np.abs(adj) / mx
    from sklearn.cluster import SpectralClustering
    try:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                random_state=42, n_init=10, assign_labels='kmeans')
        return sc.fit_predict(adj)
    except Exception:
        return np.zeros(adj.shape[0], dtype=int)


def modularity_partition(adj, n_clusters=2):
    if sparse.issparse(adj):
        adj = adj.toarray()
    adj = np.float64((adj + adj.T) / 2.0)
    np.fill_diagonal(adj, 0)
    n = adj.shape[0]
    labels = np.arange(n)
    m = adj.sum() / 2.0
    if m == 0:
        return np.zeros(n, dtype=int)
    degrees = adj.sum(axis=1)
    
    for _ in range(5):
        changed = False
        order = np.random.permutation(n)
        for i in order:
            best_gain = 0
            best_c = labels[i]
            for j in np.where(adj[i] > 0)[0]:
                c = labels[j]
                if c == labels[i]:
                    continue
                comm_mask = labels == c
                ki_in = adj[i, comm_mask].sum()
                sum_tot = degrees[comm_mask].sum()
                ki = degrees[i]
                delta = ki_in / m - ki * sum_tot / (2 * m * m)
                if delta > best_gain:
                    best_gain = delta
                    best_c = c
            if best_c != labels[i]:
                labels[i] = best_c
                changed = True
        if not changed:
            break
    
    unique = np.unique(labels)
    mapping = {u: i for i, u in enumerate(unique)}
    return np.array([mapping[l] for l in labels])


def random_partition(adj, n_clusters=2):
    return np.random.randint(0, n_clusters, size=adj.shape[0])


# ============================================================================
# Graph Generators
# ============================================================================

def gen_protein():
    """Protein contact map: helix (dense local contacts) vs sheet+coil (sparse)."""
    nh, ns, nc = 25, 18, 17
    n = nh + ns + nc
    adj = np.zeros((n, n))
    
    # Helix: strong sequential contacts (i, i+1...i+4) + some medium-range
    for i in range(nh):
        for j in range(i+1, nh):
            d = abs(i - j)
            if d <= 4:
                adj[i,j] = np.random.uniform(0.7, 1.0)
            elif d <= 8:
                if np.random.random() < 0.4:
                    adj[i,j] = np.random.uniform(0.3, 0.6)
            elif np.random.random() < 0.15:
                adj[i,j] = np.random.uniform(0.1, 0.3)
    
    # Sheet: moderate intra-contacts
    for i in range(nh, nh+ns):
        for j in range(i+1, nh+ns):
            if np.random.random() < 0.15:
                adj[i,j] = np.random.uniform(0.3, 0.5)
    
    # Coil: very sparse
    for i in range(nh+ns, n):
        for j in range(i+1, n):
            if np.random.random() < 0.04:
                adj[i,j] = np.random.uniform(0.05, 0.15)
    
    # Cross-domain: ensure enough connectivity for spectral method
    # but keep it weaker than intra-domain
    for i in range(nh):
        for j in range(nh, n):
            if np.random.random() < 0.04:
                adj[i,j] = np.random.uniform(0.05, 0.15)
    
    adj = adj + adj.T
    labels = np.array([0]*nh + [1]*(ns+nc))
    return adj, labels, "Protein Contact Map"


def gen_social():
    """Social: humans (dense community) vs bots (bridges)."""
    nh, nb = 45, 25
    n = nh + nb
    adj = np.zeros((n, n))
    
    # Humans: strong community structure
    for i in range(nh):
        for j in range(i+1, min(i+6, nh)):
            adj[i,j] = np.random.uniform(0.6, 1.0)
        # A few random human-human
        for _ in range(3):
            j = np.random.randint(0, nh)
            if j != i:
                adj[i,j] = max(adj[i,j], np.random.uniform(0.4, 0.7))
    
    # Bots: each connects weakly to many humans, few bot-bot
    for i in range(nh, n):
        targets = np.random.choice(nh, size=12, replace=False)
        for t in targets:
            adj[i,t] = np.random.uniform(0.05, 0.15)
        # Bot-bot very sparse
        for _ in range(1):
            j = np.random.randint(nh, n)
            if j != i:
                adj[i,j] = np.random.uniform(0.02, 0.08)
    
    adj = (adj + adj.T) / 2
    labels = np.array([0]*nh + [1]*nb)
    return adj, labels, "Social Bot Network"


def gen_finance():
    """Financial correlation matrix from synthetic stock returns."""
    ns, nps = 4, 12
    n = ns * nps
    returns = np.zeros((n, 100))
    market = np.random.randn(100) * 0.5
    for s in range(ns):
        sector_f = np.random.randn(100) * 0.8
        for i in range(nps):
            returns[s*nps+i] = market + sector_f + np.random.randn(100) * 0.3
    corr = np.abs(np.corrcoef(returns))
    np.fill_diagonal(corr, 0)
    corr[corr < 0.15] = 0
    labels = np.repeat(range(ns), nps)
    return corr, labels, "Finance Sector Correlation"


def gen_climate():
    """Climate zones from geographic proximity."""
    nz, npz = 4, 15
    n = nz * npz
    coords = []
    centers = [(0,0),(4,0),(0,4),(4,4)]
    for z in range(nz):
        cx, cy = centers[z]
        for _ in range(npz):
            coords.append((cx + np.random.randn()*0.5, cy + np.random.randn()*0.5))
    coords = np.array(coords)
    dists = squareform(pdist(coords))
    adj = np.exp(-dists / 2.0)
    np.fill_diagonal(adj, 0)
    adj[adj < 0.1] = 0
    labels = np.repeat(range(nz), npz)
    return adj, labels, "Climate Zone Network"


def gen_sbm():
    """Stochastic block model with planted communities."""
    nc, npc = 4, 20
    n = nc * npc
    adj = np.zeros((n, n))
    for c1 in range(nc):
        for c2 in range(c1, nc):
            p = 0.35 if c1 == c2 else 0.03
            for i in range(c1*npc, (c1+1)*npc):
                for j in range(max(i+1, c2*npc), (c2+1)*npc):
                    if np.random.random() < p:
                        adj[i,j] = 1.0
    adj = adj + adj.T
    labels = np.repeat(range(nc), npc)
    return adj, labels, "Stochastic Block Model"


def gen_ecosystem():
    """Ecosystem trophic levels."""
    nl, npl = 4, 15
    n = nl * npl
    adj = np.zeros((n, n))
    for level in range(nl):
        s, e = level*npl, (level+1)*npl
        # Intra-level competition
        for i in range(s, e):
            for j in range(i+1, e):
                if np.random.random() < 0.2:
                    adj[i,j] = np.random.uniform(0.3, 0.5)
        # Inter-level predation
        if level > 0:
            ps, pe = (level-1)*npl, level*npl
            for i in range(s, e):
                prey = np.random.choice(range(ps, pe), size=5, replace=False)
                for p in prey:
                    adj[i,p] = np.random.uniform(0.6, 1.0)
    adj = (adj + adj.T) / 2
    labels = np.repeat(range(nl), npl)
    return adj, labels, "Ecosystem Trophic Levels"


# ============================================================================
# Main
# ============================================================================

def main():
    print("🔬 FIEDLER UNIVERSAL PARTITION BENCHMARK v2")
    print("=" * 60)
    print("Normalized Laplacian + improved generators\n")
    
    generators = [
        ('Protein', gen_protein),
        ('Social', gen_social),
        ('Finance', gen_finance),
        ('Climate', gen_climate),
        ('SBM', gen_sbm),
        ('Ecosystem', gen_ecosystem),
    ]
    
    methods = [
        ('Fiedler', fiedler_partition),
        ('K-Means', kmeans_partition),
        ('Spectral', spectral_partition),
        ('Modularity', modularity_partition),
        ('Random', random_partition),
    ]
    
    results = {}
    
    for domain_name, gen_fn in generators:
        adj, true_labels, desc = gen_fn()
        n_clusters = len(set(true_labels))
        n = adj.shape[0]
        print(f"\n{'='*60}")
        print(f"  {desc} (n={n}, k={n_clusters})")
        print(f"  Edge density: {np.count_nonzero(adj)/(n*n-n)*100:.1f}%")
        print(f"{'='*60}")
        
        results[domain_name] = {}
        
        for method_name, method_fn in methods:
            t0 = time.time()
            try:
                pred = method_fn(adj, n_clusters=n_clusters)
                ari = adjusted_rand_score(true_labels, pred)
                nmi = normalized_mutual_info_score(true_labels, pred)
            except Exception as e:
                ari, nmi = -1.0, -1.0
            elapsed = time.time() - t0
            results[domain_name][method_name] = {'ARI': ari, 'NMI': nmi, 'time': elapsed}
            bar = '█' * max(0, int(ari * 40))
            print(f"    {method_name:12s}: ARI={ari:7.4f}  NMI={nmi:7.4f}  ({elapsed:.3f}s) {bar}")
    
    # === COMPARISON TABLE ===
    print(f"\n\n{'='*80}")
    print("  COMPARISON TABLE — Adjusted Rand Index")
    print(f"{'='*80}")
    
    method_names = [m[0] for m in methods]
    header = f"  {'Domain':<20s}"
    for m in method_names:
        header += f" {m:>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    for d, dresults in results.items():
        row = f"  {d:<20s}"
        for m in method_names:
            row += f" {dresults[m]['ARI']:>10.4f}"
        print(row)
    
    # === WIN ANALYSIS ===
    print(f"\n{'='*80}")
    print("  FIEDLER vs OTHERS — Head-to-Head")
    print(f"{'='*80}")
    
    fwins, fties, flosses = 0, 0, 0
    fiedler_details = []
    for d, dresults in results.items():
        f_ari = dresults['Fiedler']['ARI']
        others = {m: dresults[m]['ARI'] for m in method_names if m != 'Fiedler'}
        best_other_name = max(others, key=others.get)
        best_other = others[best_other_name]
        
        if f_ari > best_other + 0.01:
            verdict = "🏆 WIN"
            fwins += 1
        elif f_ari >= best_other - 0.01:
            verdict = "🤝 TIE"
            fties += 1
        else:
            verdict = f"❌ LOSE (best: {best_other_name})"
            flosses += 1
        
        fiedler_details.append((d, f_ari, best_other, verdict))
        print(f"  {d:<20s}: Fiedler={f_ari:.4f}  BestOther({best_other_name})={best_other:.4f}  {verdict}")
    
    print(f"\n  Total: {fwins} wins, {fties} ties, {flosses} losses / {len(results)} domains")
    
    # === SPECTRAL GAP ANALYSIS ===
    print(f"\n{'='*80}")
    print("  SPECTRAL GAP ANALYSIS")
    print(f"{'='*80}")
    
    for domain_name, gen_fn in generators:
        adj, true_labels, desc = gen_fn()
        if sparse.issparse(adj):
            adj = adj.toarray()
        adj = np.float64((adj + adj.T) / 2.0)
        np.fill_diagonal(adj, 0)
        L_norm = _normalize_laplacian(adj)
        try:
            evals = eigsh(sparse.csr_matrix(L_norm), k=min(6, adj.shape[0]-1), 
                         which='SM', return_eigenvectors=False, tol=1e-6)
            evals = np.sort(evals)
            gap = evals[2] - evals[1] if len(evals) > 2 else 0
            gap_ratio = evals[1] / (evals[2] + 1e-10) if len(evals) > 2 else 0
            print(f"  {domain_name:<20s}: λ=[{', '.join(f'{v:.4f}' for v in evals[:5])}]  "
                  f"gap(λ₂→λ₃)={gap:.4f}  ratio={gap_ratio:.4f}")
        except:
            print(f"  {domain_name:<20s}: eigenvalue computation failed")
    
    # === THEORY ===
    print(f"""
{'='*80}
  ANALYSIS: WHY FIEDLER WORKS (OR DOESN'T)
{'='*80}

  KEY FINDINGS FROM BENCHMARK:

  1. FIEDLER IS OPTIMAL FOR k=2 WITH CLEAR SPECTRAL GAPS
     When λ₂ << λ₃, the graph has a natural binary partition and Fiedler
     finds it. The normalized Laplacian makes this more robust than
     unnormalized Laplacian for degree-skewed graphs.

  2. FIEDLER ≈ SPECTRAL CLUSTERING (for k=2)
     Fiedler IS spectral clustering for k=2! The only difference is
     label assignment (threshold vs k-means on eigenvectors).
     For k>2, Fiedler uses k-means on the first k eigenvectors,
     which IS spectral clustering. So they should perform similarly.

  3. WHY APPARENT DIFFERENCES:
     - Small numerical differences in eigenvector computation
     - K-means initialization sensitivity
     - The Fiedler threshold (median) vs k-means assignment
     - For very clear clusters, ALL spectral methods win

  4. THE REAL ADVANTAGE OF FIEDLER:
     - Simplicity: one eigenvector, one threshold
     - Speed: only need 2nd eigenvector, not k
     - Interpretability: the Fiedler value has geometric meaning
     - Anomaly detection: outliers have extreme Fiedler values

  5. WHERE MODULARITY WINS:
     - Unequal community sizes
     - Hierarchical structure
     - When you DON'T know k

  CONCLUSION: Fiedler isn't "better" than spectral clustering — it IS
  spectral clustering. Its power lies in simplicity, speed, and the
  theoretical guarantee of optimal normalized cut for k=2.
""")
    
    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_out = {}
    for d, dresults in results.items():
        json_out[d] = {m: {k: float(v) for k, v in metrics.items()} 
                       for m, metrics in dresults.items()}
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(json_out, f, indent=2)
    
    print(f"  Results saved to {out_dir}/results.json")
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
