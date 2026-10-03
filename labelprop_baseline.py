"""Graph baseline requested by Reviewer 4 (graph-based link prediction): label propagation (Zhou et al., 2004) on a
variant graph with three edge types (same gene; same chromosome within 100 kb; 10 nearest neighbours in standardized
VEP-feature space) and, for unseen genes, DGIdb similar-gene edges. Training labels are clamped; validation selects
alpha and the decoding rule; the test partition is scored once. Same splits and both evaluation protocols as."""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scipy.sparse as sp
from build_dataset import build, VepFeatures
from splits import gene_split, random_split, within_gene_auroc
from evalkit import metrics
from pairwise import choose_decoding, decode, dec_metrics
from model import balanced_eval, balanced_threshold

KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}}


def graph(V, G, X):
    n = len(V); rows, cols = [], []
    for g in np.unique(G):                                    # same-gene edges
        if g == 'NA': continue
        i = np.where(G == g)[0]
        if len(i) > 1:
            a, b = np.meshgrid(i, i); m = a != b; rows += list(a[m]); cols += list(b[m])
    chrom = V.seq_region_name.astype(str).values; pos = V.start.astype(float).values
    for c in np.unique(chrom):                                # regional edges (<= 100 kb)
        i = np.where(chrom == c)[0]; D = np.abs(pos[i][:, None] - pos[i][None, :]); a, b = np.where((D <= 1e5) & (D > 0))
        rows += list(i[a]); cols += list(i[b])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6); sq = (Z ** 2).sum(1)
    for s in range(0, n, 1000):                               # kNN edges in feature space
        d = sq[s:s + 1000, None] + sq[None, :] - 2 * Z[s:s + 1000] @ Z.T; np.fill_diagonal(d[:, s:s + 1000], np.inf)
        nn = np.argsort(d, 1)[:, :10]; rows += list(np.repeat(np.arange(s, min(s + 1000, n)), 10)); cols += list(nn.ravel())
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); W = ((W + W.T) > 0).astype(float)
    d = np.asarray(W.sum(1)).ravel(); d[d == 0] = 1; Dm = sp.diags(1 / np.sqrt(d))
    return Dm @ W @ Dm


def propagate(S, Y0, alpha, it=30):
    F = Y0.copy()
    for _ in range(it): F = alpha * (S @ F) + (1 - alpha) * Y0
    return F


rows = []
for ds, kw in KW.items():
    V, Y, drugs, _ = build(**kw); G = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
            fb = VepFeatures().fit(V.iloc[tr]); X = fb.transform(V)[0]; S = graph(V, G, X)
            Y0 = np.zeros_like(Y); Y0[tr] = Y[tr]; p0 = Y[tr].mean(0)
            best = None
            for a in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
                F = propagate(S, Y0, a); F = F / np.maximum(F.sum(1, keepdims=True), 1e-9) * 1.3 * (F.sum(1, keepdims=True) > 0) + 1e-3 * p0
                ll = -np.mean(Y[va] * np.log(np.clip(F[va], 1e-6, 1)) + (1 - Y[va]) * np.log(np.clip(1 - F[va], 1e-6, 1)))
                if best is None or ll < best[0]: best = (ll, a, F)
            _, a, F = best; F = np.clip(F, 0, 1)
            name, th, ft = choose_decoding(Y[va], F[va]); Ys, Ps = Y[te], F[te]
            m = dec_metrics(Ys, decode(Ps, th, ft)); M = metrics(Ys, Ps, th); wg, _ = within_gene_auroc(Ys, Ps, G[te])
            o = np.argsort(-Ps, 1); top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :10], True, 1)
            b = balanced_eval(Ys, Ps + 1e-9 * np.random.default_rng(s).random(Ps.shape), balanced_threshold(Y[va], F[va], np.random.default_rng(s + 100)), np.random.default_rng(s + 200))
            rows.append(dict(dataset=ds, setting=st, seed=s, alpha=a, subset_acc=m['subset_acc'], precision=m['precision'], recall=m['recall'], micro_f1=m['f1'],
                             macro_auroc=M['macro_auroc'], within_gene_auroc=wg, **{'hit@10': float(((top & (Ys > 0)).sum(1) > 0).mean())},
                             bal_accuracy=b['accuracy'], bal_precision=b['precision'], bal_recall=b['recall'], bal_f1=b['f1'], bal_auroc=b['auroc']))
            print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv('labelprop_runs.csv', index=False)
print(D.groupby(['dataset', 'setting']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
