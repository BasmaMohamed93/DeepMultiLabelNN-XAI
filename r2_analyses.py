"""Second-revision analyses that use saved predictions (no retraining):
matched-subset and source-stratified evaluation, within-gene group counts, regional-prior coverage."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from build_dataset import build
from splits import gene_split, random_split, within_gene_auroc
from pairwise import gene_prior, decode, dec_metrics

KW = {'all': {}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}, 'high_evidence': {'drop_levels': ('3',)}}

def mets(Y, P, B):
    both = (Y.sum(0) > 0) & (Y.sum(0) < len(Y))
    tp = (B & (Y > 0)).sum(); fp = (B & (Y == 0)).sum(); fn = (~B & (Y > 0)).sum()
    return dict(n=len(Y), drugs=int(both.sum()),
                macro_auroc=float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in np.where(both)[0]])),
                macro_auprc=float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in np.where(both)[0]])),
                micro_f1=float(2 * tp / max(2 * tp + fp + fn, 1)))

def load(ds, st, sd, res_dir='results', suffix='_region=True_rank=False_knn=True'):
    fn = f'{res_dir}/{ds}_{st}_{sd}{suffix}'
    return np.load(fn + '.npz'), json.load(open(fn + '.json'))

def subset_eval(ds, keep_fn, label, res_dir='results', suffix='_region=True_rank=False_knn=True'):
    V, Y, drugs, _ = build(**KW[ds]); genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    rows = []
    for st in ('A_random', 'B_gene_disjoint'):
        for sd in range(5):
            z, js = load(ds, st, sd, res_dir, suffix); te = z['te']; Ps = z['Ps']; th = z['th']
            tr, va, te2 = random_split(Y, sd) if st == 'A_random' else gene_split(genes, Y, sd)
            assert (te2 == te).all()
            Gt, Gv, Gs, p0 = gene_prior(genes, Y, tr, va, te, sd)
            mask = keep_fn(V.iloc[te], Y[te])
            for who, P in (('model', Ps), ('gene_only', Gs)):
                thr = th if who == 'model' else None
                for part, m in (('all test', np.ones(len(te), bool)), (label, mask)):
                    Yt = Y[te][m]; Pt = P[m]
                    if who == 'model': B = decode(Pt, th, js['force_top1'])
                    else:
                        from pairwise import choose_decoding
                        n2, th2, ft2 = choose_decoding(Y[va], Gv); B = decode(Pt, th2, ft2)
                    rows.append(dict(dataset=ds, setting=st, seed=sd, model=who, subset=part, **mets(Yt, Pt, B)))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    out = {}
    # 1. propensity-matched (source-balanced) subset
    keep = set(json.load(open('matched_variants.json')))
    D1 = subset_eval('all', lambda Vt, Yt: Vt.index.isin(keep), 'source-matched')
    # 5. source-stratified
    def src(kind):
        return lambda Vt, Yt: (Vt.label_source.fillna('').str.contains('ClinVar') & ~Vt.label_source.fillna('').str.contains('PharmGKB')).values if kind == 'cv' else \
            (Vt.label_source.fillna('').str.contains('PharmGKB') & ~Vt.label_source.fillna('').str.contains('ClinVar')).values
    D2 = subset_eval('all', src('cv'), 'ClinVar-only variants'); D3 = subset_eval('all', src('pg'), 'PharmGKB-only variants')
    D4 = subset_eval('high_evidence', src('cv'), 'ClinVar-only variants'); D5 = subset_eval('high_evidence', src('pg'), 'PharmGKB-only variants')
    D = pd.concat([D1, D2, D3, D4, D5]).drop_duplicates()
    D.to_csv('r2_subset_eval.csv', index=False)
    S = D.groupby(['dataset', 'setting', 'subset', 'model'])[['n', 'drugs', 'macro_auroc', 'macro_auprc', 'micro_f1']].agg(['mean', 'std']).round(3)
    print(S.to_string())
