"""Additional analyses: (a) architecture ablation of the residual network on the variant-level data,
(b) random-forest multi-label baseline, (c) label-vocabulary check using training partitions only, (d) parameter count."""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from build_dataset import build, VepFeatures
from splits import gene_split, random_split, within_gene_auroc
from evalkit import metrics
from pairwise import choose_decoding, decode, dec_metrics
from model import balanced_eval, balanced_threshold
from nn_training import nn_fit
from resnet import ResMLP

stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
V, Y, drugs, _ = build(); G = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)


def evaluate(Yv, Pv, Ys, Ps, Gte, seed):
    n, th, ft = choose_decoding(Yv, Pv); m = dec_metrics(Ys, decode(Ps, th, ft)); M = metrics(Ys, Ps, th); wg, _ = within_gene_auroc(Ys, Ps, Gte)
    b = balanced_eval(Ys, Ps, balanced_threshold(Yv, Pv, np.random.default_rng(seed + 100)), np.random.default_rng(seed + 200))
    o = np.argsort(-Ps, 1); h = {}
    for k in (1, 10):
        top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :k], True, 1); h[f'hit@{k}'] = float(((top & (Ys > 0)).sum(1) > 0).mean())
    return dict(subset_acc=m['subset_acc'], **h, micro_f1=m['f1'], precision=m['precision'], recall=m['recall'], macro_auroc=M['macro_auroc'], within_gene_auroc=wg,
                bal_accuracy=b['accuracy'], bal_f1=b['f1'], bal_auroc=b['auroc'])


if stage in ('ablation', 'all'):
    CFG = {'full': {}, 'no_residual': {'residual': False}, 'no_batchnorm': {'bn': False}, 'relu_instead_of_gelu': {'act': 'relu'},
           'no_label_smoothing': {'smoothing': 0.0}, 'no_class_weights': {'class_weight': False}, 'no_cosine_annealing': {'cosine': False},
           'no_early_stopping': {'early_stop': False}}
    rows = []
    out = 'arch_ablation.csv'
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
            fb = VepFeatures().fit(V.iloc[tr]); Xt, Xv, Xs = (fb.transform(V.iloc[i])[0] for i in (tr, va, te))
            for name, kw in CFG.items():
                if len(done) and ((done.setting == st) & (done.seed == s) & (done.config == name)).any(): continue
                nn, hp = nn_fit(Xt, Y[tr], Xv, Y[va], s, no_grid=True, **kw)
                r = dict(setting=st, seed=s, config=name, best_epoch=hp.get('best_epoch'), **evaluate(Y[va], nn.predict_proba(Xv), Y[te], nn.predict_proba(Xs), G[te], s))
                done = pd.concat([done, pd.DataFrame([r])]); done.to_csv(out, index=False); print(r, flush=True)
    print(done.groupby(['setting', 'config']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())

if stage in ('rf', 'all'):
    rows = []
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
            fb = VepFeatures().fit(V.iloc[tr]); Xt, Xv, Xs = (fb.transform(V.iloc[i])[0] for i in (tr, va, te))
            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features='sqrt', n_jobs=2, random_state=s).fit(Xt, Y[tr])
            P = lambda X: np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(X)) for p in rf.predict_proba(X)])
            r = dict(setting=st, seed=s, **evaluate(Y[va], P(Xv), Y[te], P(Xs), G[te], s)); rows.append(r); print(r, flush=True)
    D = pd.DataFrame(rows); D.to_csv('random_forest.csv', index=False)
    print(D.groupby('setting').mean(numeric_only=True).drop(columns='seed').round(3).to_string())

if stage in ('labelcheck', 'all'):
    # would each retained drug also pass the >=30 variants / >=3 genes rule if counted on the training partition only
    # (threshold scaled to the 70% training fraction), and is performance different for drugs that would fail?
    import glob
    rows = []
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
            d = np.load(f'results/all_{st}_{s}_region=True_rank=False_knn=True.npz'); assert (d['te'] == te).all()
            nvar = Y[tr].sum(0); ngene = np.array([len(set(G[tr][Y[tr][:, j] == 1])) for j in range(len(drugs))])
            ok = (nvar >= 0.7 * 30) & (ngene >= 3)
            Ys, Ps = Y[te], d['Ps']
            au = np.array([roc_auc_score(Ys[:, j], Ps[:, j]) if 0 < Ys[:, j].sum() < len(Ys) else np.nan for j in range(len(drugs))])
            rows.append(dict(setting=st, seed=s, n_drugs=len(drugs), pass_train_only=int(ok.sum()), macro_auroc_all=np.nanmean(au),
                             macro_auroc_pass=np.nanmean(au[ok]), macro_auroc_fail=np.nanmean(au[~ok]) if (~ok).any() else np.nan))
    D = pd.DataFrame(rows); D.to_csv('label_vocabulary_check.csv', index=False)
    print(D.groupby('setting').mean(numeric_only=True).drop(columns='seed').round(3).to_string())

if stage in ('params', 'all'):
    m = ResMLP(181, len(drugs)); print('trainable parameters', sum(v.size for mm in m.mods for v in mm.p.values()))
