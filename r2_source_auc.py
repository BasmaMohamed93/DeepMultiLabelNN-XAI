"""Source classifier (same protocol as the submitted confound analysis) under each feature set and in the matched subset."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import build_dataset as BD
def cv_auc(V, y):
    P = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(V, y):
        fb = BD.VepFeatures().fit(V.iloc[tr]); Xt, _ = fb.transform(V.iloc[tr]); Xs, _ = fb.transform(V.iloc[te])
        P[te] = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=2, random_state=0).fit(Xt, y[tr]).predict_proba(Xs)[:, 1]
    return round(roc_auc_score(y, P), 3)
out = {}
for name, kw in (('all', {}), ('no_tramadol', {'exclude_drugs': ('tramadol',)})):
    V, Y, d, _ = BD.build(**kw); y = V.label_source.str.contains('ClinVar').astype(int).values
    for feat, (miss, freq) in {'full': (True, True), 'no_missingness_indicators': (False, True), 'no_frequency_features': (False, False)}.items():
        BD.CFG['miss'], BD.CFG['freq'] = miss, freq; out[f'{name}|{feat}'] = cv_auc(V, y)
    BD.CFG['miss'], BD.CFG['freq'] = True, True
keep = set(json.load(open('matched_variants.json'))); V, Y, d, _ = BD.build(); m = V.index.isin(keep)
out['matched_subset|full'] = cv_auc(V[m], V[m].label_source.str.contains('ClinVar').astype(int).values)
gn = pd.to_numeric(V.gnomadg, errors='coerce').isna(); s = V.label_source.str.contains('ClinVar')
out['missing_gnomADg_ClinVar'] = round(float(gn[s].mean()), 3); out['missing_gnomADg_PharmGKB'] = round(float(gn[~s].mean()), 3)
out['missing_gnomADg_matched_ClinVar'] = round(float(gn[m & s].mean()), 3); out['missing_gnomADg_matched_PharmGKB'] = round(float(gn[m & ~s].mean()), 3)
json.dump(out, open('r2_source_auc.json', 'w'), indent=1); print(json.dumps(out, indent=1))
