import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from r2_analyses import subset_eval
S = json.load(open('allele_status.json'))
D1 = subset_eval('all', lambda Vt, Yt: Vt.index.map(lambda r: S.get(r) != 'multi_mismatch').values.astype(bool), 'excluding allele-mismatched')
D2 = subset_eval('all', lambda Vt, Yt: Vt.index.map(lambda r: S.get(r) == 'biallelic').values.astype(bool), 'biallelic only')
D = pd.concat([D1, D2]).drop_duplicates(); D.to_csv('r2_allele_eval.csv', index=False)
print(D.groupby(['setting', 'subset', 'model'])[['n', 'macro_auroc', 'macro_auprc', 'micro_f1']].agg(['mean', 'std']).round(3).to_string())
