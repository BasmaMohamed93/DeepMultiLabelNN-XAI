"""Variant-level dataset: one row per rsID, labels = drugs with documented variant-level evidence.
Features come ONLY from Ensembl VEP (uniform for every variant regardless of label source)."""
import json, re
import numpy as np, pandas as pd

MIN_VAR, MIN_GENE = 30, 3
SEV = ['transcript_ablation', 'splice_acceptor_variant', 'splice_donor_variant', 'stop_gained', 'frameshift_variant',
       'stop_lost', 'start_lost', 'inframe_insertion', 'inframe_deletion', 'missense_variant',
       'protein_altering_variant', 'splice_region_variant', 'synonymous_variant', 'stop_retained_variant',
       '5_prime_UTR_variant', '3_prime_UTR_variant', 'non_coding_transcript_exon_variant', 'intron_variant',
       'upstream_gene_variant', 'downstream_gene_variant', 'regulatory_region_variant', 'intergenic_variant']
rank = {c: i for i, c in enumerate(SEV)}


def build(drop_level4=True, exclude_drugs=(), sources=None, drop_levels=()):
    flow = {}
    L = pd.read_csv('label_records_raw.csv')
    flow['label_records_raw'] = len(L)
    if drop_level4:
        L = L[L.level.astype(str) != '4']
    flow['after_drop_level4'] = len(L)
    if drop_levels:
        L = L[~L.level.astype(str).isin(drop_levels)]
    L = L[~L.drug.isin(exclude_drugs)]
    if sources is not None:
        L = L[L.source.isin(sources)]
    P = L.drop_duplicates(['variant', 'drug'])
    flow['unique_variant_drug_pairs'] = len(P)

    v = pd.read_csv('vep.csv.gz', low_memory=False)
    flow['vep_rows'] = len(v)
    v['sev'] = v.most_severe_consequence.map(rank).fillna(len(SEV))
    # one row per rsID: keep the most severe allele; population frequencies = max over alleles
    fcols = [c for c in v.columns if c == 'af' or c.startswith('gnomad')]
    fmax = v.groupby('input')[fcols].max()
    v = v.sort_values(['input', 'sev', 'cadd_phred'], ascending=[True, True, False]).drop_duplicates('input')
    v = v.set_index('input').drop(columns=fcols).join(fmax)
    flow['vep_unique_variants'] = len(v)

    P = P[P.variant.isin(v.index)]
    flow['pairs_with_vep'] = len(P)
    # gene: VEP canonical gene, fallback to label-source gene
    gmap = v.gene_symbol.dropna().to_dict()
    P = P.assign(gene=P.variant.map(gmap).fillna(P.gene))
    g = P.groupby('drug').agg(nvar=('variant', 'nunique'), ngene=('gene', 'nunique'))
    drugs = g[(g.nvar >= MIN_VAR) & (g.ngene >= MIN_GENE)].sort_values('nvar', ascending=False).index.tolist()
    P = P[P.drug.isin(drugs)]
    variants = sorted(P.variant.unique())
    idx = {x: i for i, x in enumerate(variants)}
    Y = np.zeros((len(variants), len(drugs)), np.float32)
    for x, d in zip(P.variant, P.drug):
        Y[idx[x], drugs.index(d)] = 1
    V = v.loc[variants].copy()
    V['gene'] = [gmap.get(x) or P[P.variant == x].gene.iloc[0] for x in variants]
    src = P.groupby('variant').source.agg(lambda s: '|'.join(sorted(set(s))))
    V['label_source'] = src.reindex(variants).values
    flow.update(variants_final=len(variants), genes_final=int(V.gene.nunique()), drugs_final=len(drugs),
                label_cardinality=float(Y.sum(1).mean()), label_density=float(Y.mean()))
    return V, Y, drugs, flow


CONS_VOCAB = SEV


class VepFeatures:
    """Variant-intrinsic features. Excluded as gene/position proxies: chromosome, start/end, strand,
    gene symbol/id, n_transcripts, absolute cds/protein positions."""
    NUM = ['sift_score', 'polyphen_score', 'cadd_phred', 'cadd_raw', 'blosum62', 'distance',
           'af', 'gnomade', 'gnomadg', 'gnomade_afr', 'gnomade_eas', 'gnomade_nfe', 'gnomadg_afr', 'gnomadg_eas', 'gnomadg_nfe']
    CAT = ['variant_class', 'impact', 'biotype', 'sift_prediction', 'polyphen_prediction', 'most_severe_consequence']

    def __init__(self, groups=('functional', 'consequence', 'frequency')):
        self.groups = groups

    def fit(self, V):
        X = V[self.NUM].apply(pd.to_numeric, errors='coerce')
        X[[c for c in self.NUM if c.startswith(('af', 'gnomad'))]] = np.log10(X[[c for c in self.NUM if c.startswith(('af', 'gnomad'))]] + 1e-6)
        X['distance'] = np.log1p(X['distance'])
        self.med = X.median()
        Xf = X.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        self.cat = {c: sorted(V[c].astype(str).fillna('nan').unique()) for c in self.CAT}
        aa = V.amino_acids.astype(str).fillna('nan').str.split('/')
        self.aa = sorted(set(aa.str[0]) | set(aa.str[-1]))
        return self

    def transform(self, V):
        B, names = [], []
        X = V[self.NUM].apply(pd.to_numeric, errors='coerce')
        fc = [c for c in self.NUM if c.startswith(('af', 'gnomad'))]
        X[fc] = np.log10(X[fc] + 1e-6)
        X['distance'] = np.log1p(X['distance'])
        miss = X.isna().astype(np.float32)
        Z = ((X.fillna(self.med) - self.mu) / self.sd).astype(np.float32)
        func = [c for c in self.NUM if c not in fc]
        if 'functional' in self.groups:
            B += [Z[func].values, miss[func].values]
            names += [f'num:{c}' for c in func] + [f'missing:{c}' for c in func]
            for c in ['sift_prediction', 'polyphen_prediction']:
                B.append(np.stack([(V[c].astype(str).fillna('nan') == x).values for x in self.cat[c]], 1).astype(np.float32))
                names += [f'{c}={x}' for x in self.cat[c]]
            aa = V.amino_acids.astype(str).fillna('nan').str.split('/')
            B.append(np.stack([(aa.str[0] == a).values for a in self.aa], 1).astype(np.float32))
            B.append(np.stack([(aa.str[-1] == a).values for a in self.aa], 1).astype(np.float32))
            names += [f'aa_ref:{a}' for a in self.aa] + [f'aa_alt:{a}' for a in self.aa]
        if 'frequency' in self.groups:
            B += [Z[fc].values, miss[fc].values]
            names += [f'num:{c}' for c in fc] + [f'missing:{c}' for c in fc]
        if 'consequence' in self.groups:
            for c in ['variant_class', 'impact', 'biotype', 'most_severe_consequence']:
                B.append(np.stack([(V[c].astype(str).fillna('nan') == x).values for x in self.cat[c]], 1).astype(np.float32))
                names += [f'{c}={x}' for x in self.cat[c]]
            terms = V.consequence_terms.astype(str).fillna('')
            B.append(np.stack([terms.str.contains(t, regex=False).values for t in CONS_VOCAB], 1).astype(np.float32))
            names += [f'csq:{t}' for t in CONS_VOCAB]
        return np.hstack(B).astype(np.float32), names


if __name__ == '__main__':
    V, Y, drugs, flow = build()
    print(json.dumps(flow, indent=1))
    X, n = VepFeatures().fit(V).transform(V)
    print('X', X.shape)
    print(V.label_source.value_counts().head())
