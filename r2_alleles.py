import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, re
from build_dataset import build
V,Y,drugs,_=build()
v=pd.read_csv('vep.csv.gz',low_memory=False)
prim=set([str(i) for i in range(1,23)]+['X','Y','MT'])
alle=v[v.input.isin(V.index)&v.seq_region_name.astype(str).isin(prim)].groupby('input').allele_string.first()
nalt=alle.map(lambda s: len(str(s).split('/'))-1)
multi=set(nalt[nalt>1].index)
print('multi-allelic', len(multi), 'of', len(V), round(len(multi)/len(V),3))
comp=str.maketrans('ACGT','TGCA')
def vep_alt(row):
    c=str(row.codons)
    if '/' not in c: return None
    a,b=c.split('/'); d=[(x,y) for x,y in zip(a,b) if x.isupper() or y.isupper()]
    if len(a)!=len(b) or len(d)!=1: return None
    alt=d[0][1].upper()
    if str(row.strand) in ('-1','-1.0'): alt=alt.translate(comp)
    return alt
V['vep_alt']=V.apply(vep_alt,axis=1)
cv=pd.read_csv('/home/claude/v2/clinvar_pgx_records.csv')
cv=cv[cv.CLNSIG.astype(str).str.contains('drug_response')]
cva=cv.groupby('rs').alt.apply(set)
L=pd.read_csv('label_records_raw.csv'); src=L.groupby('variant').source.apply(set)
rows=[]
for rs in multi:
    a=V.at[rs,'vep_alt']
    if a is None or pd.isna(a): continue
    s=src.get(rs,set())
    if 'ClinVar' in s and rs in cva:
        rows.append(('ClinVar',a in cva[rs]))
va=pd.read_csv('/home/claude/v2/raw/ddecc963-variantAnnotations_1/var_drug_ann.tsv',sep='\t')
va=va[va['Variant/Haplotypes'].isin(multi)]
vaa=va.groupby('Variant/Haplotypes').Alleles.apply(lambda x: set(''.join(str(i) for i in x if str(i)!='nan').replace('+','').replace('/','').replace(' ','').upper()))
ca=pd.read_csv('/home/claude/v2/raw/25eff430-clinicalAnnotations_1/clinical_annotations.tsv',sep='\t')
cal=pd.read_csv('/home/claude/v2/raw/25eff430-clinicalAnnotations_1/clinical_ann_alleles.tsv',sep='\t')
cal=cal.merge(ca[['Clinical Annotation ID','Variant/Haplotypes']],on='Clinical Annotation ID')
cal=cal[cal['Variant/Haplotypes'].isin(multi)]
caa=cal.groupby('Variant/Haplotypes')['Genotype/Allele'].apply(lambda x: set(''.join(str(i) for i in x).upper()))
for rs in multi:
    a=V.at[rs,'vep_alt']
    if a is None or pd.isna(a): continue
    s=src.get(rs,set())
    ev=set()
    if 'PharmGKB_VA' in s and rs in vaa: ev|=vaa[rs]
    if 'PharmGKB_CA' in s and rs in caa: ev|=caa[rs]
    if ev: rows.append(('PharmGKB',a in ev))
R=pd.DataFrame(rows,columns=['source','match'])
print('multi-allelic coding SNVs with checkable allele:',len(R))
print(R.groupby('source').match.agg(['size','mean']).round(3))
noncoding=sum(1 for rs in multi if pd.isna(V.at[rs,'vep_alt'])); print('multi-allelic not checkable (non-coding or indel):',noncoding)
ex=[]
for rs in list(multi):
    a=V.at[rs,'vep_alt']
    if pd.isna(a) or rs not in cva or 'ClinVar' not in src.get(rs,set()): continue
    ex.append((rs,V.at[rs,'allele_string'],V.at[rs,'codons'],V.at[rs,'strand'],a,cva[rs],cv[cv.rs==rs].ref.iloc[0]))
for e in ex[:12]: print(e)
import json
status={}
for rs in V.index:
    if rs not in multi: status[rs]='biallelic'; continue
    a=V.at[rs,'vep_alt']
    if pd.isna(a): status[rs]='multi_uncheckable'; continue
    s=src.get(rs,set()); ev=set()
    if 'ClinVar' in s and rs in cva: ev|=cva[rs]
    if 'PharmGKB_VA' in s and rs in vaa: ev|=vaa[rs]
    if 'PharmGKB_CA' in s and rs in caa: ev|=caa[rs]
    status[rs]='multi_uncheckable' if not ev else ('multi_match' if a in ev else 'multi_mismatch')
json.dump(status,open('allele_status.json','w'))
print(pd.Series(status).value_counts())
