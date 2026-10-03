"""Variant-level drug labels from ClinVar drug_response + PharmGKB/ClinPGx (clinical & variant annotations).
Input files (public releases, see Table 1 of the article) are read from raw/; the output label_records_raw.csv is included in this repository."""
import gzip, re, json, collections
import pandas as pd
R='raw/'
def norm(d):
    d=d.strip().lower()
    d=re.sub(r'\s*-\s*(toxicity|efficacy|dosage|other|metabolism)$','',d)
    d=re.sub(r'\b(hydrochloride|sodium|potassium|mesylate|maleate|sulfate|citrate|besylate)\b','',d).strip()
    return d
recs=[]  # (rs, drug, gene, source, level, category)
# ---- ClinVar drug_response
cv=[]
for l in gzip.open(R+'clinvar_pgx.vcf.gz','rt'):
    if l[0]=='#': continue
    c,p,i,ref,alt,_,_,info=l.rstrip('\n').split('\t')[:8]
    d=dict(x.split('=',1) if '=' in x else (x,'1') for x in info.split(';'))
    rs='rs'+d['RS'] if 'RS' in d else f'{c}:{p}:{ref}:{alt}'
    cv.append(dict(rs=rs,chrom=c,pos=int(p),ref=ref,alt=alt,**{k:d.get(k) for k in ['CLNSIG','CLNDN','CLNREVSTAT','MC','CLNVC','ORIGIN','AF_ESP','AF_EXAC','AF_TGP','GENEINFO']}))
    if 'drug_response' in d.get('CLNSIG',''):
        for t in d.get('CLNDN','').split('|'):
            if 'response' in t.lower():
                cat='toxicity' if 'toxicity' in t.lower() else 'response'
                recs.append((rs,norm(t.replace('_response','').replace('_',' ').replace(' response','')),d.get('GENEINFO','').split(':')[0],'ClinVar','CV',cat))
cv=pd.DataFrame(cv); cv.to_csv('clinvar_pgx_records.csv',index=False)
# ---- PharmGKB clinical annotations
ca=pd.read_csv(R+'clinical_annotations.tsv',sep='\t')
ca=ca[ca['Variant/Haplotypes'].str.match(r'^rs\d+$',na=False)]
for _,r in ca.iterrows():
    for dr in re.split(r'[;/]',str(r['Drug(s)'])):
        if dr.strip() and dr!='nan':
            recs.append((r['Variant/Haplotypes'],norm(dr),r.Gene,'PharmGKB_CA',str(r['Level of Evidence']),str(r['Phenotype Category']).lower()))
# ---- PharmGKB variant annotations (significant only)
va=pd.read_csv(R+'var_drug_ann.tsv',sep='\t')
va=va[va['Variant/Haplotypes'].str.match(r'^rs\d+$',na=False)&(va.Significance=='yes')&(va['Is/Is Not associated']=='Associated with')]
for _,r in va.iterrows():
    for dr in re.split(r'[;/]',str(r['Drug(s)'])):
        if dr.strip() and dr!='nan':
            recs.append((r['Variant/Haplotypes'],norm(dr),r.Gene,'PharmGKB_VA','VA',str(r['Phenotype Category']).lower()))
L=pd.DataFrame(recs,columns=['variant','drug','gene','source','level','category'])
L['gene']=L.gene.astype(str).str.split(r'[ ,(]').str[0]
L.to_csv('label_records_raw.csv',index=False)
print('raw records',len(L)); print(L.groupby('source').agg(n=('variant','size'),variants=('variant','nunique'),drugs=('drug','nunique')))
