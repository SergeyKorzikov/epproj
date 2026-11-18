import pandas as pd, re, numpy as np, sys
je=pd.read_parquet(sys.argv[1])
je=je[je.ts.notna()&(je.ts>='2009-01-01')&(je.ts<'2019-08-11')].copy()
def kind(s):
    s=s or ''
    m=re.search(r'([+-])(\d{2}):?(\d{2})\b',s)
    if m and re.search(r'\d{1,2}:\d{2}',s):
        off=(1 if m.group(1)=='+' else -1)*(int(m.group(2))+int(m.group(3))/60)
        return 'off',off
    m=re.search(r'\b(EDT|EST|PDT|PST|CDT|CST|MDT|MST|GMT|UTC|CET|CEST|BST|AST)\b',s)
    if m: return 'name',m.group(1)
    return 'none',None
k=je.ts_raw.map(kind)
je['kind']=[a for a,b in k]; je['tzv']=[b for a,b in k]
print(je.kind.value_counts(normalize=True).round(3))
print(je[je.kind=='off'].tzv.value_counts().head(10))
print(je[je.kind=='name'].tzv.value_counts().head(10))
print(pd.crosstab(je.ts.dt.year, je.kind, normalize='index').round(2))
je.to_pickle('data/out/je_tz.pkl')
