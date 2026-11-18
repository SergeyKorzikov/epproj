import sys; sys.path.insert(0,'src')
import pandas as pd, numpy as np, ruptures as rpt
from epf.stylometry import _features, message_level
je=pd.read_pickle('data/out/je_tz.pkl')
je=je[(je.ts>='2009-08-01')&(je.ts<'2019-07-01')].copy()   # after release, before arrest
je['m']=je.ts.dt.to_period('M')
ml=message_level(je)
rows=[]
for m,g in je.groupby('m'):
    if len(g)<300: continue
    f=_features('\n'.join(g.body.fillna('')))
    mm=ml.loc[g.index]
    rows.append(dict(m=str(m),n=len(g),posemo=f['posemo'],qw=f['question_words'],negemo=f['negemo'],i=f['i'],you=f['you'],
       q_share=mm.has_question.mean(),med_len=mm.n_words.median(),body=f['body']))
M=pd.DataFrame(rows).set_index('m')
M.to_csv('data/out/monthly.csv')
print(len(M),'months'); print(M[['posemo','qw','q_share','med_len']].describe().round(2))
cols=['posemo','qw','q_share','med_len']
Z=(M[cols]-M[cols].mean())/M[cols].std()
idx=Z.mean(axis=1)  # engagement index
M['idx']=idx
for c in cols+['idx']:
    x=M[c].values.reshape(-1,1)
    algo=rpt.Pelt(model='l2',min_size=6).fit((x-x.mean())/x.std())
    bk=algo.predict(pen=8)
    print(c,'breaks:',[M.index[b-1] for b in bk[:-1]], 'segment means:',[round(float(M[c].iloc[s:e].mean()),2) for s,e in zip([0]+bk[:-1],bk)])
# yearly trend slope and residual roughness
t=np.arange(len(M))
for c in cols:
    b=np.polyfit(t,M[c],1); res=M[c]-np.polyval(b,t)
    print(c,'slope/yr %.3f'%(b[0]*12), 'R2 %.2f'%(1-res.var()/M[c].var()))
