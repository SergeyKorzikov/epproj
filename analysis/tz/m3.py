import sys; sys.path.insert(0,'src')
import pandas as pd, numpy as np
from epf.stylometry import _features
je=pd.read_pickle('data/out/je_tz.pkl')
je=je[~((je.ts.dt.hour==0)&(je.ts.dt.minute==0)&(je.ts.dt.second==0))]
je=je[(je.ts>='2009-08-01')&(je.ts<'2019-07-01')]
# local time: explicit offsets else UTC->NY (dominant location)
off=je.tzv.where(je.kind=='off')
utc=je.ts-pd.to_timedelta(off.fillna(0),unit='h')
ny=utc.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
je=je.assign(local=ny, y=je.ts.dt.year)
je['slot']=pd.cut(je.local.dt.hour,[-1,4,11,17,23],labels=['night','morning','afternoon','evening'])
out=[]
for (y,s),g in je.groupby(['y','slot'],observed=True):
    f=_features('\n'.join(g.body.fillna(''))); out.append(dict(y=y,slot=s,n=len(g),posemo=f['posemo'],qw=f['question_words'],q=g.body.str.contains(r'\?').mean()))
D=pd.DataFrame(out)
print(D.pivot(index='y',columns='slot',values='posemo').round(2).to_string())
print(D.pivot(index='y',columns='slot',values='qw').round(2).to_string())
for v in ['posemo','qw']:
    P=D.pivot(index='y',columns='slot',values=v)
    for s in ['morning','afternoon','evening']:
        b=np.polyfit(P.index,P[s],1)[0]; print(v,s,'slope/yr %.3f'%b, 'change 2010->2018 %.0f%%'%(100*(P.loc[2018,s]/P.loc[2010,s]-1)))
# late-night nights: day after a night with msgs at 1-4 local vs not
je['d']=(je.local-pd.Timedelta(hours=5)).dt.floor('D')
late=je[je.local.dt.hour.between(1,4)].d.unique()
nxt=pd.Series(1,index=pd.DatetimeIndex(late)+pd.Timedelta(days=1))
je['after_late']=je.d.isin(nxt.index)
day=je[je.local.dt.hour.between(8,17)]
for k,g in day.groupby('after_late'):
    f=_features('\n'.join(g.body.fillna(''))); print('after_late',k,len(g),'posemo %.3f qw %.3f'%(f['posemo'],f['question_words']), 'q %.3f'%g.body.str.contains(r'\?').mean())
