import pandas as pd, numpy as np
je=pd.read_pickle('data/out/je_tz.pkl')
def prof(h):
    c=np.bincount(h,minlength=24)/len(h); return c
utc=je[(je.kind=='off')&(je.tzv==0)].copy()
none=je[je.kind=='none'].copy()
none=none[~((none.ts.dt.hour==0)&(none.ts.dt.minute==0))]   # drop date-only
ex=je[(je.kind=='off')&(je.tzv!=0)].copy()
# local NY for utc
utc['ny']=utc.ts.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
ex['utc']=ex.ts-pd.to_timedelta(ex.tzv,unit='h')
ex['ny']=ex.utc.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
rows={}
rows['UTC-stamp (raw)']=prof(utc.ts.dt.hour)
rows['UTC-stamp -> NY']=prof(utc.ny.dt.hour)
rows['no-tz (raw)']=prof(none.ts.dt.hour)
rows['explicit non-UTC (raw local)']=prof(ex.ts.dt.hour)
rows['explicit non-UTC -> NY']=prof(ex.ny.dt.hour)
df=pd.DataFrame(rows).T
pd.set_option('display.width',250)
print((df*100).round(1).to_string())
for k,v in rows.items():
    n=v; print(k,'peak',int(np.argmax(n)),'night0-5 %.1f'%(100*n[0:6].sum()), 'first>2%%', [h for h in range(4,12) if n[h]>0.02][:1])
print(len(utc),len(none),len(ex))
# cross-correlation shift between none and utc raw
a=rows['no-tz (raw)']; b=rows['UTC-stamp (raw)']
best=max(range(24),key=lambda s: np.corrcoef(a,np.roll(b,-s))[0,1])
print('best shift utc->none',best,[round(np.corrcoef(a,np.roll(b,-s))[0,1],3) for s in range(0,8)])
