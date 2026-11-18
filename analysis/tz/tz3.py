import pandas as pd, numpy as np
je=pd.read_pickle('data/out/je_tz.pkl')
je=je[~((je.ts.dt.hour==0)&(je.ts.dt.minute==0)&(je.ts.dt.second==0))]
ex=je[(je.kind=='off')&(je.tzv!=0)].copy()
ex['utc']=ex.ts-pd.to_timedelta(ex.tzv,unit='h')
ex['d']=ex.utc.dt.floor('D')
# daily mode offset as location signal
loc=ex.groupby('d').tzv.agg(lambda s:s.mode().iloc[0])
print('days with local-offset signal',len(loc)); print(loc.value_counts().head(8))
# propagate ±2 days
idx=pd.date_range('2009-01-01','2019-08-10',freq='D')
L=loc.reindex(idx).ffill(limit=2).bfill(limit=2)
rest=je[(je.kind=='none')|((je.kind=='off')&(je.tzv==0))].copy()   # treat as UTC
rest['utc']=rest.ts; rest['d']=rest.utc.dt.floor('D')
rest['off']=rest.d.map(L)
known=rest[rest.off.notna()].copy()
known['local']=known.utc+pd.to_timedelta(known.off,unit='h')
allk=pd.concat([known[['local','off']], ex.assign(local=ex.ts,off=ex.tzv)[['local','off']]])
print('messages with inferred local time',len(allk),'of',len(je))
h=np.bincount(allk.local.dt.hour,minlength=24)/len(allk)
print('local hour %:', ' '.join(f'{i}:{100*x:.1f}' for i,x in enumerate(h)))
print('peak',h.argmax(),'0-4h %.1f'%(100*h[0:5].sum()),'5-6h %.1f'%(100*h[5:7].sum()),'22-23 %.1f'%(100*h[22:].sum()))
allk['day']=(allk.local-pd.Timedelta(hours=4)).dt.floor('D')  # day starts 4am
g=allk.groupby('day').local
first=(g.min()-g.min().dt.floor('D')); last=(g.max()-g.max().dt.floor('D'))
cnt=g.size(); ok=cnt>=10
fh=first[ok].dt.total_seconds()/3600; lh=last[ok].dt.total_seconds()/3600
lh=lh.where(lh>=4,lh+24)
print('days>=10 msgs',ok.sum(),'median first %.2f IQR %.2f-%.2f'%(fh.median(),fh.quantile(.25),fh.quantile(.75)),
      'median last %.2f IQR %.2f-%.2f'%(lh.median(),lh.quantile(.25),lh.quantile(.75)))
yr=pd.DataFrame({'f':fh,'l':lh}); yr['y']=yr.index.year
print(yr.groupby('y').median().round(2).to_string())
# by location offset group
allk['grp']=allk.off.map(lambda o:'Europe(+1/+2)' if o in(1,2) else ('US East/Carib(-4/-5)' if o in(-4,-5) else ('US West/Mtn(-6..-8)' if o in(-6,-7,-8) else 'other')))
for gname,sub in allk.groupby('grp'):
    hh=np.bincount(sub.local.dt.hour,minlength=24)/len(sub)
    print(gname,len(sub),'peak',hh.argmax(),'0-4h %.1f'%(100*hh[0:5].sum()))
