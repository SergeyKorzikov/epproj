import numpy as np, pandas as pd
health=pd.PeriodIndex(['2013-10','2014-01','2014-07','2015-04','2016-07','2016-09','2017-08'],freq='M')
breaks=pd.PeriodIndex(['2012-11','2011-03','2013-09','2014-07','2016-08','2011-03','2014-07','2011-03','2017-06'],freq='M').unique()
months=pd.period_range('2009-08','2019-06',freq='M')
def score(b): return np.mean([min(abs((x-h).n) for h in health) for x in b])
obs=score(breaks); rng=np.random.default_rng(0)
sims=[score(pd.PeriodIndex(rng.choice(months,len(breaks),replace=False),freq='M')) for _ in range(20000)]
print('breaks',list(map(str,breaks)),'mean dist to nearest health record (months) %.2f'%obs,'p=%.3f'%np.mean(np.array(sims)<=obs))
within=[str(x) for x in breaks if min(abs((x-h).n) for h in health)<=2]; print('within 2 months:',within)
