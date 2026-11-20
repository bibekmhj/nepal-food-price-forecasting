import pandas as pd, numpy as np, sys
df=pd.read_csv(sys.argv[1],parse_dates=['date'])
df['ym']=df.date.dt.to_period('M')
print(df.shape, df.date.min(), df.date.max())
print(df.pricetype.unique(), df.priceflag.unique(), df.currency.unique())
d=df[df.ym>='2019-04']
# duplicates
dup=d.duplicated(['market','commodity','ym'],keep=False).sum()
print('dups in window',dup)
months=pd.period_range('2019-04','2026-09',freq='M'); print(len(months))
cov=d.groupby('market').ym.nunique().sort_values(); print(cov.to_string())
print(d.groupby('commodity').agg(mk=('market','nunique'),mo=('ym','nunique'),n=('price','size'),first=('ym','min')).sort_values('n').to_string())
print(d.groupby(['commodity'])['unit'].unique().to_string())
print(d.groupby('market')[['admin1','admin2','latitude','longitude']].first().to_string())
print((d.price<=0).sum(), d.price.isna().sum())
