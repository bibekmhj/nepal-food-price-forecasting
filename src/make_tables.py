import sys, numpy as np, pandas as pd
import analyze as an, fc
OUT='results/'
pd.set_option('display.width',250)
P,meta=an.P,an.meta
print('series',P.shape[0],'markets',meta.market.nunique(),'commodities',meta.commodity.nunique(),'fill',round((~np.isnan(P)).mean(),4))
print('obs',int((~np.isnan(P)).sum()))
print(meta.groupby('province').market.nunique().to_string())
print(meta.groupby('category').commodity.nunique().to_string())
# Holm for unseen families
T=pd.read_csv(OUT+'unseen_tests.csv')
for fam,mask in (('A',T.comp.isin(['unseen_vs_seen','unseen_vs_naive'])),('B',T.comp.str.startswith('B_'))):
    idx=T.index[mask]; p=T.loc[idx,'p'].values; o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,k in enumerate(o):
        run=max(run,(m-r)*p[k]); adj[k]=min(1,run)
    T.loc[idx,'p_holm']=adj
T.to_csv(OUT+'unseen_tests_holm.csv',index=False)
print(T[T.comp.isin(['unseen_vs_seen','unseen_vs_naive'])|T.comp.str.startswith('B_')][['mode','h','comp','ratio','lo','hi','p','p_holm']].round(4).to_string())
D=pd.read_csv(OUT+'dm_tests.csv'); print(D[D['sample']=='all_origins'][['h','vs','ratio','ci_lo','ci_hi','p','p_holm']].round(4).to_string())
print(D[D['sample']=='stat_origins'][['h','vs','ratio','ci_lo','ci_hi','p','p_holm']].round(4).to_string())
import json; print(json.load(open(OUT+'fallbacks.json')))
# per seed spread
M=pd.read_csv(OUT+'main_all_origins.csv'); print(M[M.model.str.contains('seed')][['h','model','mase']].round(4).to_string())
# worst/best markets in LOMO
G=pd.read_csv(OUT+'unseen_groups.csv'); g=G[(G.dim=='market')&(G.h==3)].sort_values('ratio_unseen_seen'); print(g[['group','ratio_unseen_seen','ratio_unseen_naive']].round(3).to_string())
