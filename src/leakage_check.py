import sys
import numpy as np, fc
P,meta=fc.load_panel('data/wfp_food_prices_npl.csv'); S=P.shape[0]
Fall=fc.Feat(P,meta)
bad=0; tot=0
for t in (35,47,62,70,85):
    V=P.copy(); V[:,t+1:]=np.nan
    Ft=fc.Feat(V,meta)
    rows=np.arange(S); cols=np.full(S,t)
    for h in fc.HS:
        a=Fall.matrix(h,rows,cols); b=Ft.matrix(h,rows,cols)
        same=(np.isnan(a)&np.isnan(b))|np.isclose(a,b,equal_nan=False)
        bad+=(~same).sum(); tot+=same.size
print('feature cells compared',tot,'mismatches between full-data and truncated-data features',bad)
# cleaning view check: flagged values at origin t only use data known by t
flag,nxt=fc.flags_full(P)
t=60; V=fc.view_at(P,t,'B',(flag,nxt))
print('cells flagged and removed at t=60:', int((flag&(nxt<=t)).sum()), 'columns after t all NaN:', bool(np.isnan(V[:,t+1:]).all()))
print('flag total',int(flag.sum()),'share of obs',round(flag.sum()/(~np.isnan(P)).sum(),5))
