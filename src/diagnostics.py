"""Post-hoc descriptive diagnostics requested after the main analysis (not part of the frozen plan).
Writes out/diagnostics_stickiness.csv and out/diagnostics_by_year.csv."""
import sys, numpy as np, pandas as pd
import analyze as an, fc

OUT = 'results/'
P, meta = an.P, an.meta
d = P[:, 1:] - P[:, :-1]
ok = ~np.isnan(d)
zero = ok & (np.abs(d) < 1e-9)
meta = meta.copy()
meta['pairs'] = ok.sum(1); meta['zeros'] = zero.sum(1)
rows = []
for dim in ('category', 'commodity'):
    g = meta.groupby(dim)[['pairs', 'zeros']].sum()
    g['share_unchanged'] = g.zeros / g.pairs
    for k, r in g.iterrows():
        rows.append(dict(dim=dim, group=k, pairs=int(r.pairs), unchanged=int(r.zeros), share_unchanged=r.share_unchanged))
rows.append(dict(dim='all', group='all', pairs=int(ok.sum()), unchanged=int(zero.sum()), share_unchanged=zero.sum() / ok.sum()))
pd.DataFrame(rows).to_csv(OUT + 'diagnostics_stickiness.csv', index=False)

zm = np.load(OUT + 'fc_A_naive_snaive_scm_ridge_lgbm.npz')
out = []
for h in fc.HS:
    o = list(zm[f'origins|{h}'])
    y, sc, okk = an.triples(h, o)
    yr = np.array([2019 + (3 + t) // 12 for t in o])   # calendar year of the origin month (s=0 is April 2019)
    res = {}
    for m in ('naive', 'lgbm'):
        e = np.abs(zm[f'{m}|{h}'] - y) / sc
        res[m] = an.origin_mean(e, okk)
    for k in np.unique(yr):
        sel = yr == k
        have = ~np.isnan(res['naive'][sel])
        out.append(dict(h=h, origin_year=int(k), origins=int(have.sum()), mase_naive=np.nanmean(res['naive'][sel]),
                        mase_lgbm=np.nanmean(res['lgbm'][sel]), ratio=np.nanmean(res['lgbm'][sel]) / np.nanmean(res['naive'][sel])))
pd.DataFrame(out).to_csv(OUT + 'diagnostics_by_year.csv', index=False)
print(pd.read_csv(OUT + 'diagnostics_by_year.csv').round(3).to_string())
print(pd.read_csv(OUT + 'diagnostics_stickiness.csv').query("dim!='commodity'").round(3).to_string())
