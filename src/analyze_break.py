import sys, numpy as np, pandas as pd
import analyze as an, fc

OUT = 'results/'
zm = np.load(OUT + 'fc_A_naive_snaive_scm_ridge_lgbm.npz')
meta = an.meta
rows = []
for h in fc.HS:
    o = list(zm[f'origins|{h}'])
    y, sc, ok = an.triples(h, o)
    sc_err = {m: np.abs(zm[f'{m}|{h}'] - y) / sc for m in ('naive', 'snaive', 'lgbm', 'ridge')}
    abs_err = {m: np.abs(zm[f'{m}|{h}'] - y) for m in ('naive', 'lgbm')}
    for dim in ('category', 'province', 'iso_group', 'commodity'):
        for g, idx in meta.groupby(dim).groups.items():
            msk = np.zeros(an.S, bool); msk[list(idx)] = True
            okg = ok & msk[None, :]
            r = {'h': h, 'dim': dim, 'group': g, 'n': int(okg.sum()), 'series': int(msk.sum())}
            for m in sc_err:
                r['mase_' + m] = np.nanmean(an.origin_mean(sc_err[m], okg))
            r['ratio_lgbm_naive'] = r['mase_lgbm'] / r['mase_naive']
            r['mae_ratio_lgbm_naive'] = abs_err['lgbm'][okg].mean() / abs_err['naive'][okg].mean()
            rows.append(r)
B = pd.DataFrame(rows)
B.to_csv(OUT + 'breakdown_A.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.float_format', lambda x: f'{x:.3f}')
for dim in ('category', 'province', 'iso_group'):
    print(dim); print(B[B.dim == dim].pivot(index='group', columns='h', values=['ratio_lgbm_naive', 'mae_ratio_lgbm_naive']).to_string())
print('commodity'); print(B[B.dim == 'commodity'].pivot(index='group', columns='h', values=['ratio_lgbm_naive', 'mae_ratio_lgbm_naive']).to_string())
print(meta.groupby('category').commodity.unique().to_string())
print(an.mt if hasattr(an, 'mt') else '')
print(an.mt.groupby('iso_group').apply(lambda d: list(d.index)).to_string())
