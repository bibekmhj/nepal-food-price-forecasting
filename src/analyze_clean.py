import sys, numpy as np, pandas as pd
import analyze as an, fc

OUT = 'results/'
models = ['naive', 'snaive', 'scm', 'ridge', 'lgbm']
rows = []
flag_share = an.FLAG.sum() / (~np.isnan(an.P)).sum()
print('share of observations flagged by cleaning rule', round(flag_share, 5), int(an.FLAG.sum()))
for sc_name in ('A', 'B', 'C'):
    z = np.load(OUT + f'fc_{sc_name}_naive_snaive_scm_ridge_lgbm.npz')
    for h in fc.HS:
        o = list(z[f'origins|{h}'])
        R = {k: z[k] for k in z.files}
        scen_eval = 'C' if sc_name == 'C' else 'A'
        tab, rr, es, y, sc, ok = an.run_main(R, h, models, o, scenario=scen_eval)
        rank = tab.mase.rank().astype(int)
        for m in models:
            rows.append(dict(scenario=sc_name, h=h, model=m, mase=tab.mase[m], mae_log=tab.mae_log[m], mae_npr=tab.mae_npr[m],
                             ratio_naive=tab.mase_ratio_naive[m], rank=rank[m], n=int(tab.n[m])))
Cn = pd.DataFrame(rows)
Cn.to_csv(OUT + 'cleaning_sensitivity.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(Cn.pivot_table(index=['h', 'model'], columns='scenario', values=['mase', 'mae_log', 'ratio_naive', 'rank']).to_string())
