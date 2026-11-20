import sys, json, numpy as np, pandas as pd
import analyze as an
import fc

OUT = 'results/'
res = an.load([OUT + 'fc_A_naive_snaive_scm_ridge_lgbm.npz', OUT + 'fc_A_stat.npz'])
P, meta, S = an.P, an.meta, an.S

# --- fallbacks for statistical models: failed or NaN -> naive
stat_orig = {h: list(res[f'origins|{h}'][(res[f'origins|{h}'] - fc.FIRST_ORIGIN) % 3 == 0]) for h in fc.HS}
stat_models = ['ets', 'sarima']
for h in fc.HS:
    nor = list(range(fc.FIRST_ORIGIN, fc.NT - h, 1))
    allo = res[f'origins|{h}']
    # the statistical file shares the key origins|h; recover its origins from the stat file
zs = np.load(OUT + 'fc_A_stat.npz')
zm = np.load(OUT + 'fc_A_naive_snaive_scm_ridge_lgbm.npz')

models_all = ['naive', 'snaive', 'scm', 'ridge', 'lgbm']
rows_main, rows_stat = [], []
fb_report = {}
tests = []
for h in fc.HS:
    o_all = list(zm[f'origins|{h}'])
    o_stat = list(zs[f'origins|{h}'])
    # main table (all origins)
    R = {k: zm[k] for k in zm.files}
    tab, rows, es, y, sc, ok = an.run_main(R, h, models_all, o_all)
    tab['origins'] = len(o_all)
    rows_main.append(tab.assign(model=tab.index))
    # per seed
    for sd in (11, 22, 33):
        f = zm[f'lgbm_s{sd}|{h}']
        e = np.abs(f - y)
        om = an.origin_mean(e / sc, ok)
        rows_main.append(pd.DataFrame({'h': h, 'mase': np.nanmean(om), 'model': f'lgbm_seed{sd}'}, index=[f'lgbm_seed{sd}']))
    # stat origin table
    Rs = {}
    pos = {t: k for k, t in enumerate(o_all)}
    for m in models_all + ['lgbm_s11', 'lgbm_s22', 'lgbm_s33']:
        Rs[f'{m}|{h}'] = zm[f'{m}|{h}'][[pos[t] for t in o_stat]]
    nv = Rs[f'naive|{h}']
    for m in stat_models:
        f = zs[f'{m}|{h}'].copy()
        fb = zs[f'fb_{m}|{h}'] | ~np.isfinite(f) | (np.abs(f - nv) > 1.0)   # deviation D3: divergent forecasts replaced by naive
        f = np.where(fb, nv, f)
        Rs[f'{m}|{h}'] = f
        Rs[f'fb_{m}|{h}'] = fb
    Rs[f'origins|{h}'] = np.array(o_stat)
    y2, sc2, ok2 = an.triples(h, o_stat)
    for m in stat_models:
        fb_report[f'{m}|{h}'] = float(Rs[f'fb_{m}|{h}'][ok2].mean())
    mods = models_all + stat_models
    tab2, rows2, es2, y2, sc2, ok2 = an.run_main(Rs, h, mods, o_stat)
    tab2['origins'] = len(o_stat)
    rows_stat.append(tab2.assign(model=tab2.index))
    # DM tests: lgbm vs each other, on stat origins for all (common sample)
    for m in mods:
        if m == 'lgbm':
            continue
        d = rows2['lgbm']['origin_mase'] - rows2[m]['origin_mase']
        dbar, st, p = an.dm_test(d, h)
        lo, hi = an.block_boot_ratio(rows2['lgbm']['origin_mase'], rows2[m]['origin_mase'], block=6, B=2000)
        tests.append(dict(h=h, sample='stat_origins', vs=m, diff_mase=dbar, dm_stat=st, p=p,
                          ratio=rows2['lgbm']['mase'] / rows2[m]['mase'], ci_lo=lo, ci_hi=hi, n_origins=len(o_stat)))
    # DM tests on all origins vs the non stat models
    for m in models_all:
        if m == 'lgbm':
            continue
        d = rows['lgbm']['origin_mase'] - rows[m]['origin_mase']
        dbar, st, p = an.dm_test(d, h)
        lo, hi = an.block_boot_ratio(rows['lgbm']['origin_mase'], rows[m]['origin_mase'], block=6, B=2000)
        tests.append(dict(h=h, sample='all_origins', vs=m, diff_mase=dbar, dm_stat=st, p=p,
                          ratio=rows['lgbm']['mase'] / rows[m]['mase'], ci_lo=lo, ci_hi=hi, n_origins=len(o_all)))
    # fallback shares of baselines
    for k in ('snaive', 'scm'):
        fb_report[f'{k}|{h}'] = float(zm[f'fb_{k}|{h}'][ok].mean())
    fb_report[f'n_triples|{h}'] = int(ok.sum()); fb_report[f'n_triples_stat|{h}'] = int(ok2.sum())

main = pd.concat(rows_main)
stat = pd.concat(rows_stat)
T = pd.DataFrame(tests)
# Holm correction within sample over all comparisons
for samp in T['sample'].unique():
    idx = T.index[T['sample'] == samp]
    p = T.loc[idx, 'p'].values
    order = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for rank, k in enumerate(order):
        run = max(run, (m - rank) * p[k]); adj[k] = min(1, run)
    T.loc[idx, 'p_holm'] = adj
main.to_csv(OUT + 'main_all_origins.csv', index=False)
stat.to_csv(OUT + 'main_stat_origins.csv', index=False)
T.to_csv(OUT + 'dm_tests.csv', index=False)
json.dump(fb_report, open(OUT + 'fallbacks.json', 'w'), indent=1)
pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print('ALL ORIGINS'); print(main.drop(columns=['model']).to_string())
print('STAT ORIGINS'); print(stat.drop(columns=['model']).to_string())
print(T.to_string())
print(fb_report)
