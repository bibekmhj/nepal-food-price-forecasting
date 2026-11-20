import sys, numpy as np, pandas as pd
import analyze as an, fc

OUT = 'results/'
meta, mt = an.meta, an.mt
zm = np.load(OUT + 'fc_A_naive_snaive_scm_ridge_lgbm.npz')
pd.set_option('display.width', 220); pd.set_option('display.float_format', lambda x: f'{x:.4f}')
allrows, tests, byg = [], [], []
for mode in ('LOMO', 'LOPO'):
    try:
        z = np.load(OUT + f'unseen_{mode}.npz')
    except FileNotFoundError:
        continue
    for h in fc.HS:
        o = list(z[f'origins|{h}'])
        y, sc, ok = an.triples(h, o)
        # ---- variant A
        e = {k: np.abs(z[f'A_{k}|{h}'] - y) for k in ('unseen', 'seen', 'naive')}
        # common valid: all finite
        okA = ok & np.isfinite(e['unseen']) & np.isfinite(e['seen']) & np.isfinite(e['naive'])
        om = {k: an.origin_mean(e[k] / sc, okA) for k in e}
        r = dict(mode=mode, h=h, variant='A', n=int(okA.sum()), mase_unseen=np.nanmean(om['unseen']), mase_seen=np.nanmean(om['seen']),
                 mase_naive=np.nanmean(om['naive']))
        r['ratio_unseen_seen'] = r['mase_unseen'] / r['mase_seen']
        r['ratio_unseen_naive'] = r['mase_unseen'] / r['mase_naive']
        r['ratio_seen_naive'] = r['mase_seen'] / r['mase_naive']
        for a, b, nm in (('unseen', 'seen', 'unseen_vs_seen'), ('unseen', 'naive', 'unseen_vs_naive'), ('seen', 'naive', 'seen_vs_naive')):
            d = om[a] - om[b]
            dbar, st, p = an.dm_test(d, h)
            lo, hi = an.block_boot_ratio(om[a], om[b])
            tests.append(dict(mode=mode, h=h, comp=nm, diff=dbar, dm=st, p=p, ratio=np.nanmean(om[a]) / np.nanmean(om[b]), lo=lo, hi=hi))
        allrows.append(r)
        # by isolation group and by province (ratio unseen/seen and unseen/naive)
        for dim in ('iso_group', 'province', 'category'):
            for g, idx in meta.groupby(dim).groups.items():
                msk = np.zeros(an.S, bool); msk[list(idx)] = True
                okg = okA & msk[None, :]
                m_ = {k: np.nanmean(an.origin_mean(e[k] / sc, okg)) for k in e}
                byg.append(dict(mode=mode, h=h, dim=dim, group=g, n=int(okg.sum()), mase_unseen=m_['unseen'], mase_seen=m_['seen'],
                                mase_naive=m_['naive'], ratio_unseen_seen=m_['unseen'] / m_['seen'], ratio_unseen_naive=m_['unseen'] / m_['naive']))
        # ---- variant B (cold start): level MAE in log points
        eb = {k: np.abs(z[f'B_{k}|{h}'] - y) for k in ('lgbmB', 'B0', 'B1')}
        eb['naive_own'] = e['naive']
        okB = ok & np.isfinite(eb['lgbmB']) & np.isfinite(eb['B0']) & np.isfinite(eb['B1']) & np.isfinite(e['naive'])
        rb = dict(mode=mode, h=h, variant='B', n=int(okB.sum()))
        for k in eb:
            rb['mae_' + k] = eb[k][okB].mean()
        rb['ratio_lgbmB_B0'] = rb['mae_lgbmB'] / rb['mae_B0']
        rb['ratio_lgbmB_B1'] = rb['mae_lgbmB'] / rb['mae_B1']
        rb['ratio_lgbmB_naiveown'] = rb['mae_lgbmB'] / rb['mae_naive_own']
        omb = {k: an.origin_mean(eb[k], okB) for k in ('lgbmB', 'B0', 'B1', 'naive_own')}
        for b_, nm in (('B0', 'B_lgbm_vs_B0'), ('B1', 'B_lgbm_vs_B1'), ('naive_own', 'B_lgbm_vs_own_naive')):
            d = omb['lgbmB'] - omb[b_]
            dbar, st, p = an.dm_test(d, h)
            lo, hi = an.block_boot_ratio(omb['lgbmB'], omb[b_])
            tests.append(dict(mode=mode, h=h, comp=nm, diff=dbar, dm=st, p=p, ratio=np.nanmean(omb['lgbmB']) / np.nanmean(omb[b_]), lo=lo, hi=hi))
        allrows.append(rb)
        for dim in ('iso_group', 'province'):
            for g, idx in meta.groupby(dim).groups.items():
                msk = np.zeros(an.S, bool); msk[list(idx)] = True
                okg = okB & msk[None, :]
                byg.append(dict(mode=mode, h=h, dim=dim, group=g, variant='B', n=int(okg.sum()),
                                mae_lgbmB=eb['lgbmB'][okg].mean(), mae_B0=eb['B0'][okg].mean(), mae_B1=eb['B1'][okg].mean(),
                                mae_naive_own=eb['naive_own'][okg].mean()))
        # per market (variant A) for LOMO
        if mode == 'LOMO':
            for mk in sorted(mt.index):
                msk = (meta.market == mk).values
                okg = okA & msk[None, :]
                m_ = {k: np.nanmean(an.origin_mean(e[k] / sc, okg)) for k in e}
                byg.append(dict(mode=mode, h=h, dim='market', group=mk, n=int(okg.sum()), mase_unseen=m_['unseen'], mase_seen=m_['seen'],
                                mase_naive=m_['naive'], ratio_unseen_seen=m_['unseen'] / m_['seen'], ratio_unseen_naive=m_['unseen'] / m_['naive']))
R = pd.DataFrame(allrows); Tn = pd.DataFrame(tests); G = pd.DataFrame(byg)
R.to_csv(OUT + 'unseen_summary.csv', index=False); Tn.to_csv(OUT + 'unseen_tests.csv', index=False); G.to_csv(OUT + 'unseen_groups.csv', index=False)
print(R.to_string()); print(Tn.to_string())
g = G[(G.dim.isin(['iso_group', 'province'])) & (G.get('variant').isna())]
print(g.pivot_table(index=['mode', 'dim', 'group'], columns='h', values=['ratio_unseen_seen', 'ratio_unseen_naive']).to_string())
