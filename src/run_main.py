import sys, time, json, numpy as np, pandas as pd
import fc
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

scenario = sys.argv[1]            # A, B or C
models = sys.argv[2].split(',')   # subset of naive,snaive,scm,ridge,lgbm
SEEDS = (11, 22, 33)
P, meta = fc.load_panel('data/wfp_food_prices_npl.csv')
S = P.shape[0]
fl = fc.flags_full(P) if scenario in ('B', 'C') else None
origins = list(range(fc.FIRST_ORIGIN, fc.NT - 1))
featcache = {}


def get_feat(t):
    if scenario == 'A':
        if 'A' not in featcache:
            featcache['A'] = fc.Feat(P, meta)
        return featcache['A'], P
    V = fc.view_at(P, t, scenario, fl)
    return fc.Feat(V, meta), V


def make_ridge():
    num = [i for i in range(len(fc.Feat.names)) if i not in (15, 16)]
    ct = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), [15, 16])])
    return Pipeline([('ct', ct), ('rg', Ridge(alpha=1.0))])


def train_rows(F, V, t, h):
    s = np.arange(3, t - h + 1)
    ii, ss = np.meshgrid(np.arange(S), s, indexing='ij')
    ii, ss = ii.ravel(), ss.ravel()
    ok = ~np.isnan(V[ii, ss]) & ~np.isnan(V[ii, ss + h])
    ii, ss = ii[ok], ss[ok]
    X = F.matrix(h, ii, ss)
    y = V[ii, ss + h] - V[ii, ss]
    return X, y


res = {}
t_start = time.time()
for h in fc.HS:
    nor = [t for t in origins if t + h <= fc.NT - 1]
    out = {m: np.full((len(nor), S), np.nan) for m in models}
    if 'lgbm' in models:
        for sd in SEEDS:
            out[f'lgbm_s{sd}'] = np.full((len(nor), S), np.nan)
    fallback = {'snaive': np.zeros((len(nor), S), bool), 'scm': np.zeros((len(nor), S), bool)}
    cur = {}
    for oi, t in enumerate(nor):
        F, V = get_feat(t)
        base = fc.naive_level(V, t)
        if 'naive' in models:
            out['naive'][oi] = base
        if 'snaive' in models:
            f, miss = fc.seasonal_naive(V, t, h)
            out['snaive'][oi] = f; fallback['snaive'][oi] = miss
        if 'scm' in models:
            f, miss = fc.scm_forecast(F, t, h)
            out['scm'][oi] = f; fallback['scm'][oi] = miss
        if ('ridge' in models or 'lgbm' in models):
            if (t - fc.FIRST_ORIGIN) % 3 == 0:
                X, y = train_rows(F, V, t, h)
                cur = {}
                if 'ridge' in models:
                    r = make_ridge(); r.fit(X, y); cur['ridge'] = r
                if 'lgbm' in models:
                    for sd in SEEDS:
                        g = lgb.LGBMRegressor(objective='l1', n_estimators=200, learning_rate=0.05, num_leaves=15,
                                              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                              random_state=sd, n_jobs=2, verbose=-1)
                        g.fit(X, y, categorical_feature=[15, 16]); cur[f'lgbm_s{sd}'] = g
            rows = np.arange(S); cols = np.full(S, t)
            Xt = F.matrix(h, rows, cols)
            if 'ridge' in models:
                out['ridge'][oi] = base + cur['ridge'].predict(Xt)
            if 'lgbm' in models:
                ps = []
                for sd in SEEDS:
                    p = base + cur[f'lgbm_s{sd}'].predict(Xt)
                    out[f'lgbm_s{sd}'][oi] = p; ps.append(p)
                out['lgbm'][oi] = np.mean(ps, axis=0)
    for k, v in out.items():
        res[f'{k}|{h}'] = v
    for k, v in fallback.items():
        if k in models:
            res[f'fb_{k}|{h}'] = v
    res[f'origins|{h}'] = np.array(nor)
    print('h', h, 'done', round(time.time() - t_start), 's', flush=True)
np.savez_compressed(f'results/fc_{scenario}_{"_".join(models)}.npz', **res)
meta.to_csv('results/meta.csv', index=False)
