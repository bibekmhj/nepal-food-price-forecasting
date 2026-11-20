import sys, time, numpy as np, pandas as pd
import fc
import lightgbm as lgb

mode = sys.argv[1]          # LOMO or LOPO
P, meta = fc.load_panel('data/wfp_food_prices_npl.csv')
S, T = P.shape
mt, D = fc.market_table(meta)
markets = list(mt.index)
mk_idx = np.array([markets.index(m) for m in meta.market])
units = markets if mode == 'LOMO' else sorted(mt.province.unique())
SEED = 11
REFIT = 6


def newmodel():
    return lgb.LGBMRegressor(objective='l1', n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=2, verbose=-1)


def train_rows_A(F, V, t, h, series):
    s = np.arange(3, t - h + 1)
    ii, ss = np.meshgrid(series, s, indexing='ij')
    ii, ss = ii.ravel(), ss.ravel()
    ok = ~np.isnan(V[ii, ss]) & ~np.isnan(V[ii, ss + h])
    ii, ss = ii[ok], ss[ok]
    return F.matrix(h, ii, ss), V[ii, ss + h] - V[ii, ss]


# ---- variant B reference features
def ref_feats(src_markets_mask, targets):
    """For target series indices, mean log price of the same commodity over source markets (excluding own market),
    and mean over the 3 nearest source markets observed at each time. Returns ref, nn3 (len(targets), T)."""
    ref = np.full((len(targets), T), np.nan); nn3 = np.full((len(targets), T), np.nan)
    comm = meta.commodity.values
    by_c = {c: np.where(comm == c)[0] for c in np.unique(comm)}
    for a, i in enumerate(targets):
        ids = by_c[comm[i]]
        ids = ids[(mk_idx[ids] != mk_idx[i]) & src_markets_mask[mk_idx[ids]]]
        if len(ids) == 0:
            continue
        X = P[ids]
        n = (~np.isnan(X)).sum(0)
        with np.errstate(all='ignore'):
            r = np.nansum(X, 0) / np.where(n > 0, n, np.nan)
        ref[a] = np.where(n >= 2, r, np.nan)
        order = ids[np.argsort(D[mk_idx[i], mk_idx[ids]])]
        acc = np.zeros(T); cnt = np.zeros(T)
        for j in order:
            ok = (~np.isnan(P[j])) & (cnt < 3)
            acc[ok] += P[j][ok]; cnt[ok] += 1
        nn3[a] = np.where(cnt >= 1, acc / np.where(cnt > 0, cnt, np.nan), np.nan)
    return ref, nn3


def featsB(ref, nn3, targets, rows_a, cols):
    mon = (cols + 3) % 12
    r = ref[rows_a, cols]; n3 = nn3[rows_a, cols]
    r3 = ref[rows_a, np.maximum(cols - 3, 0)]; n33 = nn3[rows_a, np.maximum(cols - 3, 0)]
    i = targets[rows_a]
    return np.column_stack([r, n3 - r, r - r3, n3 - n33, meta.lat.values[i], meta.lon.values[i],
                            pd.Categorical(meta.commodity).codes[i].astype(float), np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)])


t0 = time.time()
outA = {h: {k: np.full((T - 1 - h - fc.FIRST_ORIGIN + 1, S), np.nan) for k in ('unseen', 'seen', 'naive')} for h in fc.HS}
outB = {h: {k: np.full((T - 1 - h - fc.FIRST_ORIGIN + 1, S), np.nan) for k in ('lgbmB', 'B0', 'B1')} for h in fc.HS}
F_all = fc.Feat(P, meta)
scm_cache = F_all.scm
# seen comparator models (all markets), same schedule
seen_models = {}
allser = np.arange(S)
for h in fc.HS:
    nor = list(range(fc.FIRST_ORIGIN, T - h))
    for t in nor:
        if (t - fc.FIRST_ORIGIN) % REFIT == 0:
            X, y = train_rows_A(F_all, P, t, h, allser)
            g = newmodel(); g.fit(X, y, categorical_feature=[15, 16]); seen_models[(h, t)] = g
print('seen done', round(time.time() - t0), flush=True)

for u in units:
    if mode == 'LOMO':
        held_m = np.array([m == u for m in markets])
    else:
        held_m = np.array([mt.province[m] == u for m in markets])
    held_s = np.where(held_m[mk_idx])[0]
    kept_s = np.where(~held_m[mk_idx])[0]
    use = ~held_m[mk_idx]
    FU = fc.Feat(P, meta, use=use, scm_cache=scm_cache)
    # variant B reference features
    src = ~held_m
    ref_k, nn_k = ref_feats(src, kept_s)       # kept targets use kept sources only (self excluded inside)
    ref_h, nn_h = ref_feats(src, held_s)
    for h in fc.HS:
        nor = list(range(fc.FIRST_ORIGIN, T - h))
        cur = None; curB = None
        for oi, t in enumerate(nor):
            if (t - fc.FIRST_ORIGIN) % REFIT == 0:
                X, y = train_rows_A(FU, P, t, h, kept_s)
                cur = newmodel(); cur.fit(X, y, categorical_feature=[15, 16])
                # variant B training rows
                s_ = np.arange(3, t - h + 1)
                aa, ss = np.meshgrid(np.arange(len(kept_s)), s_, indexing='ij'); aa, ss = aa.ravel(), ss.ravel()
                ii = kept_s[aa]
                ok = ~np.isnan(P[ii, ss + h]) & ~np.isnan(ref_k[aa, ss])
                aa, ss, ii = aa[ok], ss[ok], ii[ok]
                XB = featsB(ref_k, nn_k, kept_s, aa, ss); yB = P[ii, ss + h] - ref_k[aa, ss]
                curB = newmodel(); curB.fit(XB, yB, categorical_feature=[6])
            base = fc.naive_level(P, t)
            rows = held_s; cols = np.full(len(rows), t)
            outA[h]['naive'][oi, rows] = base[rows]
            outA[h]['unseen'][oi, rows] = base[rows] + cur.predict(FU.matrix(h, rows, cols))
            outA[h]['seen'][oi, rows] = base[rows] + seen_models[(h, t - (t - fc.FIRST_ORIGIN) % REFIT)].predict(F_all.matrix(h, rows, cols))
            a = np.arange(len(held_s)); cc = np.full(len(held_s), t)
            XBh = featsB(ref_h, nn_h, held_s, a, cc)
            r = ref_h[a, cc]; n3 = nn_h[a, cc]
            outB[h]['B0'][oi, held_s] = r
            outB[h]['B1'][oi, held_s] = np.where(np.isnan(n3), r, n3)
            outB[h]['lgbmB'][oi, held_s] = r + curB.predict(XBh)
    print(mode, u, 'done', round(time.time() - t0), 's', flush=True)

res = {}
for h in fc.HS:
    for k, v in outA[h].items():
        res[f'A_{k}|{h}'] = v
    for k, v in outB[h].items():
        res[f'B_{k}|{h}'] = v
    res[f'origins|{h}'] = np.arange(fc.FIRST_ORIGIN, T - h)
np.savez_compressed(f'results/unseen_{mode}.npz', **res)
