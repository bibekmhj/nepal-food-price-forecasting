import sys, json, numpy as np, pandas as pd
import fc
from scipy import stats

P, meta = fc.load_panel('data/wfp_food_prices_npl.csv')
S = P.shape[0]
mt, _ = fc.market_table(meta)
meta['iso_group'] = meta.market.map(mt.iso_group)
FLAG, _ = fc.flags_full(P)
NPR = np.exp


def load(files):
    res = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            res[k] = z[k]
    return res


def triples(h, origins, scenario='A'):
    """Valid mask (n_origins, S) and truth/scale arrays for horizon h."""
    n = len(origins)
    y = np.full((n, S), np.nan); sc = np.full((n, S), np.nan); ok = np.zeros((n, S), bool)
    for oi, t in enumerate(origins):
        y[oi] = P[:, t + h]
        sc[oi] = fc.mase_scale(P, t, h)
        ok[oi] = ~np.isnan(P[:, t]) & ~np.isnan(P[:, t + h]) & ~np.isnan(sc[oi])
        if scenario == 'C':
            ok[oi] &= ~FLAG[:, t + h]
    return y, sc, ok


def get_fc(res, m, h, origins, fallback_to=None):
    allo = res[f'origins|{h}']
    pos = {t: k for k, t in enumerate(allo)}
    idx = [pos[t] for t in origins]
    return res[f'{m}|{h}'][idx]


def loss_tables(res, models, h, origins, scenario='A', with_fallback=None):
    y, sc, ok = triples(h, origins, scenario)
    out = {}
    base = get_fc(res, 'naive', h, origins) if 'naive' in res.keys() or f'naive|{h}' in res else None
    for m in models:
        f = get_fc(res, m, h, origins)
        if with_fallback is not None:
            f = np.where(np.isnan(f), with_fallback, f)
        e = np.abs(f - y)
        out[m] = e
    return out, y, sc, ok


def origin_mean(arr, ok):
    with np.errstate(all='ignore'):
        n = ok.sum(1)
        return np.where(n > 0, np.where(ok, arr, 0).sum(1) / np.where(n > 0, n, 1), np.nan)


def dm_test(d, h):
    """Diebold-Mariano with Newey-West variance (lag h-1) and Harvey-Leybourne-Newbold correction."""
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    lag = h - 1
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(lag + 1)]
    lrv = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))
    lrv = max(lrv, 1e-12)
    dm = dbar / np.sqrt(lrv / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    st = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(st), n - 1))
    return dbar, st, p


def block_boot_ratio(a, b, block=6, B=2000, seed=0):
    """CI for mean(a)/mean(b) over origins with a moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    keep = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.asarray(a)[keep], np.asarray(b)[keep]
    n = len(a)
    nb = int(np.ceil(n / block))
    r = np.empty(B)
    for k in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r[k] = a[idx].mean() / b[idx].mean()
    return np.percentile(r, [2.5, 97.5])


def summarize(res, models, h, origins, scenario='A', nm='naive'):
    es, y, sc, ok = loss_tables(res, models + ([nm] if nm not in models else []), h, origins, scenario)
    rows = {}
    for m in models + ([nm] if nm not in models else []):
        e = es[m]
        scaled = e / sc
        om = origin_mean(scaled, ok)
        with np.errstate(over='ignore'):
            nprerr = np.abs(NPR(get_fc(res, m, h, origins)) - NPR(y))
        rows[m] = dict(mase=np.nanmean(om), mae_log=np.nanmean(e[ok]), mae_npr=np.nanmean(nprerr[ok]),
                       n=int(ok.sum()), origin_mase=om)
    for m in rows:
        rows[m]['mae_ratio_naive'] = rows[m]['mae_log'] / rows[nm]['mae_log']
        rows[m]['mase_ratio_naive'] = rows[m]['mase'] / rows[nm]['mase']
    return rows, es, y, sc, ok


def run_main(res, h, models, origins, scenario='A', label=''):
    rows, es, y, sc, ok = summarize(res, models, h, origins, scenario)
    tab = pd.DataFrame({m: {k: v for k, v in r.items() if k != 'origin_mase'} for m, r in rows.items()}).T
    tab.insert(0, 'h', h)
    return tab, rows, es, y, sc, ok


if __name__ == '__main__':
    pass
