"""Forecasting library for the Nepal WFP retail price study. Follows analysis_plan_v1.md."""
import numpy as np, pandas as pd

T0 = pd.Period('2019-04', 'M')
NT = 90
HS = (1, 3, 6)
FIRST_ORIGIN = 35


def load_panel(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df['ym'] = df.date.dt.to_period('M')
    df = df[(df.ym >= T0) & (df.ym <= pd.Period('2026-09', 'M'))].copy()
    df['t'] = (df.ym - T0).apply(lambda x: x.n)
    assert not df.duplicated(['market', 'commodity', 't']).any()
    cnt = df.groupby(['market', 'commodity']).t.nunique()
    keep = cnt[cnt >= 60].index
    df = df.set_index(['market', 'commodity']).loc[keep].reset_index()
    keys = sorted(keep)
    idx = {k: i for i, k in enumerate(keys)}
    P = np.full((len(keys), NT), np.nan)
    for (m, c), t, p in zip(zip(df.market, df.commodity), df.t, df.price):
        P[idx[(m, c)], t] = np.log(p)
    meta = pd.DataFrame(keys, columns=['market', 'commodity'])
    mk = df.groupby('market')[['admin1', 'latitude', 'longitude']].first()
    cm = df.groupby('commodity').category.first()
    meta['province'] = meta.market.map(mk.admin1)
    meta['lat'] = meta.market.map(mk.latitude)
    meta['lon'] = meta.market.map(mk.longitude)
    meta['category'] = meta.commodity.map(cm)
    return P, meta


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def market_table(meta):
    m = meta.groupby('market')[['province', 'lat', 'lon']].first()
    d = haversine(m.lat.values[:, None], m.lon.values[:, None], m.lat.values[None, :], m.lon.values[None, :])
    np.fill_diagonal(d, np.inf)
    m['isolation'] = np.sort(d, axis=1)[:, :3].mean(axis=1)
    q = m.isolation.quantile([1 / 3, 2 / 3]).values
    m['iso_group'] = np.where(m.isolation <= q[0], 'low', np.where(m.isolation <= q[1], 'mid', 'high'))
    return m, d


# ---------------------------------------------------------------- cleaning
def flags_full(P, thr=np.log(1.5), ret=np.log(1.15)):
    """Flag value at s if |change vs previous observed| > thr and next observed returns within ret of the value before."""
    S, T = P.shape
    flag = np.zeros((S, T), bool)
    nxt = np.full((S, T), 10 ** 6)
    for i in range(S):
        obs = np.where(~np.isnan(P[i]))[0]
        for j in range(1, len(obs) - 1):
            s, sp, sn = obs[j], obs[j - 1], obs[j + 1]
            nxt[i, s] = sn
            if abs(P[i, s] - P[i, sp]) > thr and abs(P[i, sn] - P[i, sp]) < ret:
                flag[i, s] = True
        if len(obs) > 0:
            nxt[i, obs[-1]] = 10 ** 6
    return flag, nxt


def view_at(P, t, scenario, fl=None):
    """Data as known at origin t (columns after t are NaN)."""
    V = P.copy()
    V[:, t + 1:] = np.nan
    if scenario in ('B', 'C'):
        flag, nxt = fl
        m = flag & (nxt <= t)
        V[m] = np.nan
    return V


# ---------------------------------------------------------------- features
def scm_all(P, h):
    """Seasonal change mean feature for every (series, origin s): mean of P[u+h]-P[u], u = s-12k, u+h <= s."""
    S, T = P.shape
    out = np.full((S, T), np.nan)
    for s in range(T):
        acc = np.zeros(S); cnt = np.zeros(S)
        k = 1
        while s - 12 * k >= 0:
            u = s - 12 * k
            if u + h <= s:
                d = P[:, u + h] - P[:, u]
                ok = ~np.isnan(d)
                acc[ok] += d[ok]; cnt[ok] += 1
            k += 1
        good = cnt >= 2
        out[good, s] = acc[good] / cnt[good]
    return out


def lagdiff(P, k):
    out = np.full_like(P, np.nan)
    out[:, k:] = P[:, k:] - P[:, :-k]
    return out


def roll_mean_rel(P, w):
    S, T = P.shape
    out = np.full((S, T), np.nan)
    for s in range(w - 1, T):
        seg = P[:, s - w + 1:s + 1]
        n = (~np.isnan(seg)).sum(1)
        mu = np.nanmean(np.where(np.isnan(seg), np.nan, seg), axis=1) if False else None
        with np.errstate(all='ignore'):
            mu = np.nansum(seg, axis=1) / np.where(n > 0, n, np.nan)
        out[:, s] = np.where(n >= 2, mu - P[:, s], np.nan)
    return out


def roll_sd_changes(P, w=6):
    d = lagdiff(P, 1)
    S, T = P.shape
    out = np.full((S, T), np.nan)
    for s in range(w, T):
        seg = d[:, s - w + 1:s + 1]
        n = (~np.isnan(seg)).sum(1)
        with np.errstate(all='ignore'):
            mu = np.nansum(seg, 1) / np.where(n > 0, n, np.nan)
            var = np.nansum((seg - mu[:, None]) ** 2, 1) / np.where(n > 1, n - 1, np.nan)
        out[:, s] = np.sqrt(var)
    return out


def cross_change(P, meta_comm, use, k):
    """Mean k-month log change of same commodity over other series in `use`, excluding self."""
    d = lagdiff(P, k)
    S, T = P.shape
    out = np.full((S, T), np.nan)
    comm = meta_comm
    for c in np.unique(comm):
        ids = np.where(comm == c)[0]
        dd = d[ids]
        u = use[ids][:, None]
        val = np.where(np.isnan(dd) | ~u, 0.0, dd)
        n = (~np.isnan(dd) & u).sum(0)
        tot = val.sum(0)
        for a, i in enumerate(ids):
            own_ok = (~np.isnan(dd[a])) & use[i]
            nn = n - own_ok
            tt = tot - np.where(own_ok, dd[a], 0.0)
            with np.errstate(all='ignore'):
                out[i] = np.where(nn >= 2, tt / np.where(nn > 0, nn, np.nan), np.nan)
    return out


class Feat:
    """Feature cube for one data view. Arrays are (series, time)."""
    names = ['d1', 'd2', 'd3', 'd6', 'd12', 'rm3', 'rm6', 'sd6', 'scm', 'msin', 'mcos', 'cc1', 'cc3', 'lat', 'lon', 'comm', 'prov']

    def __init__(self, V, meta, use=None, scm_cache=None):
        S = V.shape[0]
        self.V = V
        use = np.ones(S, bool) if use is None else use
        self.base = {k: lagdiff(V, k) for k in (1, 2, 3, 6, 12)}
        self.rm3 = roll_mean_rel(V, 3)
        self.rm6 = roll_mean_rel(V, 6)
        self.sd6 = roll_sd_changes(V, 6)
        comm = meta.commodity.values
        self.cc1 = cross_change(V, comm, use, 1)
        self.cc3 = cross_change(V, comm, use, 3)
        self.scm = {h: scm_all(V, h) for h in HS} if scm_cache is None else scm_cache
        self.meta = meta
        self.comm_code = pd.Categorical(meta.commodity).codes.astype(float)
        self.prov_code = pd.Categorical(meta.province).codes.astype(float)
        self.lat = meta.lat.values
        self.lon = meta.lon.values

    def matrix(self, h, rows, cols):
        """Feature matrix for series indices `rows` at times `cols` (same length arrays)."""
        mon = (cols + 3) % 12
        X = np.column_stack([
            self.base[1][rows, cols], self.base[2][rows, cols], self.base[3][rows, cols],
            self.base[6][rows, cols], self.base[12][rows, cols],
            self.rm3[rows, cols], self.rm6[rows, cols], self.sd6[rows, cols],
            self.scm[h][rows, cols],
            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
            self.cc1[rows, cols], self.cc3[rows, cols],
            self.lat[rows], self.lon[rows], self.comm_code[rows], self.prov_code[rows]])
        return X


# ---------------------------------------------------------------- baselines
def naive_level(V, t, maxgap=3):
    lv = V[:, t].copy()
    for g in range(1, maxgap + 1):
        miss = np.isnan(lv) & (t - g >= 0)
        lv[miss] = V[miss, t - g]
    return lv


def seasonal_naive(V, t, h):
    base = naive_level(V, t)
    f = V[:, t + h - 12].copy() if t + h - 12 >= 0 else np.full(V.shape[0], np.nan)
    fb = np.zeros(V.shape[0], bool)
    for dlt in (-1, 1):
        j = t + h - 12 + dlt
        if 0 <= j <= t:
            m = np.isnan(f)
            f[m] = V[m, j]
    miss = np.isnan(f)
    f[miss] = base[miss]
    return f, miss


def scm_forecast(F, t, h):
    base = naive_level(F.V, t)
    adj = F.scm[h][:, t]
    return np.where(np.isnan(adj), base, base + adj), np.isnan(adj)


def mase_scale(P, t, h):
    """Mean absolute h-step change using pairs with u+h <= t (raw P); NaN if fewer than 12 pairs."""
    d = np.abs(P[:, h:t + 1] - P[:, :t + 1 - h]) if t + 1 - h > 0 else np.empty((P.shape[0], 0))
    n = (~np.isnan(d)).sum(1)
    with np.errstate(all='ignore'):
        sc = np.nansum(d, 1) / np.where(n > 0, n, np.nan)
    return np.where(n >= 12, sc, np.nan)
