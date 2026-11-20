import sys, time, warnings, numpy as np
import fc
from multiprocessing import Pool
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

P, meta = fc.load_panel('data/wfp_food_prices_npl.csv')
S = P.shape[0]
ORIG = list(range(fc.FIRST_ORIGIN, fc.NT - 1, 3))


def interp_known(y):
    y = y.copy()
    idx = np.arange(len(y)); ok = ~np.isnan(y)
    if ok.sum() < 2:
        return y
    first, last = idx[ok][0], idx[ok][-1]
    y[first:last + 1] = np.interp(idx[first:last + 1], idx[ok], y[ok])
    return y


def work(i):
    out = {}
    for t in ORIG:
        y = P[i, :t + 1].copy()
        base = fc.naive_level(P[i:i + 1], t)[0] if False else None
        last = y[~np.isnan(y)][-1] if (~np.isnan(y)).any() else np.nan
        fe = np.full(6, np.nan); fs = np.full(6, np.nan); ef = 0; sf = 0
        try:
            yi = interp_known(y)
            yi = yi[~np.isnan(yi)]
            m = ExponentialSmoothing(yi, trend='add', damped_trend=True, initialization_method='estimated').fit()
            fe = m.forecast(6)
        except Exception:
            ef = 1
        try:
            m = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), trend='n',
                        enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False, maxiter=100)
            fs = r.forecast(6)
        except Exception:
            sf = 1
        out[t] = (fe, fs, ef, sf)
    return i, out


if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p:
        results = p.map(work, range(S), chunksize=8)
    res = {}
    for h in fc.HS:
        nor = [t for t in ORIG if t + h <= fc.NT - 1]
        e = np.full((len(nor), S), np.nan); s = np.full((len(nor), S), np.nan)
        ef = np.zeros((len(nor), S), bool); sf = np.zeros((len(nor), S), bool)
        for i, o in results:
            for oi, t in enumerate(nor):
                fe, fs, a, b = o[t]
                e[oi, i] = fe[h - 1]; s[oi, i] = fs[h - 1]; ef[oi, i] = a; sf[oi, i] = b
        res[f'ets|{h}'] = e; res[f'sarima|{h}'] = s; res[f'fb_ets|{h}'] = ef; res[f'fb_sarima|{h}'] = sf
        res[f'origins|{h}'] = np.array(nor)
    np.savez_compressed('results/fc_A_stat.npz', **res)
    print('done', round(time.time() - t0), 's')
