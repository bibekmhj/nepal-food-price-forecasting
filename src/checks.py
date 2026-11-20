import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], parse_dates=["date"])

# 1. calendar of the core panel (markets with data in 2025)
core_ids = df[df.date.dt.year == 2025].market_id.unique()
core = df[df.market_id.isin(core_ids) & (df.date >= "2019-04-01")].copy()
print("core markets:", core.market_id.nunique())
months = pd.period_range("2019-04", "2026-09", freq="M")
have = set(core.date.dt.to_period("M"))
missing = [str(m) for m in months if m not in have]
print("calendar months expected:", len(months), "| months with any data:", len(have))
print("calendar months missing entirely:", missing)
print()

# 2. coverage per core market by month
cov = core.groupby([core.market_id, core.date.dt.to_period("M")]).size().unstack(0)
cov = cov.reindex(months)
print("months where fewer than 20 of 25 markets reported:")
n_m = (cov.notna()).sum(axis=1)
print(n_m[n_m < 20].to_string())
print()

# 3. units per commodity
u = df.groupby("commodity").unit.unique()
print("units per commodity:")
print(u.to_string())
print()

# 4. gaps within the 21-commodity core series
core21 = core[~core.commodity.isin(["Mustard leaves", "Sugar", "Salt", "Fuel (petrol-gasoline)", "Fuel (diesel)"])]
piv = core21.pivot_table(index=core21.date.dt.to_period("M"), columns=["market_id", "commodity"], values="price")
piv = piv.reindex(months)
share_missing = piv.isna().mean()
print("series in core panel:", piv.shape[1])
print("share of months missing per series, describe:")
print(share_missing.describe().to_string())
print("series with >20% missing:", (share_missing > 0.2).sum())
print("total cells:", piv.size, "| filled:", int(piv.notna().sum().sum()), "| filled share: %.3f" % piv.notna().values.mean())
print()

# 5. spikes: month-to-month ratio
lp = np.log(piv)
d = lp.diff()
big = (d.abs() > np.log(2)).sum().sum()
print("month-to-month moves of more than a factor of 2 (up or down):", int(big), "of", int(d.notna().sum().sum()))
s = d.stack([0, 1])
s = s[s.abs() > np.log(2)].sort_values()
print(s.head(8).to_string())
print(s.tail(8).to_string())
print()

# 6. description of how similar markets move (descriptive only)
res = []
for com in ["Tomatoes", "Rice (medium grain)", "Potatoes (red)", "Eggs", "Oil (soybean)", "Meat (chicken)"]:
    sub = d.xs(com, axis=1, level=1)
    c = sub.corr(min_periods=40).values
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu]
    res.append((com, np.nanmean(vals), np.nanmedian(vals)))
print("mean / median pairwise correlation of monthly log price changes between markets")
for r in res:
    print("%-22s mean %.2f median %.2f" % r)
print()

# 7. remote markets present?
remote = ["Simikot", "Gamgadhi", "Dunai", "Jumla Khalanga", "Mangalsen", "Martadi", "Chainpur", "Musikot", "Birendranagar"]
print(core[core.market.isin(remote)].groupby("market").agg(n=("price", "size"), alt_lat=("latitude", "first"), lon=("longitude", "first")).to_string())
