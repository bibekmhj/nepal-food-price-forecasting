import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], parse_dates=["date"])
print("rows", len(df))
print("date range", df.date.min().date(), df.date.max().date())
print("distinct dates", df.date.nunique())
print("markets", df.market_id.nunique(), "| admin2", df.admin2.nunique(), "| provinces", df.admin1.nunique())
print("commodities", df.commodity.nunique())
print()
for c in ["pricetype", "priceflag", "currency", "unit", "category"]:
    print(df[c].value_counts(dropna=False).to_string())
    print()

df["year"] = df.date.dt.year
print("rows per year")
print(df.groupby("year").size().to_string())
print()
print("markets per year")
print(df.groupby("year").market_id.nunique().to_string())
print()

# series = market x commodity
g = df.groupby(["market_id", "market", "admin1", "commodity"])
s = g.agg(n=("price", "size"), first=("date", "min"), last=("date", "max")).reset_index()
print("series (market x commodity):", len(s))
print(s.n.describe().to_string())
print("series with >=100 obs:", (s.n >= 100).sum(), "| >=150:", (s.n >= 150).sum(), "| >=200:", (s.n >= 200).sum())
print()

print("commodity: n obs, n markets")
cm = df.groupby("commodity").agg(obs=("price", "size"), markets=("market_id", "nunique"),
                                 first=("date", "min"), last=("date", "max")).sort_values("obs", ascending=False)
print(cm.to_string())
print()

mk = df.groupby(["market_id", "market", "admin1"]).agg(obs=("price", "size"), commodities=("commodity", "nunique"),
                                                          months=("date", "nunique"),
                                                          first=("date", "min"), last=("date", "max"),
                                                          lat=("latitude", "first"), lon=("longitude", "first")
                                                          ).sort_values("obs", ascending=False)
print(mk.to_string())
print()
print("nulls")
print(df.isna().sum().to_string())
print("price<=0:", (df.price <= 0).sum())
