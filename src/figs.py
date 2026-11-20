import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import geopandas as gpd
import analyze as an, fc

OUT = 'results/'; FIG = 'figures/'
import os; os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100, 'savefig.dpi': 300})
NAMES = {'naive': 'Naive', 'snaive': 'Seasonal naive', 'scm': 'Seasonal change mean', 'ridge': 'Ridge', 'lgbm': 'LightGBM'}

# ---------- Fig 1 workflow (conceptual)
fig, ax = plt.subplots(figsize=(6.0, 3.9)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 66)
def box(x, y, w, h, t):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc='#eef3fb', ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8, color=INK, linespacing=1.25)
def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))
bw = 21.5
xs = [0.5, 26.5, 52.5, 78.0]
txt = ['WFP / HDX\nretail price file\n48,535 rows', 'Monthly panel\nApr 2019 to\nSep 2026\n24 markets,\n21 foods', 'Cleaning rule\n(causal; raw data\nare the primary\ncase)', 'Features known\nat the forecast\norigin']
for x, t in zip(xs, txt):
    box(x, 47, bw, 17, t)
for i in range(3):
    arrow(xs[i] + bw + 0.6, 55.5, xs[i + 1] - 0.6, 55.5)
ax.plot([88.7, 88.7], [46.6, 41], color=INK2, lw=0.9); ax.plot([16, 88.7], [41, 41], color=INK2, lw=0.9)
tx = [0.5, 35.5, 70.5]; tw = 29
bt = ['Rolling-origin\nevaluation\nhorizons 1, 3, 6 months\nnaive, seasonal, ETS,\nSARIMA, Ridge,\nLightGBM',
      'Unseen-market tests\nleave one market out,\nleave one province out\n(own history or\ncold start)',
      'Inference\nDiebold-Mariano test,\nblock bootstrap,\nHolm correction']
for x, t in zip(tx, bt):
    box(x, 14, tw, 22, t); arrow(x + tw / 2, 41, x + tw / 2, 36.6)
ax.text(50, 4, 'Conceptual workflow. All steps use only data available\nat each forecast origin.', ha='center', va='center', fontsize=8, color=INK2)
fig.savefig(FIG + 'fig1_workflow.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 2 market locations (no boundary drawn)
from adjustText import adjust_text
mt = an.mt
fig, ax = plt.subplots(figsize=(6.0, 3.6))
mk = {'low': ('o', 'Low'), 'mid': ('s', 'Medium'), 'high': ('^', 'High')}
for g, (m, lab) in mk.items():
    d = mt[mt.iso_group == g]
    ax.scatter(d.lon, d.lat, marker=m, s=38, color=BLUE, edgecolor='white', lw=0.6, label=lab, zorder=3)
texts = [ax.text(r.lon, r.lat, n, fontsize=8, color=INK) for n, r in mt.iterrows()]
ax.set_xlim(80.2, 88.2); ax.set_ylim(26.1, 30.5)
ax.set_aspect(1 / np.cos(np.radians(28.5)))
adjust_text(texts, ax=ax, expand=(1.5, 1.9), arrowprops=dict(arrowstyle='-', color=GREY, lw=0.5))
for tx in texts:
    if tx.get_text() == 'Rajbiraj':
        tx.set_position((86.62, 26.4)); tx.set_ha('right'); tx.set_va('top')
ax.set_xlabel('Longitude (degrees east)'); ax.set_ylabel('Latitude (degrees north)')
ax.legend(loc='upper right', fontsize=8, frameon=False, title='Isolation group\n(mean distance to 3\nnearest panel markets)', title_fontsize=8, labelspacing=0.6, borderaxespad=0.2)
fig.savefig(FIG + 'fig2_markets.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 3 main results by horizon
M = pd.read_csv(OUT + 'main_all_origins.csv')
M = M[M.model.isin(NAMES)]
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.7), sharey=True)
for ax, h in zip(axs, fc.HS):
    d = M[M.h == h].set_index('model').loc[list(NAMES)]
    cols = [BLUE if m == 'lgbm' else GREY for m in d.index]
    ax.barh(range(5), d.mase, color=cols, height=0.62)
    for i, v in enumerate(d.mase):
        ax.text(v + 0.03, i, f'{v:.2f}', va='center', fontsize=8, color=INK)
    ax.axvline(d.mase['naive'], color=INK2, lw=0.8, ls='--')
    ax.set_xlim(0, 2.45); ax.set_xticks([0, 1.0, 2.0])
    ax.set_title(f'{h} month' + ('s' if h > 1 else ''), fontsize=8, color=INK)
    ax.set_xlabel('Mean MASE', fontsize=8)
axs[0].set_yticks(range(5)); axs[0].set_yticklabels([NAMES[m] for m in list(NAMES)], fontsize=8); axs[0].invert_yaxis()
fig.tight_layout(); fig.savefig(FIG + 'fig3_main.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 4 commodity heterogeneity
B = pd.read_csv(OUT + 'breakdown_A.csv'); B = B[B.dim == 'commodity']
fig, axs = plt.subplots(1, 3, figsize=(6.0, 4.6), sharey=True)
order = B[B.h == 3].sort_values('ratio_lgbm_naive').group.tolist()
for ax, h in zip(axs, fc.HS):
    d = B[B.h == h].set_index('group').loc[order]
    cols = [BLUE if v < 1 else ORANGE for v in d.ratio_lgbm_naive]
    ax.barh(range(len(d)), d.ratio_lgbm_naive - 1, left=1, color=cols, height=0.68)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_title(f'{h} month' + ('s' if h > 1 else ''), fontsize=8, color=INK); ax.set_xlim(0.5, 1.65)
    ax.set_xlabel('LightGBM / naive MASE', fontsize=8)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order, fontsize=8); axs[0].invert_yaxis()
fig.tight_layout(); fig.savefig(FIG + 'fig4_commodity.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 5 unseen markets
U = pd.read_csv(OUT + 'unseen_summary.csv')
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.6))
ax = axs[0]
d = U[(U['mode'] == 'LOMO') & (U.variant == 'A')]
ax.plot(d.h, d.ratio_seen_naive, marker='D', color=GREY, lw=1.6, ms=5, mec='white', ls='--', label='Market in training')
ax.plot(d.h, d.ratio_unseen_naive, marker='o', color=BLUE, lw=1.6, ms=6, mec='white', label='Leave one market out')
d2 = U[(U['mode'] == 'LOPO') & (U.variant == 'A')]
ax.plot(d2.h, d2.ratio_unseen_naive, marker='s', color=ORANGE, lw=1.6, ms=5.5, mec='white', label='Leave one province out')
ax.axhline(1, color=INK, lw=0.8); ax.set_xticks([1, 3, 6])
ax.set_xlabel('Forecast horizon (months)'); ax.set_ylabel('LightGBM MASE / naive MASE'); ax.set_title('(a) Own history available', fontsize=8, loc='left')
ax.legend(fontsize=8, frameon=False, loc='upper right')
ax = axs[1]
d = U[(U['mode'] == 'LOMO') & (U.variant == 'B')].set_index('h')
series = [('mae_B0', 'Mean of other markets', GREY, 'o'), ('mae_B1', 'Nearest 3 markets', AQUA, 's'), ('mae_lgbmB', 'LightGBM, no own history', BLUE, 'D'), ('mae_naive_own', 'Naive with own history', ORANGE, '^')]
for k, lab, col, m in series:
    ax.plot(d.index, d[k], marker=m, color=col, lw=1.6, ms=5.5, mec='white', label=lab)
ax.set_xticks([1, 3, 6]); ax.set_xlabel('Forecast horizon (months)'); ax.set_ylabel('Mean absolute error (log points)')
ax.set_title('(b) No own history (market left out)', fontsize=8, loc='left')
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.set_ylim(0.07, 0.27)
fig.tight_layout(); fig.savefig(FIG + 'fig5_unseen.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 6 stability over origins (h = 3)
zm = np.load(OUT + 'fc_A_naive_snaive_scm_ridge_lgbm.npz')
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.6), sharey=True)
dates = pd.period_range('2019-04', '2026-09', freq='M')
for ax, h in zip(axs, fc.HS):
    o = list(zm[f'origins|{h}']); y, sc, ok = an.triples(h, o)
    om = {m: an.origin_mean(np.abs(zm[f'{m}|{h}'] - y) / sc, ok) for m in ('naive', 'lgbm')}
    r = om['lgbm'] / om['naive']
    x = [dates[t].to_timestamp() for t in o]
    ax.plot(x, r, color=BLUE, lw=1.4); ax.axhline(1, color=INK, lw=0.8)
    ax.set_title(f'{h} month' + ('s' if h > 1 else ''), fontsize=8, color=INK)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax.tick_params(axis='x', labelsize=8)
axs[0].set_ylabel('LightGBM MASE / naive MASE')
fig.tight_layout(); fig.savefig(FIG + 'fig6_stability.png', bbox_inches='tight'); plt.close(fig)
print('figures written')
