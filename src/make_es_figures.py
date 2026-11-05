"""Figures for Paper 2. Reads results_es_grid.csv (+ recomputes a couple of light
series) and writes PDF+PNG figures into /home/claude/paper2/figs/."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os as _os; sys.path.insert(0, _os.path.dirname(__file__))
import es_pipeline as E
from var_models import hs_var, vol_regime

import os
DATA = os.environ.get("ES_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(os.path.dirname(__file__),"..","figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
INK = "#20222a"; ACC = "#b4232a"; CALM = "#6b8fb5"; STRESS = "#c0603a"
g = pd.read_csv(os.path.join(os.path.dirname(__file__),"..","results","results_es_grid.csv")); g97 = g[g.level == 0.025]
LAB = {"sp500":"S&P 500","nasdaq":"Nasdaq","btc":"Bitcoin","tlt_bonds":"US 20Y Bonds",
       "eurusd":"EUR/USD","usdjpy":"USD/JPY"}

def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=150); plt.close(fig)

# ---- Fig 1: Lambda_ES calm vs stress by model (S&P 500) ----
d = g97[g97.asset == "sp500"].set_index("model")
models = [m for m in ["Historical","Gaussian","GARCH-t","ML-GBM"] if m in d.index]
x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(x-w/2, [d.loc[m,"lambda_calm"] for m in models], w, label="Calm regime", color=CALM)
ax.bar(x+w/2, [d.loc[m,"lambda_stress"] for m in models], w, label="Stress regime", color=STRESS)
ax.axhline(1.0, color=INK, lw=1, ls="--"); ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel(r"ES severity ratio  $\Lambda_{ES}$"); ax.legend(frameon=False)
ax.set_title("Realized tail loss relative to predicted ES, by regime (S&P 500, 97.5%)")
savefig(fig, "fig1_lambda_by_regime")

# ---- Fig 2: rolling realized tail severity vs predicted ES (S&P 500, Historical) ----
r = E.load_returns(DATA, "sp500"); rg = vol_regime(r); es = E.hs_es(r, 0.025); var = hs_var(r, 0.025)
dd = pd.concat([r.rename("r"), var.rename("v"), es.rename("es"), rg.rename("g")], axis=1).dropna()
br = dd["r"] < dd["v"]
sev = (dd["r"].where(br)/dd["es"]).rolling(252, min_periods=30).mean()
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(sev.index, sev.values, color=ACC, lw=1.1, label="Rolling realized/predicted ES (1y)")
ax.axhline(1.0, color=INK, lw=1, ls="--")
instress = (dd["g"] == "stress").values
ax.fill_between(dd.index, 0, 1, where=instress, transform=ax.get_xaxis_transform(),
                color=STRESS, alpha=0.12, label="Stress regime")
ax.set_ylabel(r"Severity ratio  $r_t/\mathrm{ES}_t$"); ax.set_ylim(0.6, 2.2)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Historical-Simulation ES severity clusters in stress (S&P 500, 97.5%)")
savefig(fig, "fig2_rolling_severity")

# ---- Fig 3: stress Lambda_ES by asset (Historical model) ----
h = g97[g97.model == "Historical"].set_index("asset")
assets = [a for a in ["sp500","nasdaq","btc","tlt_bonds","eurusd","usdjpy"] if a in h.index]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
vals = [h.loc[a,"lambda_stress"] for a in assets]
ax.bar(range(len(assets)), vals, color=[STRESS if v>1.05 else CALM for v in vals])
ax.axhline(1.0, color=INK, lw=1, ls="--")
ax.set_xticks(range(len(assets))); ax.set_xticklabels([LAB[a] for a in assets], rotation=20, ha="right")
ax.set_ylabel(r"Stress-regime $\Lambda_{ES}$")
ax.set_title("ES understates stress-regime tail loss across every asset (Historical, 97.5%)")
savefig(fig, "fig3_lambda_by_asset")

# ---- Fig 4 (if remedy ran): Lambda stress before vs after conformal recalibration ----
if os.path.exists(os.path.join(os.path.dirname(__file__),"..","results","results_remedy.csv")):
    rem = pd.read_csv(os.path.join(os.path.dirname(__file__),"..","results","results_remedy.csv"))
    hh = rem[rem.model == "Historical"].set_index("asset")
    assets = [a for a in ["sp500","nasdaq","btc","tlt_bonds","eurusd","usdjpy"] if a in hh.index]
    x = np.arange(len(assets)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x-w/2, [hh.loc[a,"lam_stress_before"] for a in assets], w, label="Before", color=STRESS)
    ax.bar(x+w/2, [hh.loc[a,"lam_stress_after"] for a in assets], w, label="After conformal", color=CALM)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in assets], rotation=20, ha="right")
    ax.set_ylabel(r"Stress-regime $\Lambda_{ES}$"); ax.legend(frameon=False)
    ax.set_title("Regime-aware conformal recalibration closes the ES gap (Historical, 97.5%)")
    savefig(fig, "fig4_remedy")

print("FIGURES:", sorted(os.listdir(OUT)))
