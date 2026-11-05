"""
Full ES validation grid: 6 assets x {97.5%, 99%} x {Historical, Gaussian, GARCH-t, ML}.
Writes incrementally to results_es_grid.csv so partial progress survives timeouts.
"""
import sys, os, time, numpy as np, pandas as pd
import sys, os as _os; sys.path.insert(0, _os.path.dirname(__file__))
import es_pipeline as E
from var_models import hs_var, garch_var, gbm_var, vol_regime, make_features

import os
DATA = os.environ.get("ES_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
OUT  = "os.path.join(os.path.dirname(__file__),"..","results",os.path.join(os.path.dirname(__file__),"..","results","results_es_grid.csv"))"
ASSETS = ["sp500", "nasdaq", "btc", "tlt_bonds", "eurusd", "usdjpy"]
LEVELS = [0.025, 0.01]

def es_overall_gap(r, es):                      # mean realized tail loss / mean ES on breach-of-ES... use VaR breaches
    return np.nan

def row_for(asset, a, model, r, rg, var, es):
    var, es = var.dropna(), es.dropna()
    k  = E.kupiec(E.exceed(r, var), a)
    AS = E.acerbi_szekely_Z2(r, var, es, a)
    MF = E.mcneil_frey(r, var, es, a)
    L  = E.lambda_es(r, var, es, rg, a)
    RE = E.regime_effect(r, var, es, rg, a)
    return {
        "asset": asset, "level": a, "model": model,
        "n": k["n"], "breaches": k["breaches"], "breach_rate": k["obs_rate"],
        "kupiec_p": k["p_value"],
        "AS_Z2": AS["Z2"], "AS_p": AS["AS_p"],
        "MF_resid": MF["MF_mean_resid"], "MF_p": MF["MF_p"], "n_breach": MF["n_breach"],
        "lambda_calm": L.get("calm", np.nan), "lambda_stress": L.get("stress", np.nan),
        "regime_beta": RE["beta_stress"], "regime_p": RE["p_value"],
    }

def append(row):
    df = pd.DataFrame([row])
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)

def done(asset, a, model):
    if not os.path.exists(OUT): return False
    d = pd.read_csv(OUT)
    return ((d.asset==asset)&(d.level==a)&(d.model==model)).any()

if __name__ == "__main__":
    for asset in ASSETS:
        r = E.load_returns(DATA, asset); rg = vol_regime(r); f = make_features(r)
        for a in LEVELS:
            specs = [
                ("Historical", lambda: (hs_var(r, a),            E.hs_es(r, a))),
                ("Gaussian",   lambda: (E.norm_var(r, a),        E.norm_es(r, a))),
                ("GARCH-t",    lambda: (garch_var(r, a),         E.garch_t_es(r, a))),
            ]
            if a == 0.025:   # ML/quantile ES at the FRTB 97.5% level only (compute-heavy)
                specs.append(("ML-GBM", lambda: (gbm_var(f, a), E.gbm_es(f, a))))
            for model, build in specs:
                if done(asset, a, model):
                    print(f"skip {asset} {a} {model}", flush=True); continue
                t0 = time.time()
                try:
                    var, es = build()
                    append(row_for(asset, a, model, r, rg, var, es))
                    print(f"OK   {asset} {a} {model}  ({time.time()-t0:.0f}s)", flush=True)
                except Exception as ex:
                    print(f"FAIL {asset} {a} {model}: {ex}", flush=True)
    print("GRID DONE", flush=True)
