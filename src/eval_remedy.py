"""Evaluate the conformal regime-aware ES recalibration on the deficient static
models (Historical, Gaussian) across all assets at the FRTB 97.5% level.
Reports Lambda_ES and AS/MF p-values before vs after. Writes results_remedy.csv."""
import sys, numpy as np, pandas as pd
import sys, os as _os; sys.path.insert(0, _os.path.dirname(__file__))
import es_pipeline as E
from var_models import hs_var, vol_regime

import os
DATA = os.environ.get("ES_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
ASSETS = ["sp500", "nasdaq", "btc", "tlt_bonds", "eurusd", "usdjpy"]
a = 0.025
rows = []
for asset in ASSETS:
    r = E.load_returns(DATA, asset); rg = vol_regime(r)
    for model, var, es in [("Historical", hs_var(r, a), E.hs_es(r, a)),
                           ("Gaussian",   E.norm_var(r, a), E.norm_es(r, a))]:
        var, es = var.dropna(), es.dropna()
        L0 = E.lambda_es(r, var, es, rg, a)
        AS0 = E.acerbi_szekely_Z2(r, var, es, a); MF0 = E.mcneil_frey(r, var, es, a)
        es_c = E.conformal_es(r, var, es, rg, a, regime_aware=True)
        L1 = E.lambda_es(r, var, es_c, rg, a)
        AS1 = E.acerbi_szekely_Z2(r, var, es_c, a); MF1 = E.mcneil_frey(r, var, es_c, a)
        rows.append(dict(asset=asset, model=model,
            lam_calm_before=L0.get("calm", np.nan), lam_stress_before=L0.get("stress", np.nan),
            AS_p_before=AS0["AS_p"], MF_p_before=MF0["MF_p"],
            lam_calm_after=L1.get("calm", np.nan), lam_stress_after=L1.get("stress", np.nan),
            AS_p_after=AS1["AS_p"], MF_p_after=MF1["MF_p"]))
        print(f"{asset:10s} {model:10s} stress Lambda {L0.get('stress',float('nan')):.3f} -> "
              f"{L1.get('stress',float('nan')):.3f} | MF p {MF0['MF_p']:.3f} -> {MF1['MF_p']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__),"..","results",os.path.join(os.path.dirname(__file__),"..","results","results_remedy.csv")), index=False)
print("REMEDY DONE")
