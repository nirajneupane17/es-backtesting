"""
Paper 2 — Expected Shortfall validation under regime shifts.
Extends the Paper 1 VaR pipeline (ml-var-validation) with:
  * ES forecasters (Historical, Gaussian, GARCH-t closed-form, ML/quantile)
  * ES backtests: Acerbi-Szekely Z2, McNeil-Frey exceedance residuals
  * Regime-conditional ES severity ratio Lambda_ES  (ES analog of Paper 1 Lambda)
  * Formal regime-effect test on ES exceedance residuals

Conventions: everything in RETURN space (signed). For level a (e.g. 0.025),
VaR_a and ES_a are the a-quantile and lower-tail mean of the return, both
NEGATIVE numbers. A breach is r_t < VaR_t. Good ES => E[r_t - ES_t | breach]=0.
"""
import sys, warnings, itertools
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from var_models import (load_returns, make_features, vol_regime,
                      hs_var, garch_var, gbm_var, kupiec, christoffersen, exceed)

WINDOW, REFIT = 500, 21

# ----------------------------------------------------------------------
#  ES forecasters (one-step-ahead, walk-forward) -- all return RETURN-space ES
# ----------------------------------------------------------------------
def hs_es(r, a, window=WINDOW, start=None):
    v = r.values; n = len(v); start = start or window; o = {}
    for t in range(start, n):
        w = v[t-window:t]; q = np.quantile(w, a); tail = w[w <= q]
        o[r.index[t]] = float(tail.mean()) if tail.size else float(q)
    return pd.Series(o, name="HistoricalSim")

def norm_var(r, a, window=WINDOW, start=None):
    v = r.values; n = len(v); start = start or window; o = {}
    for t in range(start, n):
        w = v[t-window:t]; o[r.index[t]] = w.mean() + w.std(ddof=1)*stats.norm.ppf(a)
    return pd.Series(o, name="Gaussian")

def _norm_es(mu, sg, a):
    return mu - sg*stats.norm.pdf(stats.norm.ppf(a))/a

def norm_es(r, a, window=WINDOW, start=None):
    v = r.values; n = len(v); start = start or window; o = {}
    for t in range(start, n):
        w = v[t-window:t]; o[r.index[t]] = _norm_es(w.mean(), w.std(ddof=1), a)
    return pd.Series(o, name="Gaussian")

def _t_es(mu, sg, nu, a):
    # standardized (unit-variance) Student-t ES in return space, lower tail
    q = stats.t.ppf(a, nu); s = np.sqrt((nu-2)/nu)
    es_std = -(stats.t.pdf(q, nu)/a)*(nu+q**2)/(nu-1)   # E[T|T<=q], standard t
    return mu + sg*s*es_std

def garch_t_es(r, a, window=WINDOW, start=None, refit=REFIT):
    from arch import arch_model
    v = r.values*100; n = len(v); start = start or window; o = {}; res = None
    for i, t in enumerate(range(start, n)):
        if res is None or i % refit == 0:
            res = arch_model(v[t-window:t], mean="Constant", vol="GARCH",
                             p=1, q=1, dist="t").fit(disp="off")
        f = res.forecast(horizon=1, reindex=False)
        sg = np.sqrt(f.variance.values[-1,0]); mu = res.params.get("mu",0.0)
        nu = res.params["nu"]
        o[r.index[t]] = _t_es(mu, sg, nu, a)/100.0
    return pd.Series(o, name="GARCH-t")

def gbm_es(f, a, window=WINDOW, start=None, refit=REFIT, M=5, n_est=60):
    # ES via integral representation ES_a = (1/a) int_0^a VaR_u du ~ mean of VaR on a grid
    from sklearn.ensemble import GradientBoostingRegressor
    grid = a*(np.arange(1, M+1)/M)                       # u in (0, a]
    X = f.drop(columns="y").values; y = f["y"].values; idx = f.index
    n = len(f); start = start or window; o = {}; models = None
    for i, t in enumerate(range(start, n)):
        if models is None or i % refit == 0:
            models = [GradientBoostingRegressor(loss="quantile", alpha=u,
                        n_estimators=n_est, max_depth=2, learning_rate=0.05,
                        subsample=0.9, random_state=0).fit(X[t-window:t], y[t-window:t])
                      for u in grid]
        o[idx[t]] = float(np.mean([m.predict(X[t:t+1])[0] for m in models]))
    return pd.Series(o, name="GBM-ES")

# ----------------------------------------------------------------------
#  ES backtests
# ----------------------------------------------------------------------
def _align(r, var, es):
    d = pd.concat([r.rename("r"), var.rename("var"), es.rename("es")], axis=1).dropna()
    return d["r"].values, d["var"].values, d["es"].values

def acerbi_szekely_Z2(r, var, es, a, B=10000, seed=0):
    """AS (2014) Z2 severity statistic with a breach-conditional bootstrap p-value.
    On breach days the ratio r_t/ES_t has H0 mean 1 (since ES is the tail mean).
    Z2 = mean(r/ES on breaches) - 1  (>0 => ES understates realized severity).
    p-value: bootstrap the breach-day ratios recentred to the H0 mean of 1."""
    rr, vv, ee = _align(r, var, es); br = rr < vv; m = int(br.sum())
    if m < 2:
        return {"Z2": np.nan, "AS_p": np.nan}
    ratio = rr[br]/ee[br]                       # >0; H0 mean 1
    stat = ratio.mean()
    rng = np.random.default_rng(seed); rc = ratio - stat + 1.0
    boot = np.array([rng.choice(rc, m, replace=True).mean() for _ in range(B)])
    p = float((boot >= stat).mean())            # one-sided: ES too small
    return {"Z2": float(stat - 1.0), "AS_p": p}

def mcneil_frey(r, var, es, a, B=10000, seed=0):
    """Exceedance-residual bootstrap. On breach days d = r - ES; H0 mean(d)=0,
    H1 mean(d)<0 (realized losses worse than ES). Studentized bootstrap p-value."""
    rr, vv, ee = _align(r, var, es); br = rr < vv
    d = rr[br] - ee[br]; m = d.size
    if m < 2: return {"MF_mean_resid": np.nan, "MF_p": np.nan, "n_breach": int(m)}
    stat = d.mean()
    rng = np.random.default_rng(seed); dc = d - d.mean(); boot = np.empty(B)
    for b in range(B): boot[b] = rng.choice(dc, m, replace=True).mean()
    p = float((boot <= stat).mean())                     # one-sided (ES too small)
    return {"MF_mean_resid": float(stat), "MF_p": p, "n_breach": int(m)}

def lambda_es(r, var, es, rg, a):
    """Regime-conditional ES severity ratio = mean realized tail loss / mean predicted ES,
    computed within each regime on VaR-breach days. >1 => ES understates severity."""
    d = pd.concat([r.rename("r"), var.rename("var"), es.rename("es"),
                   rg.rename("g")], axis=1).dropna()
    out = {}
    for k, g in d.groupby("g"):
        br = g["r"] < g["var"]
        if br.sum() >= 2:
            out[str(k)] = float(g.loc[br, "r"].mean()/g.loc[br, "es"].mean())
        else:
            out[str(k)] = np.nan
    return out

def regime_effect(r, var, es, rg, a):
    """Formal test of the regime effect on the ES exceedance residual.
    OLS: (r - ES) on stress dummy, restricted to VaR-breach days. Negative,
    significant coef => ES is extra-deficient in stress."""
    import statsmodels.api as sm
    d = pd.concat([r.rename("r"), var.rename("var"), es.rename("es"),
                   rg.rename("g")], axis=1).dropna()
    br = d["r"] < d["var"]; d = d[br]
    if d.empty or d["g"].nunique() < 2:
        return {"beta_stress": np.nan, "p_value": np.nan, "n": int(len(d))}
    d["resid"] = d["r"] - d["es"]; d["stress"] = (d["g"] == "stress").astype(float)
    X = sm.add_constant(d["stress"].values); m = sm.OLS(d["resid"].values, X).fit()
    return {"beta_stress": float(m.params[1]), "p_value": float(m.pvalues[1]), "n": int(len(d))}

# ----------------------------------------------------------------------
#  REMEDY: split-conformal, regime-aware ES severity recalibration
# ----------------------------------------------------------------------
def conformal_es(r, var, es, rg, a, calib=500, regime_aware=True, min_breach=8):
    """Walk-forward recalibration of ES severity. Over a trailing window, estimate
    the realized severity ratio s = mean(r/ES) on VaR-breach days (optionally within
    the current regime) and rescale: ES_adj_t = ES_t * s. Closes Lambda_ES toward 1.
    Returns the recalibrated ES series (NaN until the first calibration is available)."""
    d = pd.concat([r.rename("r"), var.rename("var"), es.rename("es"),
                   rg.rename("g")], axis=1).dropna()
    rr, vv, ee, gg = d["r"].values, d["var"].values, d["es"].values, d["g"].values
    n = len(d); adj = np.full(n, np.nan)
    for t in range(calib, n):
        wr, wv, we, wg = rr[t-calib:t], vv[t-calib:t], ee[t-calib:t], gg[t-calib:t]
        br = wr < wv
        if regime_aware:
            sel = br & (wg == gg[t])
            if sel.sum() < min_breach:          # fall back to all-regime breaches
                sel = br
        else:
            sel = br
        s = (wr[sel]/we[sel]).mean() if sel.sum() >= 2 else 1.0
        s = min(max(s, 0.5), 3.0)               # guard against degenerate windows
        adj[t] = ee[t]*s
    return pd.Series(adj, index=d.index, name=es.name + "-Conformal")

# ----------------------------------------------------------------------
#  self-tests for the closed-form ES formulas
# ----------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(1)
    # Gaussian
    x = rng.normal(0.001, 0.02, 4_000_000); a = 0.025
    q = np.quantile(x, a); emp = x[x <= q].mean(); ana = _norm_es(0.001, 0.02, a)
    assert abs(emp-ana) < 5e-4, (emp, ana)
    # Student-t (unit-variance scaled)
    nu = 6; s = np.sqrt((nu-2)/nu); mu, sg = 0.0, 0.015
    xt = mu + sg*s*rng.standard_t(nu, 4_000_000)
    qt = np.quantile(xt, a); empt = xt[xt <= qt].mean(); anat = _t_es(mu, sg, nu, a)
    assert abs(empt-anat) < 6e-4, (empt, anat)
    print(f"[selftest] Gaussian ES emp={emp:.5f} ana={ana:.5f} | t ES emp={empt:.5f} ana={anat:.5f}  OK")

if __name__ == "__main__":
    _selftest()
