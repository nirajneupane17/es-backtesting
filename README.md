# Governance-Grade Backtesting of Expected Shortfall under Regime Shifts

Replication code and data for:

> **When the Tail Lies Too: Governance-Grade Backtesting of Expected Shortfall
> under Regime Shifts**
> Niraj Neupane, CA (ICAI) — Korvane / Calderyn Institute
> SSRN working paper, 2026.

Sequel to *When Backtests Lie* ([ml-var-validation](https://github.com/nirajneupane17/ml-var-validation),
SSRN 7652318). That paper showed Value-at-Risk models that pass standard
backtests break down in stress regimes. This one turns the same governance lens
on **Expected Shortfall** — the measure Basel III FRTB actually uses to set
market-risk capital, and the one regulators do *not* backtest directly, because
ES is not elicitable on its own.

This repository reproduces every table and figure from public data, with fixed
random seeds.

---

## The result in one line

Across equities, fixed income, FX, and crypto, Expected Shortfall models in
common use — Historical Simulation and Gaussian — **understate realized
tail losses by 10–40% in stress regimes** while looking far healthier overall.
A standard scalar ES backtest cannot see *where* the failure lives; a
**regime-conditional severity ratio** Λ_ES localizes it (Λ_ES ≈ 1 in calm
regimes, materially > 1 in stress, with a statistically significant regime
effect), and in some cases (e.g. GARCH-_t_ on Bitcoin) the standard test
**passes** while the regime test still flags significant stress deficiency. A
simple, transparent **regime-aware conformal recalibration** closes Λ_ES back to
≈ 1 and restores passing ES backtests on every asset.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── src/
│   ├── var_models.py      # VaR forecasters + coverage tests (from Paper 1, vendored)
│   ├── es_pipeline.py     # ES forecasters, ES backtests, Λ_ES, conformal remedy
│   ├── run_es_grid.py     # full grid: 6 assets × {97.5%, 99%} × 4 models → results/
│   ├── eval_remedy.py     # regime-aware conformal ES recalibration → results/
│   └── make_es_figures.py # publication figures → figures/
├── data/                  # daily returns + regime / stress indicators (reproducible)
├── results/               # ES scorecards (CSV) produced by the pipeline
├── figures/               # PDF/PNG figures used in the paper
└── paper/                 # manuscript PDF
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt

python src/run_es_grid.py       # 1. full ES grid        → results/results_es_grid.csv
python src/eval_remedy.py       # 2. conformal remedy    → results/results_remedy.csv
python src/make_es_figures.py   # 3. figures             → figures/
```

Everything reads only from `data/` and is fully offline and seeded. The ML /
quantile ES model is compute-heavy (it averages several quantile fits per
re-estimation); it is run at the 97.5% FRTB level.

## Data

All data is **public and free**, and identical to the Paper 1 dataset for
direct comparability. Equity indices, the Treasury ETF, and Bitcoin come from
Yahoo Finance (Stooq fallback); FX and the financial-stress indices come from
FRED. Daily log returns, 2000–2026 where available (Treasuries from 2002,
Bitcoin from 2014). No proprietary or employer data is used.

| Asset | Ticker | Class |
|---|---|---|
| S&P 500 | `^GSPC` | Equity (primary) |
| Nasdaq Composite | `^IXIC` | Equity |
| Long Treasuries | `TLT` | Fixed income |
| EUR/USD | `DEXUSEU` (FRED) | FX |
| USD/JPY | `DEXJPUS` (FRED) | FX |
| Bitcoin | `BTC-USD` | Crypto (high-vol) |

Stress / regime inputs: realized-volatility regime split (primary), plus VIX,
the Chicago Fed NFCI and the St. Louis Fed Financial Stress Index for robustness.

## Method

Four one-step-ahead, walk-forward ES forecasters at the FRTB **97.5%** level
(and 99% for the analytic models): Historical Simulation, Gaussian, GARCH-_t_
(closed-form Student-_t_ ES), and a gradient-boosted quantile ES (ES recovered
by integrating VaR over sub-levels). Each is run through an ES validation
battery:

1. **Unconditional ES adequacy** — Acerbi–Székely Z2 severity statistic
2. **Exceedance-residual test** — McNeil–Frey bootstrap of breach-day residuals
3. **Regime-conditional severity** — Λ_ES = mean realized tail loss ÷ mean
   predicted ES, computed separately in calm and stress regimes
4. **Formal regime effect** — regression of the ES exceedance residual on a
   stress-regime indicator
5. **Remedy** — regime-aware split-conformal recalibration of ES severity

The closed-form Gaussian and Student-_t_ ES formulas are unit-tested against
Monte-Carlo simulation (`python src/es_pipeline.py`).

## Reproducibility

All randomness is seeded. `run_es_grid.py` writes `results/results_es_grid.csv`
(one row per asset × level × model); `eval_remedy.py` writes
`results/results_remedy.csv` (Λ_ES and backtest p-values before vs. after
recalibration). The numbers in the paper's tables correspond exactly to these
files (500-day window, monthly re-estimation `REFIT = 21`).

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`) and, if
relevant, the companion VaR paper (SSRN 7652318).

## License

MIT — see `LICENSE`. Data is redistributed under the terms of its original
public providers; this repository stores derived daily returns for convenience
and reproducibility.
