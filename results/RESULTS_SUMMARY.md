# Results summary — ES backtesting under regime shifts

Cells: 42 (6 assets x 2 levels x up to 4 models).

## Regime-conditional ES deficiency (stress Lambda_ES, 97.5%)

| Model | min | mean | max |
|---|---|---|---|
| Historical | 1.083 | 1.117 | 1.145 |
| Gaussian | 1.174 | 1.326 | 1.438 |
| GARCH-t | 1.061 | 1.104 | 1.171 |
| ML-GBM | 1.238 | 1.326 | 1.398 |

- Regime effect on ES exceedance residual is statistically significant (p<0.05) in **69% of all cells**.
- Calm-regime Lambda_ES sits near 1.0; the deficiency is concentrated in stress.

## 'False comfort' cells: standard ES test PASSES but regime test flags stress deficiency

| Asset | Level | Model | AS p | MF p | stress Lambda | Regime p |
|---|---|---|---|---|---|---|
| btc | 0.025 | GARCH-t | 0.112 | 0.351 | 1.140 | 0.034 |
| eurusd | 0.025 | GARCH-t | 0.104 | 0.392 | 1.066 | 0.017 |
| usdjpy | 0.01 | Historical | 0.057 | 0.143 | 1.097 | 0.033 |

## Remedy: regime-aware conformal recalibration (97.5%)

| Asset | Model | stress Lambda before -> after | MF p before -> after |
|---|---|---|---|
| sp500 | Historical | 1.132 -> 0.991 | 0.010 -> 0.683 |
| sp500 | Gaussian | 1.381 -> 1.002 | 0.000 -> 0.467 |
| nasdaq | Historical | 1.145 -> 1.001 | 0.004 -> 0.537 |
| nasdaq | Gaussian | 1.396 -> 1.014 | 0.000 -> 0.328 |
| btc | Historical | 1.083 -> 0.974 | 0.544 -> 0.804 |
| btc | Gaussian | 1.438 -> 0.983 | 0.000 -> 0.792 |
| tlt_bonds | Historical | 1.122 -> 0.997 | 0.071 -> 0.831 |
| tlt_bonds | Gaussian | 1.174 -> 1.040 | 0.001 -> 0.306 |
| eurusd | Historical | 1.103 -> 1.026 | 0.037 -> 0.550 |
| eurusd | Gaussian | 1.200 -> 1.014 | 0.000 -> 0.540 |
| usdjpy | Historical | 1.115 -> 0.973 | 0.041 -> 0.871 |
| usdjpy | Gaussian | 1.366 -> 1.038 | 0.000 -> 0.262 |