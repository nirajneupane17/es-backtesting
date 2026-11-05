"""Emit LaTeX tables + a human-readable results summary from the grid/remedy CSVs."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "..", "results")
g = pd.read_csv(os.path.join(RES, "results_es_grid.csv"))
rem = pd.read_csv(os.path.join(RES, "results_remedy.csv"))
LAB = {"sp500":"S\\&P 500","nasdaq":"Nasdaq","btc":"Bitcoin","tlt_bonds":"US 20Y Bonds",
       "eurusd":"EUR/USD","usdjpy":"USD/JPY"}
MORD = {"Historical":0,"Gaussian":1,"GARCH-t":2,"ML-GBM":3}
def star(p):
    return "" if pd.isna(p) else ("$^{***}$" if p<.01 else "$^{**}$" if p<.05 else "$^{*}$" if p<.10 else "")

def main_table(level, fname):
    d = g[g.level==level].copy()
    d["mo"]=d.model.map(MORD); d=d.sort_values(["asset","mo"])
    lines=[r"\begin{tabular}{llrrrrrr}", r"\toprule",
      r"Asset & Model & Breach & AS-$Z_2$ & MF $p$ & $\Lambda_{ES}^{\text{calm}}$ & $\Lambda_{ES}^{\text{stress}}$ & Regime $p$ \\",
      r"\midrule"]
    cur=None
    for _,r_ in d.iterrows():
        a = LAB[r_.asset] if r_.asset!=cur else ""
        cur=r_.asset
        lines.append(f"{a} & {r_.model} & {r_.breach_rate:.3f} & {r_.AS_Z2:+.3f} & "
                     f"{r_.MF_p:.3f} & {r_.lambda_calm:.3f} & {r_.lambda_stress:.3f}{star(r_.regime_p)} & {r_.regime_p:.3f} \\\\")
        if r_.model=="ML-GBM" or (r_.asset!=cur): pass
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES,fname),"w").write("\n".join(lines))

def remedy_table(fname):
    d = rem.copy(); d["mo"]=d.model.map(MORD); d=d.sort_values(["asset","mo"])
    lines=[r"\begin{tabular}{llrrrr}", r"\toprule",
      r"Asset & Model & $\Lambda_{ES}^{\text{stress}}$ before & after & MF $p$ before & after \\",
      r"\midrule"]
    cur=None
    for _,r_ in d.iterrows():
        a = LAB[r_.asset] if r_.asset!=cur else ""; cur=r_.asset
        lines.append(f"{a} & {r_.model} & {r_.lam_stress_before:.3f} & {r_.lam_stress_after:.3f} & "
                     f"{r_.MF_p_before:.3f} & {r_.MF_p_after:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES,fname),"w").write("\n".join(lines))

main_table(0.025,"table_es_975.tex")
main_table(0.010,"table_es_99.tex")
remedy_table("table_remedy.tex")

# human-readable summary
g0=g[g.level==0.025]
pass_overall=(g["AS_p"]>=0.05)&(g["MF_p"]>=0.05)
false_comfort=g[pass_overall & (g["regime_p"]<0.05)]
sig_regime=(g["regime_p"]<0.05).mean()
lines=["# Results summary — ES backtesting under regime shifts","",
 f"Cells: {len(g)} (6 assets x 2 levels x up to 4 models).","",
 "## Regime-conditional ES deficiency (stress Lambda_ES, 97.5%)","",
 "| Model | min | mean | max |","|---|---|---|---|"]
for m in ["Historical","Gaussian","GARCH-t","ML-GBM"]:
    s=g0[g0.model==m]["lambda_stress"]
    if len(s): lines.append(f"| {m} | {s.min():.3f} | {s.mean():.3f} | {s.max():.3f} |")
lines += ["",
 f"- Regime effect on ES exceedance residual is statistically significant (p<0.05) in "
 f"**{sig_regime*100:.0f}% of all cells**.",
 f"- Calm-regime Lambda_ES sits near 1.0; the deficiency is concentrated in stress.",
 "",
 "## 'False comfort' cells: standard ES test PASSES but regime test flags stress deficiency","",
 "| Asset | Level | Model | AS p | MF p | stress Lambda | Regime p |","|---|---|---|---|---|---|---|"]
for _,r_ in false_comfort.drop_duplicates(['asset','level','model']).iterrows():
    lines.append(f"| {r_.asset} | {r_.level} | {r_.model} | {r_.AS_p:.3f} | {r_.MF_p:.3f} | {r_.lambda_stress:.3f} | {r_.regime_p:.3f} |")
lines += ["","## Remedy: regime-aware conformal recalibration (97.5%)","",
 "| Asset | Model | stress Lambda before -> after | MF p before -> after |","|---|---|---|---|"]
for _,r_ in rem.iterrows():
    lines.append(f"| {r_.asset} | {r_.model} | {r_.lam_stress_before:.3f} -> {r_.lam_stress_after:.3f} | {r_.MF_p_before:.3f} -> {r_.MF_p_after:.3f} |")
open(os.path.join(RES,"RESULTS_SUMMARY.md"),"w").write("\n".join(lines))
print("tables + summary written to results/")
print("false-comfort cells:", len(false_comfort.drop_duplicates(['asset','level','model'])))
