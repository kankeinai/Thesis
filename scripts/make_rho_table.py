import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1.  Load the summary CSV produced by rho_sensitivity_study.py
# ------------------------------------------------------------------
summary = pd.read_csv("results/rho_sensitivity_summary.csv")

# ------------------------------------------------------------------
# 2.  Choose the metric you want to show in the table
#     (here: relative state‑error at T, mean ± std)
# ------------------------------------------------------------------
def fmt(mean, std):
    """Return 'mean ± std' with 3 sig‑fig mean, 2‑sig std."""
    return f"{mean:.3e} ± {std:.1e}"

summary["cell"] = summary.apply(
    lambda r: fmt(r.rel_err_u_mean, r.rel_err_u_std), axis=1
)

# ------------------------------------------------------------------
# 3.  Pivot to wide format: rows = benchmark·arch, cols = ρ
# ------------------------------------------------------------------
pretty = (
    summary
      .pivot(index=["benchmark", "arch"],
             columns="rho",
             values="cell")
      .rename(columns={0.1: r"$\\rho=0.1$",
                       1.0: r"$\\rho=1$",
                       10.0: r"$\\rho=10$"})
)

# ------------------------------------------------------------------
# 4.  Write LaTeX table body (no wrapper) to file
# ------------------------------------------------------------------
Path("tables").mkdir(exist_ok=True)
latex_path = Path("tables/pilot_rho_landscape.tex")
pretty.to_latex(
    latex_path,
    column_format="llccc",  # 'l' benchmark | 'l' arch | 3 numeric cols
    escape=False,           # keep the math $\\rho=...$
    multicolumn_format="c",
    bold_rows=False
)
print(f"LaTeX table body written → {latex_path}")