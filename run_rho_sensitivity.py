# rho_sensitivity_study.py
"""
Pilot sensitivity study on control weight ρ (R/Q ratio)
across all architectures and benchmarks.

Assumes:
• Trained models exist and can be loaded by architecture name.
• solve_optimization() returns analytics, x_pred, u_pred, x_opt_fn, u_opt_fn
• Control effort weight is w[1] = ρ (second term of w-vector)
• Time grid is fixed: N = 200 (Δt = 0.005)

Outputs:
• results/rho_sensitivity.csv — all 135 runs
• results/rho_sensitivity_summary.csv — group means ± std
"""

import itertools, time
from pathlib import Path
from typing import List, Dict
import torch
import pandas as pd
import numpy as np
from utils.scripts import load_pretrained_model, generate_weights, solve_optimization
from utils.settings import boundaries

# ----------------------- Constants ------------------------ #

BENCHMARKS     = ["linear", "oscillatory", "polynomial_tracking", "nonlinear", "singular_arc"]
ARCHITECTURES  = ["deeponet", "fno", "lno"]
RHOS           = [0.1, 1.0, 10.0]
SEEDS          = [42, 43, 44]
HORIZON_STEPS  = 200
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Main Loop ------------------------ #

records: List[Dict] = []

for bench, arch, rho, seed in itertools.product(BENCHMARKS, ARCHITECTURES, RHOS, SEEDS):
    print(f"[{bench} | {arch} | ρ={rho} | seed={seed}]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = load_pretrained_model(bench, arch, device=DEVICE)
    model.eval()

    t_grid = np.linspace(0, 1, HORIZON_STEPS)

    if bench == 'singular_arc':
        initial_guess = torch.rand((1, HORIZON_STEPS), dtype=torch.float32, device=DEVICE, requires_grad=True)-3
    else:
        initial_guess = torch.rand((1, HORIZON_STEPS), dtype=torch.float32, device=DEVICE, requires_grad=True)

    w_vec = generate_weights(rho, bench, arch)
    tic = time.time()

    print(f"Solving {bench} with {arch} and rho={rho} and seed={seed}")

    analytics, x_pred, u_pred, x_opt_fn, u_opt_fn = solve_optimization(
        model=model,
        problem=bench,
        initial_guess=initial_guess,
        lr=1e-3,
        architecture=arch,
        w=w_vec,
        num_epochs=50000,
        m=HORIZON_STEPS,
        bounds=boundaries[bench],
        device=DEVICE,
        result=True,
        early_stopping=True,
        patience=1000,
        plots=False
    )

    toc = time.time()

    rel_err_u = float(analytics['rel_err_u'][-1])
    rel_err_x = float(analytics['rel_err_x'][-1])

    # Store summary metrics
    records.append(dict(
        benchmark = bench,
        arch = arch,
        rho = rho,
        seed = seed,
        rel_err_u = rel_err_u,
        rel_err_x = rel_err_x,
        obj = analytics['obj'][-1],
        stop_epoch = len(analytics['obj']),
        solve_s = toc - tic
    ))

# ----------------------- Save Results --------------------- #

Path("results").mkdir(exist_ok=True)
raw_df = pd.DataFrame(records)
raw_df.to_csv("results/rho_sensitivity.csv", index=False)

summary_df = (
    raw_df.groupby(["benchmark", "arch", "rho"], as_index=False)
    .agg(
        rel_err_x_mean=("rel_err_x", "mean"),
        rel_err_x_std=("rel_err_x", "std"),
        rel_err_u_mean=("rel_err_u", "mean"),
        rel_err_u_std=("rel_err_u", "std"),
        obj_mean=("obj", "mean"),
        obj_std=("obj", "std"),
        stop_epoch_mean=("stop_epoch", "mean"),
        solve_s_mean=("solve_s", "mean")
    )
)

summary_df.to_csv("results/rho_sensitivity_summary.csv", index=False)

print("Saved 135 raw runs → results/rho_sensitivity.csv")
print("Saved 45-group summary → results/rho_sensitivity_summary.csv")
