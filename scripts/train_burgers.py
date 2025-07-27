import os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_pde import load_pde1d_dataset, custom_collate_fno1d_fn
from utils.settings import compute_residual_burgers, dirichlet_boundary_loss
from models.fno import FNO1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(d, **kwargs):
    for k, v in kwargs.items():
        d.setdefault(k, []).append(v)

def build_model():
    return FNO1d(
        modes=32,
        width=64,
        in_dim=2,  # u(x), x
        out_dim=100,  # Nt
        depth=5,
        activation="gelu",
        ).to(device)

epochs       = 10000
batch_size   = 128
lr           = 1e-3
save_every   = 20
patience     = 15
min_delta    = 1e-5
best_metric  = float("inf")
wait         = 0

root_dir     = Path("trained_models/burgers1d")
root_dir.mkdir(parents=True, exist_ok=True)
ckpt_pattern = root_dir / "ckpt_ep{:04d}.pt"
best_ckpt    = root_dir / "best.pt"
stats_file   = root_dir / "training_stats.pt"

train_ds = load_pde1d_dataset("datasets/burgers1d/burgers_1d_dataset_train_2025-07-27.h5")
train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=custom_collate_fno1d_fn)

model = build_model()
opt   = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=5, threshold=min_delta, min_lr=1e-6,
)

stats, start_epoch = {}, 1

for ep in range(start_epoch, epochs + 1):
    t0 = time.perf_counter()

    model.train()
    b_tot, b_phys, b_data = [], [], []

    for u, y, mask in tqdm(train_loader, desc=f"Epoch {ep:03d}"):
        u, y, mask = u.to(device), y.to(device), mask.to(device)

        B, Nx, _ = u.shape
        Nt = y.shape[-1]

        u_values = u[..., 0] 
        x_grid   = u[..., 1]  

        pred = model(u_values , x_grid).view(B, Nx, Nt)  # Output: y(x, t)
        
        dx, dt = 1 / (Nx - 1), 1 / (Nt - 1)
        phys = compute_residual_burgers(pred, u_values, dx, dt, nu=0.01).pow(2).mean()
        init_loss = (pred[:, :, 0]**2).mean()  # since target is zero
        bc = dirichlet_boundary_loss(pred)
        loss = phys  + 0.01 * init_loss + 0.01 * bc
        opt.zero_grad(); loss.backward(); opt.step()

        b_tot.append(loss.item()); b_phys.append(phys.item())

    tr_tot_mu, tr_tot_sd = float(np.mean(b_tot)), float(np.std(b_tot))
    tr_phy_mu, tr_phy_sd = float(np.mean(b_phys)), float(np.std(b_phys))
    tr_dat_mu, tr_dat_sd = float(np.mean(b_data)), float(np.std(b_data))

    _log(stats,
            epoch=ep,
            train_total=tr_tot_mu, train_total_sd=tr_tot_sd,
            train_phys=tr_phy_mu, train_phys_sd=tr_phy_sd,
            train_data=tr_dat_mu, train_data_sd=tr_dat_sd)
    print(f"Epoch {ep:03d} ↳ train: total={tr_tot_mu:.3e}±{tr_tot_sd:.1e}, phys={tr_phy_mu:.3e}±{tr_phy_sd:.1e}, data={tr_dat_mu:.3e}±{tr_dat_sd:.1e} (t={time.perf_counter() - t0:.2f}s)")
    
    improvement = best_metric - tr_phy_mu
    if improvement > min_delta:
        best_metric = tr_phy_mu
        wait = 0
        torch.save(model.state_dict(), best_ckpt)
        print(f" ⤷ best model updated (val phys={best_metric:.3e})")
    else:
        wait += 1
        if wait >= patience:
            print(f"[early‑stop] no improvement in {patience} epochs.")
            break

    scheduler.step(tr_phy_mu)

    if ep % save_every == 0:
        torch.save(model.state_dict(), ckpt_pattern.as_posix().format(ep))
        torch.save(stats, stats_file)
        print(f" ⤷ checkpoint + stats saved @ epoch {ep}")

torch.save(stats, stats_file)
print("Training finished.")
