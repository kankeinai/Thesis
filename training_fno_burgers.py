"""
Parametric FNO training for 1‑D viscous Burgers’ equation
Author: you
"""

# ----------------------------------------------------------------------
# 0.  imports
# ----------------------------------------------------------------------
import os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_burgers import load_burgers_dataset, custom_collate_fno_fn
from models.fno           import FNO2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 1.  Burgers residual
# ----------------------------------------------------------------------
def compute_residual_burgers(u, f, dx, dt, nu):
    u, f = u.squeeze(-1), f.squeeze(-1)

    dudt   = (u[:, :, 1:] - u[:, :, :-1]) / dt
    dudx   = (u[:, 2:, :-1] - u[:, :-2, :-1]) / (2 * dx)
    d2udx2 = (u[:, 2:, :-1] - 2 * u[:, 1:-1, :-1] + u[:, :-2, :-1]) / dx ** 2

    res = dudt[:, 1:-1] + u[:, 1:-1, :-1] * dudx - nu * d2udx2 - f[:, 1:-1, :-1]
    return res


# ----------------------------------------------------------------------
# 2.  dict‑of‑lists logger
# ----------------------------------------------------------------------
def _log(d, **kwargs):
    for k, v in kwargs.items():
        d.setdefault(k, []).append(v)


# ----------------------------------------------------------------------
# 3.  build FNO2d
# ----------------------------------------------------------------------
def build_model():
    return FNO2d(
        modes1=[24] * 6,
        modes2=[24] * 6,
        width=64,
        layers=[64] * 7,
        in_dim=4,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.05, 0.05],
    ).to(device)


# ----------------------------------------------------------------------
# 4.  main
# ----------------------------------------------------------------------
def main():

    # ----------------------- configuration ---------------------------
    epochs       = 10000
    batch_size   = 64
    lr           = 1e-3
    save_every   = 20

    # ----- early‑stopping parameters --------------------------------
    patience     = 15          # epochs with no improvement
    min_delta    = 1e-5        # minimum change to count as improvement
    best_metric  = float("inf")
    wait         = 0           # counter

    # ----------------------- paths ----------------------------------
    root_dir     = Path("trained_models/burgers")
    root_dir.mkdir(parents=True, exist_ok=True)
    ckpt_pattern = root_dir / "ckpt_ep{:04d}.pt"
    best_ckpt    = root_dir / "best.pt"
    stats_file   = root_dir / "training_stats.pt"

    # ----------------------- data -----------------------------------
    train_ds = load_burgers_dataset("datasets/burgers/train-date-2025-07-25.pt")
    test_ds  = load_burgers_dataset("datasets/burgers/test-date-2025-07-25.pt")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=custom_collate_fno_fn)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              collate_fn=custom_collate_fno_fn)

    # ----------------------- model & optimiser ----------------------
    model = build_model()
    opt   = optim.Adam(model.parameters(), lr=lr)
    mse   = nn.MSELoss()

    # ----------------------- resume (optional) ----------------------
    latest = max(root_dir.glob("ckpt_ep*.pt"), default=None,
                 key=lambda p: p.stat().st_mtime)
    if latest is not None:
        print(f"[resume] loading {latest.name}")
        model.load_state_dict(torch.load(latest, map_location=device))
        if stats_file.exists():
            stats = torch.load(stats_file)
            start_epoch = stats["epoch"][-1] + 1
        else:
            stats, start_epoch = {}, int(latest.stem.split("_ep")[-1]) + 1
    else:
        stats, start_epoch = {}, 1

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",          # lower metric = better
        factor=0.5,          # γ
        patience=5,          # epochs to wait before decay
        threshold=min_delta, # same tolerance as early‑stop
        min_lr=1e-6,
        verbose=True,
    )

    # ----------------------- training loop --------------------------
    for ep in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()

        # ----- train ------------------------------------------------
        model.train()
        b_tot, b_phys, b_data = [], [], []
        for x, y, mask, nus in tqdm(train_loader, desc=f"Epoch {ep:03d}"):
            pred = model(x)
            dx, dt = 1 / (x.shape[1] - 1), 1 / (x.shape[2] - 1)
            phys = compute_residual_burgers(
                pred, x[..., 0:1], dx, dt, nus.exp().view(-1, 1, 1, 1)
            ).pow(2).mean()
            data = mse(pred[mask], y[mask]) if mask.any() else torch.tensor(0.0, device=device)
            ic   = pred[:, :, 0].pow(2).mean()
            loss = phys + data + 10.0 * ic

            opt.zero_grad(); loss.backward(); opt.step()
            b_tot.append(loss.item()); b_phys.append(phys.item()); b_data.append(data.item())

        tr_tot_mu, tr_tot_sd = float(np.mean(b_tot)),  float(np.std(b_tot))
        tr_phy_mu, tr_phy_sd = float(np.mean(b_phys)), float(np.std(b_phys))
        tr_dat_mu, tr_dat_sd = float(np.mean(b_data)), float(np.std(b_data))

        # ----- validate --------------------------------------------
        model.eval()
        te_mse_b, te_phy_b = [], []
        with torch.no_grad():
            for x, y, _, nus in test_loader:
                pred = model(x)
                te_mse_b.append(mse(pred, y).item())
                dx, dt = 1 / (x.shape[1] - 1), 1 / (x.shape[2] - 1)
                te_phy_b.append(
                    compute_residual_burgers(
                        pred, x[..., 0:1], dx, dt, nus.exp().view(-1, 1, 1, 1)
                    ).pow(2).mean().item()
                )

        te_mse_mu, te_mse_sd = float(np.mean(te_mse_b)), float(np.std(te_mse_b))
        te_phy_mu, te_phy_sd = float(np.mean(te_phy_b)), float(np.std(te_phy_b))
        epoch_time = time.perf_counter() - t0

        # ----- log --------------------------------------------------
        _log(stats,
             epoch=ep,
             train_total=tr_tot_mu,  train_total_sd=tr_tot_sd,
             train_phys=tr_phy_mu,   train_phys_sd=tr_phy_sd,
             train_data=tr_dat_mu,   train_data_sd=tr_dat_sd,
             test_mse=te_mse_mu,     test_mse_sd=te_mse_sd,
             test_phys=te_phy_mu,    test_phys_sd=te_phy_sd,
             epoch_time=epoch_time)

        print(
            f"Ep {ep:03d} | "
            f"train μ±σ: total={tr_tot_mu:.3e}±{tr_tot_sd:.1e}, "
            f"phys={tr_phy_mu:.3e}±{tr_phy_sd:.1e}, "
            f"data={tr_dat_mu:.3e}±{tr_dat_sd:.1e} || "
            f"test μ±σ: mse={te_mse_mu:.3e}±{te_mse_sd:.1e}, "
            f"phys={te_phy_mu:.3e}±{te_phy_sd:.1e} | "
            f"t={epoch_time:.1f}s"
        )

        # ----- early stopping --------------------------------------
        improvement = best_metric - te_mse_mu
        if improvement > min_delta:
            best_metric = te_mse_mu
            wait = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f" ⤷ best model updated (val MSE={best_metric:.3e})")
        else:
            wait += 1
            if wait >= patience:
                print(f"[early‑stop] no improvement in {patience} epochs.")
                break

        scheduler.step(te_mse_mu)
        current_lr = opt.param_groups[0]["lr"]
        print(f" ⤷ current LR: {current_lr:.1e}")

        # ----- checkpoint & stats ----------------------------------
        if ep % save_every == 0:
            torch.save(model.state_dict(), ckpt_pattern.as_posix().format(ep))
            torch.save(stats, stats_file)
            print(f" ⤷ checkpoint + stats saved @ epoch {ep}")

    # always save final stats
    torch.save(stats, stats_file)
    print("Training finished.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
