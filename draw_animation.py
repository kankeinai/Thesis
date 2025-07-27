import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import random

def animate_solution_gif(model, test_ds, save_path="fno_heat_prediction.gif"):
    """
    Animate model prediction vs ground truth on a random test sample.
    Left: static control input u(x)
    Right: animated prediction and ground truth y(x, t)
    """

    model.eval()

    idx = random.randint(0, len(test_ds) - 1)
    u_raw, _, _ = test_ds[idx]  # u_raw: [Nx], y_true: [Nx, Nt]
   
    Nx, Nt = (64,100)
    x = torch.linspace(0, 1, Nx).view(1, Nx).to(device)
    u = u_raw.view(1, Nx).to(device)
    
    with torch.no_grad():
        y_pred = model(u,x).view(1, Nx, Nt)

    # Move to CPU
    y_pred = y_pred.squeeze().cpu().numpy()
    u_np   = u.squeeze().cpu().numpy()
    x_np   = np.linspace(0, 1, Nx)

    # Set up subplots
    fig, (ax_u, ax_sol) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 2]})

    # Plot u(x)
    ax_u.plot(x_np, u_np, color='green')
    ax_u.set_title("Control Input $u(x)$")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u(x)")
    ax_u.grid(True)

    # Set up animation axes
    pred_line, = ax_sol.plot([], [], label="Prediction", color="orange")
    ax_sol.set_xlim(0, 1)
    ax_sol.set_xlabel("x")
    ax_sol.set_ylabel("y(x,t)")
    ax_sol.set_title("FNO Burger Equation Prediction")
    ax_sol.legend()

    def init():
    
        pred_line.set_data([], [])
        return  pred_line

    def update(t):
    
        pred_line.set_data(x_np, y_pred[:, t])
        ax_sol.set_title(f"FNO Prediction at t={t / (Nt - 1):.2f}")
        return pred_line

    ani = animation.FuncAnimation(
        fig, update, frames=Nt, init_func=init, blit=True, interval=80
    )

    ani.save(save_path, writer="pillow", fps=12)
    print(f"✅ Saved animation to {save_path}")
    plt.close()

from utils.data_heat import HeatEquation1DControlDataset, load_heat1d_dataset, custom_collate_fno1d_fn
from models.fno import FNO1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO1d(
        modes=32,
        width=64,
        depth=5,
        out_dim=100,    # Nt output points
        activation='gelu'
    ).to(device)

model.load_state_dict(torch.load("trained_models/burgers1d/best.pt"))
model.eval()

test_ds = load_heat1d_dataset("datasets/heat1d/heat_1d_dataset_train_2025-07-26.h5")

animate_solution_gif(model, test_ds, save_path="fno_burger_prediction.gif")

