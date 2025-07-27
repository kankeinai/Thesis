import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.data_heat import load_heat1d_dataset, custom_collate_fno1d_fn
from models.fno import FNO1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# 1. Autoencoder definition
# ----------------------------------------------------------------------
class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim=100, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# ----------------------------------------------------------------------
# 2. Training function using streamed y₁ from heat model
# ----------------------------------------------------------------------
def train_autoencoder_streaming(train_loader, heat_model, Nt=100, k=4, epochs=200):
    model = TemporalAutoencoder(input_dim=Nt, latent_dim=k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for u, _, _ in tqdm(train_loader, desc=f"Epoch {ep:03d}"):
            u = u.to(device)
            B, Nx, _ = u.shape
            u_values = u[..., 0]
            x_grid   = u[..., 1]

            with torch.no_grad():
                y1_pred = heat_model(u_values, x_grid).view(B, Nx, Nt)  # [B, Nx, Nt]

            # Flatten: [B*Nx, Nt]
            x = y1_pred.reshape(-1, Nt)

            x_recon, _ = model(x)
            loss = loss_fn(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {ep:03d} | Recon Loss: {np.mean(losses):.4e}")

    torch.save(model.state_dict(), "trained_models/autoencoder_y1.pt")
    return model

# ----------------------------------------------------------------------
# 3. Load heat model + dataset
# ----------------------------------------------------------------------
def main():
    Nt = 100
    k  = 4
    epochs = 100

    train_ds = load_heat1d_dataset("datasets/heat1d/heat_1d_dataset_train_2025-07-26.h5")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=custom_collate_fno1d_fn)

    heat_model = FNO1d(modes=32,
                       width=64,
                       in_dim=2,
                       out_dim=Nt,
                       depth=4,
                       activation="gelu").to(device)

    heat_model.load_state_dict(torch.load("trained_models/heat1d/ckpt_ep0180.pt"))
    heat_model.eval()

    model = train_autoencoder_streaming(train_loader, heat_model, Nt=Nt, k=k, epochs=epochs)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
