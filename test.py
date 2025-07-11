from utils.scripts import cubic_spline_interp
import torch
x = torch.tensor([0.0, 0.1, 0.05, 0.9, 0.8])
t = torch.linspace(0, 1, 5)
S, dS_dt = cubic_spline_interp(x.unsqueeze(0), t)
print("S:", S)
print("dS_dt:", dS_dt)