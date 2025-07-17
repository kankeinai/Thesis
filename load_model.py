from models.lno import LNO1d
from models.fno import FNO1d
import torch


model = LNO1d(
    width=4,
    modes=16,
    hidden_layer=128,
)

# 2) Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


ckpt = torch.load(
    "trained_models/linear/lno/attempt_started20250715_113835/epoch[80]_model_time_[20250715-120337]_loss_[0.0032].pth",
    map_location=device,
    weights_only=False
)

# if it’s a dict with model_state_dict:
if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
else:
    state = ckpt

# 4) Switch to evaluation mode (disables dropout, etc.)
model.eval()

model = LNO1d(
    width=4,
    modes=16,
    hidden_layer=128,
)

# 2) Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


ckpt = torch.load(
    "trained_models/linear/lno/attempt_started20250715_113835/epoch[80]_model_time_[20250715-120337]_loss_[0.0032].pth",
    map_location=device,
    weights_only=False
)

# if it’s a dict with model_state_dict:
if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
else:
    state = ckpt

# 4) Switch to evaluation mode (disables dropout, etc.)
model.eval()

model = FNO1d(modes=32, width=64, depth=4, activation="silu", hidden_layer = 128).to(device)

# 2) Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


ckpt = torch.load(
    "trained_models/linear/fno/attempt_started20250715_121249/epoch[40]_model_time_[20250715-122503]_loss_[0.0003].pth",
    map_location=device,
    weights_only=False
)

# if it’s a dict with model_state_dict:
if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
else:
    state = ckpt

# 4) Switch to evaluation mode (disables dropout, etc.)
model.eval()
