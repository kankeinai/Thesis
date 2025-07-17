from models.lno import LNO1d
from models.fno import FNO1d
import torch



modes = 32
width = 64
model = FNO1d(modes=32, width=64, depth=4, activation="silu", hidden_layer = 128)
# 2) Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ckpt = torch.load("trained_models/singular_arc/fno/attempt_started20250715_182204/epoch[40]_model_time_[20250715-182628]_loss_[0.0460].pth", map_location=device, weights_only=False)

model.load_state_dict(ckpt["model_state_dict"])

# 5) Switch to eval mode
model.eval()