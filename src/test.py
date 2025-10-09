from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

w = SummaryWriter(log_dir="../results/sft_results/runs/debug-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
w.add_scalar("debug/value", 1.0, 0)
w.close()
