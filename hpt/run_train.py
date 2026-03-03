import os
import sys

os.environ["PYTHONPATH"] = r"d:\scientific_research\code\MoDP-Demo\MoDP-Demo"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_API_KEY"] = "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"

sys.path.insert(0, r"d:\scientific_research\code\MoDP-Demo\MoDP-Demo")

import subprocess
subprocess.run([
    "python", "run.py",
    "+tasks=metaworld_20task",
    "++train.pretrained_dir=pretrained/hpt-small",
    "++dataset.episode_cnt=200",
    "++dataset.regenerate=False",
    "++train.total_epochs=50",
    "++train.total_iters=5000",
    "++train.freeze_trunk=False",
    "++dataloader.batch_size=64",
    "++dataloader.num_workers=4",
    "script_name=DyCoP_train"
], cwd=r"d:\scientific_research\code\MoDP-Demo\MoDP-Demo\hpt")
