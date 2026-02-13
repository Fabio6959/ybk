import os
import subprocess
import sys
from datetime import datetime

date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + f"_{os.getpid()}"
script_name = os.path.basename(__file__).replace(".py", "")
# pretrained = sys.argv[1] if len(sys.argv) > 1 else ""
pretrained = "pretrained/hpt-small"
pretrained_cmd = sys.argv[2] if len(sys.argv) > 2 else ""
postfix = "_hpt_baseline_metaworld_4task_freezeFalse"
print(f"RUNNING {script_name}!")


train_cmd = (
    f"HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 time python -m hpt.run "
    f"script_name={script_name} "
    f"train.pretrained_dir={pretrained} "
    f"dataset.episode_cnt=200 "
    f"domains=metaworld_4task "
    f"+tasks=metaworld_4task "
    f"train.freeze_trunk=False "
    f"output_dir=output/{date_str}{postfix} "
)

print("Executing training command:")
print(train_cmd)
os.system(train_cmd)

eval_cmd = (
    f"HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 time python -m hpt.run_eval --multirun "
    f"--config-name=config "
    f"--config-path=../output/{date_str}{pretrained_cmd} "
    f"train.pretrained_dir='output/{date_str}{pretrained_cmd}' "
    # f"seed=range(3) "
    f"hydra.sweep.dir=output/{date_str}{pretrained_cmd} "
    f"hydra/launcher=joblib "
    f"hydra.launcher.n_jobs=3"
)

print("Executing evaluation command:")
print(eval_cmd)
os.system(eval_cmd)
