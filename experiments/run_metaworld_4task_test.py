import os
import subprocess
import sys
from datetime import datetime

# date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + f"_{os.getpid()}"
date_str = "28_04_2025_18_21_42_1307789_hpt_baseline_metaworld_4task_freezeFalse"
script_name = os.path.basename(__file__).replace(".py", "")
# pretrained = sys.argv[1] if len(sys.argv) > 1 else ""
pretrained = "pretrained/hpt-small"
pretrained_cmd = sys.argv[2] if len(sys.argv) > 2 else ""
postfix = "_hpt_baseline_metaworld_4task_freezeFalse"
print(f"RUNNING {script_name}!")


# train_cmd = (
#     f"HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 time python -m hpt.run "
#     f"script_name={script_name} "
#     f"train.pretrained_dir=output/{pretrained} "
#     f"dataset.episode_cnt=200 "
#     f"domains=metaworld_4task "

#     f"train.freeze_trunk=False "
#     f"output_dir=output/{date_str}_{postfix} "
# )

# print("Executing training command:")
# print(train_cmd)
# os.system(train_cmd)

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




'''

w/ env, agent prototypes, and w/o policy prototypes

success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [05:43<00:00, 11.45s/it]
env_name: ('push-v2',) 30████████████████████████████████████████████████████████████████████████| 30/30 [05:43<00:00,  3.99s/it]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [06:09<00:00, 12.31s/it]
env_name: ('button-press-topdown-v2',) 30████████████████████████████████████████████████████████| 30/30 [06:09<00:00, 11.69s/it]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [01:45<00:00,  3.51s/it]
env_name: ('door-open-v2',) 30███████████████████████████████████████████████████████████████████| 30/30 [01:45<00:00,  3.94s/it]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [02:17<00:00,  4.58s/it]
+-------------------------+----------+███████████████████████████████████████████████████████████| 30/30 [02:17<00:00,  4.94s/it]
| task                    |   reward |
|-------------------------+----------|
| button-press-topdown-v2 |     1.00 |
| door-open-v2            |     1.00 |
| push-v2                 |     0.57 |
| reach-v2                |     0.60 |
| avg                     |     0.79 |
+-------------------------+----------+




w/ env, agent, and policy prototypes
                                                                                               Found 4 GPUs for rendering. Using device 1.                                                                | 0/30 [00:00<?, ?it/s]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [05:41<00:00, 11.38s/it]
env_name: ('push-v2',) 30████████████████████████████████████████████████████████████████████████| 30/30 [05:41<00:00,  4.90s/it]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [08:36<00:00, 17.21s/it]
env_name: ('button-press-topdown-v2',) 30████████████████████████████████████████████████████████| 30/30 [08:36<00:00, 13.36s/it]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [02:10<00:00,  4.35s/it]
env_name: ('door-open-v2',) 30███████████████████████████████████████████████████████████████████| 30/30 [02:10<00:00,  4.14s/it]
success: 1.0: 100%|██████████████████████████████████████████████████████████████████████████████| 30/30 [02:51<00:00,  5.71s/it]
+-------------------------+----------+███████████████████████████████████████████████████████████| 30/30 [02:51<00:00,  5.71s/it]
| task                    |   reward |
|-------------------------+----------|
| button-press-topdown-v2 |     1.00 |
| door-open-v2            |     1.00 |
| push-v2                 |     0.57 |
| reach-v2                |     0.73 |
| avg                     |     0.82 |
+-------------------------+----------+

'''