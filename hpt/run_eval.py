# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
import torch

from hpt.utils import utils
from hpt import train_test
import numpy as np


@hydra.main(config_path="../experiments/configs", config_name="config", version_base="1.2")
def run_eval(cfg):
    """
    This script runs through the eval loop. It loads the model and run throug hthe rollout functions.
    """
    cfg.output_dir = cfg.output_dir + "/" + str(cfg.seed)
    utils.set_seed(cfg.seed)
    print(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0]
    
    # 初始化数据集以获取 action_dim
    print("初始化数据集...")
    dataset = hydra.utils.instantiate(
        cfg.dataset, dataset_name=domain, env_rollout_fn=cfg.dataset_generator_func, **cfg.dataset
    )
    
    # initialize policy
    policy = hydra.utils.instantiate(cfg.network).to(device)
    
    # 更新网络维度
    from hpt.utils.utils import update_network_dim
    update_network_dim(cfg, dataset, policy)
    
    policy.init_domain_stem(domain, cfg.stem)
    normalizer = dataset.get_normalizer()
    policy.init_domain_head(domain, normalizer, cfg.head)
    policy.finalize_modules()
    policy.print_model_stats()
    utils.set_seed(cfg.seed)

    # add encoders into policy parameters
    if cfg.network.finetune_encoder:
        utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), cfg.dataset.image_encoder)
        from hpt.utils.utils import global_vision_model
        policy.init_encoders("image", global_vision_model)

    # load the model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 优先加载训练好的完整模型，其次加载预训练 trunk
    trained_model_path = os.path.join(project_root, cfg.train.pretrained_dir, "model.pth")
    trunk_path = os.path.join(project_root, cfg.train.pretrained_dir, "trunk.pth")
    
    if os.path.exists(trained_model_path):
        print(f"Loading trained full model from: {trained_model_path}")
        policy.load_model(trained_model_path)
        print("Loaded complete model (trunk + stem + head).")
    elif os.path.exists(trunk_path):
        print(f"Loading pretrained trunk from: {trunk_path}")
        policy.load_trunk(trunk_path)
        print("Note: Only trunk loaded. Stem and head are randomly initialized.")
    else:
        print("Warning: No pretrained model found. Using random initialization.")
    policy.to(device)
    policy.eval()
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")
    
    # evaluate jointly trained policy
    total_rewards = train_test.eval_policy_sequential(policy, cfg)

    # save the results
    utils.log_results(cfg, total_rewards)


if __name__ == "__main__":
    run_eval()
