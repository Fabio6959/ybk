# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
from functools import partial
import hydra
from omegaconf import OmegaConf
from hpt.utils.utils import download_from_huggingface
import os
from typing import List, Optional

import torch
import torch.nn as nn
from hpt.utils.normalizer import LinearNormalizer
from hpt.models.transformer import MultiheadAttention, SimpleTransformer
from typing import List
import numpy as np
import einops
from collections import defaultdict
from sklearn.cluster import KMeans
import torch.nn.functional as F


from hpt.utils.utils import (
    dict_apply,
    get_sinusoid_encoding_table,
    EinOpsRearrange,
    get_image_embeddings,
    normalize_image_numpy,
    get_t5_embeddings,
)

STD_SCALE = 0.02


def compute_ortho_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Compute orthogonal loss for prototypes to prevent mode collapse.
    
    Args:
        prototypes: [num_protos, embed_dim] or [num_protos, T, embed_dim]
    
    Returns:
        loss: orthogonal regularization loss
    """
    if prototypes.dim() == 3:
        # [num_protos, T, embed_dim] -> [num_protos, T*embed_dim]
        prototypes_flat = prototypes.view(prototypes.shape[0], -1)
    else:
        prototypes_flat = prototypes
    
    # L2 normalize each prototype
    prototypes_norm = F.normalize(prototypes_flat, p=2, dim=-1)  # [num_protos, dim]
    
    # Compute similarity matrix [num_protos, num_protos]
    sim_matrix = torch.matmul(prototypes_norm, prototypes_norm.T)  # [num_protos, num_protos]
    
    # Identity matrix
    identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    
    # Frobenius norm squared
    loss = F.mse_loss(sim_matrix, identity, reduction='sum')
    
    return loss


class TaskEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_task_protos=21):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_task_protos = num_task_protos
        
        # 形状 [21, 128]
        self.task_prototypes = nn.Parameter(torch.randn(self.num_task_protos, self.embed_dim))
        
        # 将 128维映射到 CLIP 文本特征的 512维
        self.semantic_proj = nn.Linear(self.embed_dim, 512)

    def forward(self, text_features):
        # 计算路由权重
        w_t = F.softmax(torch.matmul(text_features, self.task_prototypes.T), dim=-1)
        return w_t


class MoLoRALayer(nn.Module):
    """
    Mixture of LoRAs (MoLoRA) for efficient fine-tuning.
    Replaces nn.Linear layers with low-rank adapters weighted by routing scores.
    """
    def __init__(self, in_features: int, out_features: int, num_combinations: int, r: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_combinations = num_combinations
        self.r = r
        
        # LoRA matrices: [num_combinations, in_features, r]
        self.lora_A = nn.Parameter(torch.randn(num_combinations, in_features, r) * 0.02)
        # LoRA matrices: [num_combinations, r, out_features]
        self.lora_B = nn.Parameter(torch.randn(num_combinations, r, out_features) * 0.02)
        
        # Initialize as identity for small initial effect
        with torch.no_grad():
            self.lora_B[:, :, :] = 0
        
        # Cache for routing weights (set externally by Policy)
        self._current_w_route = None
        
        # Base layer weight and bias (will be set by _replace_linear_layer)
        self.weight = None
        self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with soft routing.
        Uses weight blending approach: first blend LoRA weights, then apply to input.
        
        Args:
            x: [B, in_features] or [B, T, in_features] input tensor
        
        Returns:
            y: [B, out_features] or [B, T, out_features] output tensor
        """
        # 1. 获取缓存的路由权重
        w = getattr(self, "_current_w_route", None)
        B_x = x.shape[0]  # 当前实际流入的数据 Batch Size
        
        if w is None:
            N = self.lora_A.shape[0]
            w = torch.ones(B_x, N, device=x.device, dtype=x.dtype) / N
        else:
            # 关键修复：确保 w 的 batch size 与 x 对齐！
            B_w = w.shape[0]
            if B_w != B_x:
                # 如果缓存的 w 比当前数据大，说明数据被截断了，我们切片取前 B_x 个
                # （这在 HPT/MoDP 的特征分流/重构过程中非常常见）
                if B_w > B_x:
                    w = w[:B_x, :]
                # 如果意外出现了 w 比 x 小的情况，做 padding 或 repeat 兜底
                else:
                    repeats = (B_x + B_w - 1) // B_w
                    w = w.repeat(repeats, 1)[:B_x, :]

        # 2. 动态加权融合该 Batch 的 LoRA 专家矩阵
        # w shape: [B_x, N_total]
        blended_A = torch.einsum('bn, nir -> bir', w, self.lora_A)
        blended_B = torch.einsum('bn, nro -> bro', w, self.lora_B)

        # 3. 计算 LoRA 输出
        if x.dim() == 3:
            lora_mid = torch.einsum('bti, bir -> btr', x, blended_A)
            lora_out = torch.einsum('btr, bro -> bto', lora_mid, blended_B)
        elif x.dim() == 2:
            lora_mid = torch.einsum('bi, bir -> br', x, blended_A)
            lora_out = torch.einsum('br, bro -> bo', lora_mid, blended_B)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # 4. 计算 Base 层的输出并相加
        if self.weight is not None:
            base_out = F.linear(x, self.weight, self.bias)
            return base_out + lora_out
        else:
            # If no base layer (standalone MoLoRA), return only LoRA output
            return lora_out


class Policy(nn.Module):
    """The HPT Policy class.
    Usage for pretraining:
    >>> policy = Policy.from_pretrained("hf://liruiw/hpt-xlarge")
    >>> output = policy.get_action(data_dict)

    Usage for inference:
    >>> policy = Policy.from_pretrained_full_model("hf://liruiw/hpt-xlarge")
    See full usage examples in run.py.

    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        use_modality_embedding: bool = True,
        token_postprocessing: bool = False,
        observation_horizon: int = 4,
        action_horizon: int = 1,
        no_trunk: bool = False,
        shared_modality_trunk: Optional[nn.Module] = None,
        use_domain_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_modality_trunk = shared_modality_trunk
        self.no_trunk = no_trunk

        self.trunk = self._create_policy_trunk(embed_dim, num_blocks, num_heads, **kwargs)
        self.stems = {}
        self.heads = {}
        self.normalizer = {}
        self.encoders = {}
        self.domains = []
        self.use_modality_embedding = use_modality_embedding
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.token_postprocessing = token_postprocessing
        self.modalities_tokens = {}
        self.action_tokens = {}

        # agent/env/task prototype 
        self.prototype_num = 6
        self.prototype_dim = 64
        self.num_task_protos = 21
        self.lora_r = 64  # LoRA rank

        # self.agent_head = nn.Linear(32 * embed_dim, embed_dim)
        self.agent_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.prototype_dim)
        )
        # self.env_head = nn.Linear(32 * embed_dim, embed_dim)
        self.env_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.prototype_dim)
        )
        
        # Task encoder for task intentions
        self.text_dim = 512  # Default text embedding dimension
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.prototype_dim)
        )
        
        # Task encoder for processing text features
        self.task_encoder = TaskEncoder(
            embed_dim=embed_dim,
            num_task_protos=self.num_task_protos
        )
        
        # self.decoder = nn.Linear(2 * embed_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(3 * self.prototype_dim, embed_dim)
        )

        # 重构为 2D 纯语义特征向量，彻底解耦序列长度
        self.register_buffer("agent_prototypes", torch.randn(self.prototype_num, self.embed_dim))
        self.register_buffer("env_prototypes", torch.randn(self.prototype_num, self.embed_dim))

        self.prototype_momentum = 0.9
        
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        
        # MoLoRA layers for blocks.15.mlp.fc1 and blocks.15.mlp.fc2
        # Total combinations: N_a * N_e * N_t = 6 * 6 * 6 = 216
        self.num_combinations = self.prototype_num * self.prototype_num * self.num_task_protos
        self.molora_layers = nn.ModuleDict()
        self._init_molora_layers()
        
        # Store references to original layers for replacement
        self.original_fc1 = None
        self.original_fc2 = None
        
        # Routing weights storage
        self.current_w_a = None
        self.current_w_e = None
        self.current_w_t = None
        
        # === 语义对齐新增：注册 CLIP 文本锚点 ===
        # 模拟 metaworld 21个任务的 CLIP text embeddings (512维)
        dummy_clip_features = torch.randn(21, 512)
        dummy_clip_features = F.normalize(dummy_clip_features, p=2, dim=-1)  # L2归一化
        self.register_buffer("task_clip_anchors", dummy_clip_features)

    def _init_molora_layers(self):
        """Initialize MoLoRA layers and directly replace target layers in trunk."""
        # Get dimensions from trunk and replace layers
        for name, module in self.trunk["trunk"].named_modules():
            if isinstance(module, nn.Linear):
                if "blocks.15.mlp.fc1" in name:
                    in_features = module.in_features
                    out_features = module.out_features
                    
                    # Create MoLoRA layer
                    molora_fc1 = MoLoRALayer(
                        in_features=in_features,
                        out_features=out_features,
                        num_combinations=self.num_combinations,
                        r=self.lora_r
                    )
                    
                    # Directly replace the original layer
                    self._replace_linear_layer(module, molora_fc1, name)
                    
                    # Store reference for routing weight assignment
                    self.molora_layers["fc1"] = molora_fc1
                    break
        
        for name, module in self.trunk["trunk"].named_modules():
            if isinstance(module, nn.Linear):
                if "blocks.15.mlp.fc2" in name:
                    in_features = module.in_features
                    out_features = module.out_features
                    
                    # Create MoLoRA layer
                    molora_fc2 = MoLoRALayer(
                        in_features=in_features,
                        out_features=out_features,
                        num_combinations=self.num_combinations,
                        r=self.lora_r
                    )
                    
                    # Directly replace the original layer
                    self._replace_linear_layer(module, molora_fc2, name)
                    
                    # Store reference for routing weight assignment
                    self.molora_layers["fc2"] = molora_fc2
                    break
        
        # Freeze trunk parameters (excluding MoLoRA layers)
        self._freeze_trunk_parameters()
    
    def _replace_linear_layer(self, original_layer: nn.Linear, new_layer: nn.Module, layer_name: str):
        """
        Replace a nn.Linear layer with a new layer (MoLoRA).
        This modifies the module in-place to avoid breaking the module tree.
        """
        # Copy all attributes from original to new layer
        for attr_name, attr_value in original_layer.__dict__.items():
            if not attr_name.startswith('_'):
                setattr(new_layer, attr_name, attr_value)
        
        # Find parent module and replace
        parent = None
        child_name = None
        for parent_name, parent_module in self.trunk["trunk"].named_modules():
            if parent_module == original_layer:
                continue
            for child_name_candidate, child_module in parent_module.named_children():
                if child_module is original_layer:
                    parent = parent_module
                    child_name = child_name_candidate
                    break
            if parent is not None:
                break
        
        if parent is not None and child_name is not None:
            setattr(parent, child_name, new_layer)
            print(f"Replaced layer '{layer_name}' with MoLoRA layer")
        else:
            print(f"Warning: Could not replace layer '{layer_name}'")
    
    def _freeze_trunk_parameters(self):
        """Freeze all trunk parameters for MoLoRA fine-tuning."""
        for param in self.trunk.parameters():
            param.requires_grad = False
    
    def init_encoders(self, modality, encoder):
        """
        Add image/language encoders into the policy parameters in the case of joint finetuning
        """
        self.encoders[modality] = encoder
        self.encoders = nn.ModuleDict(self.encoders)

    def init_domain_stem(self, domain_name, stem_spec):
        """
        Initialize an observation stem for each domain
        """
        self.stem_spec = stem_spec
        self.modalities = stem_spec.modalities
        for modality in self.modalities:
            self.stems[domain_name + "_" + modality] = hydra.utils.instantiate(getattr(stem_spec, modality))
            self.stems[domain_name + "_" + modality].init_cross_attn(stem_spec, modality)
            self.modalities_tokens[modality] = nn.Parameter(torch.randn(1, 1, stem_spec.modality_embed_dim) * STD_SCALE)

        if self.token_postprocessing == "action_token":
            self.action_tokens[domain_name] = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * STD_SCALE
            )

    def init_domain_head(self, domain_name, normalizer=None, head_spec=None):
        """initialize an action head for each domain, along with normalizer"""
        self.head_spec = head_spec
        self.domains.append(domain_name)
        self.heads[domain_name] = hydra.utils.instantiate(head_spec)
        self.normalizer[domain_name] = LinearNormalizer()

        if normalizer is not None:
            self.normalizer[domain_name].load_state_dict(normalizer.state_dict())

    def finalize_modules(self):
        """
        Finalizes the modules of the policy.

        This method converts the stems, heads, normalizer, modalities_tokens
        attentive_pool, and action_tokens into ModuleDict or ParameterDict objects, depending
        on the configuration. It also initializes the weights of the policy.
        """
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)
        self.normalizer = nn.ModuleDict(self.normalizer)
        self.modalities_tokens = nn.ParameterDict(self.modalities_tokens)

        self.apply(self._init_weights)
        if self.token_postprocessing == "action_token":
            self.action_tokens = nn.ParameterDict(self.action_tokens)

    def _create_policy_trunk(self, embed_dim: int = 1024, num_blocks: int = 24,
                num_heads: int = 16, drop_path: float = 0.0,
                weight_init_style: str = "pytorch", **kwargs
            ):
        """create the shared representation for pretraining"""

        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
            )

        trunk = {}
        trunk["trunk"] = instantiate_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )
        if hasattr(self, "shared_modality_trunk") and self.shared_modality_trunk is not None:
            for modality in self.shared_modality_trunk.modalities:
                trunk[modality] = self.shared_modality_trunk[modality]

        return nn.ModuleDict(trunk)

    def get_position_embedding(self, feature: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """
        Add positional embedding to the features
        """
        tokensize = int(feature.shape[1])
        tokens = get_sinusoid_encoding_table(0, tokensize, self.embed_dim)
        return tokens.repeat((1, 1, 1)).to(feature.device)


    def update_masks_with_weights(self, w_a, w_e):
        device = next(self.parameters()).device
        w_a = w_a.to(device)
        w_e = w_e.to(device)
        w_a_mean = w_a.mean(dim=0)  # [P]
        w_e_mean = w_e.mean(dim=0)  # [P]

        for a_idx in range(self.prototype_num):
            for e_idx in range(self.prototype_num):
                importance = w_a_mean[a_idx] * w_e_mean[e_idx]
                if importance > 0.01:
                    mask_key = f"{a_idx}_{e_idx}"
                    masks = self.masks[mask_key]
                    
                    mask_idx = 0
                    for name, module in self.trunk["trunk"].named_modules():
                        if isinstance(module, nn.Linear):
                            if "blocks.15.mlp.fc2" in name:
                                if module.weight.grad is not None:
                                    importance_score = torch.abs(module.weight.grad * module.
                                    weight)
                                    new_mask = (importance_score > importance_score.mean()).float()
                                    if not masks[mask_idx].device == device:
                                        masks[mask_idx] = masks[mask_idx].to(device)
                                    masks[mask_idx] = (1 - importance) * masks[mask_idx] + importance * new_mask
                                mask_idx += 1
    
    def feature2proto(self, tokens, text_features=None):
        B, L, D = tokens.shape
        
        pooled_tokens = tokens.mean(dim=1)
        
        task_protos = self.task_encoder.task_prototypes
        agent_protos = self.agent_prototypes.clone()
        env_protos = self.env_prototypes.clone()
        
        if self.training:
            self.global_step += 1
        
        progress = min(1.0, self.global_step.item() / 20000.0)
        tau = 1.0 - 0.9 * progress
        
        pooled_tokens_norm = F.normalize(pooled_tokens, p=2, dim=-1)
        task_protos_norm = F.normalize(task_protos, p=2, dim=-1)
        agent_protos_norm = F.normalize(agent_protos, p=2, dim=-1)
        env_protos_norm = F.normalize(env_protos, p=2, dim=-1)
        
        w_t = torch.softmax(torch.matmul(pooled_tokens_norm, task_protos_norm.T) / tau, dim=-1)
        w_a = torch.softmax(torch.matmul(pooled_tokens_norm, agent_protos_norm.T) / tau, dim=-1)
        w_e = torch.softmax(torch.matmul(pooled_tokens_norm, env_protos_norm.T) / tau, dim=-1)
        
        w_route = torch.einsum('bi, bj, bk -> bijk', w_a, w_e, w_t).reshape(B, -1)
        
        combined_features = torch.matmul(w_a, agent_protos) + torch.matmul(w_e, env_protos) + torch.matmul(w_t, task_protos)
        
        proto_tokens = combined_features.unsqueeze(1).expand(-1, L, -1)
        
        self.current_w_a = w_a.detach()
        self.current_w_e = w_e.detach()
        self.current_w_t = w_t.detach()
        
        with torch.no_grad():
            self.agent_prototypes.copy_(
                self._update_prototypes(self.agent_prototypes, pooled_tokens.detach(), w_a.detach())
            )
            self.env_prototypes.copy_(
                self._update_prototypes(self.env_prototypes, pooled_tokens.detach(), w_e.detach())
            )
        
        trunk_tokens_rec = proto_tokens
        
        return trunk_tokens_rec, w_route

    def preprocess_tokens(self, domain: str, features: List[torch.Tensor], text_features: torch.Tensor = None) -> torch.Tensor:
        """
        Shared modality layers and add modality tokens. Add positional and time embeddings.
        """
        tokens = torch.cat(features, dim=-2)
        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens[domain].repeat(len(tokens), 1, 1)
            tokens = torch.cat([tokens, action_tokens], dim=-2)
        
        ori_tokens = tokens
        proto_tokens, w_route = self.feature2proto(tokens, text_features)
        position_tokens = self.get_position_embedding(proto_tokens, self.embed_dim)
        tokens = tokens + position_tokens
        # proto_tokens = self.feature2proto(tokens)
        return tokens, ori_tokens, proto_tokens, w_route
    
    def _init_prototypes(self, pooled_features: torch.Tensor, num_proto: int):
        """Initialize prototypes using KMeans or random initialization.

        Args:
            pooled_features: [B, embed_dim]
            Returns: [num_proto, embed_dim]
        """
        if pooled_features.shape[0] >= num_proto:
            features_np = pooled_features.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=num_proto, random_state=0, n_init='auto').fit(features_np)
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        else:
            centers = torch.randn(num_proto, pooled_features.shape[1]) * 0.02
        
        return centers.to(pooled_features.device)

    def _update_prototypes(self, prototypes: torch.Tensor, pooled_features: torch.Tensor, weights: torch.Tensor):
        """Update prototypes using momentum EMA.

        Args:
            prototypes: [P, embed_dim]
            pooled_features: [B, embed_dim] from Mean Pooling
            weights: [B, P] routing weights
        """
        P = prototypes.shape[0]
        
        norm_weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)
        
        proto_update = torch.matmul(norm_weights.T, pooled_features)
        
        updated = self.prototype_momentum * prototypes + (1 - self.prototype_momentum) * proto_update
        updated = F.normalize(updated, p=2, dim=-1)
        return updated


    def postprocess_tokens(self, trunk_tokens: torch.Tensor) -> torch.Tensor:
        """
        Postprocesses the trunk tokens to obtain the final features.

        Args:
            trunk_tokens (torch.Tensor): The trunk tokens of shape (N, L, D), where N is the batch size,
                                        L is the sequence length, and D is the token dimension.

        Returns:
            torch.Tensor: The postprocessed tokens of shape (N, D), where N is the batch size and D is the
                          final feature dimension.
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "action_token":
            return trunk_tokens[:, -self.action_horizon :]
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)[0]
        elif self.token_postprocessing == "last":
            return trunk_tokens[:, -1]
        elif self.token_postprocessing == "attentive":
            return self.attentive_pool(trunk_tokens)[:, 0]
        elif self.token_postprocessing == "no-op":
            # recommended for transformer decoder
            return trunk_tokens
        elif self.token_postprocessing == "proto":
            trunk_tokens_rec = trunk_tokens.mean(dim=1)
            return trunk_tokens_rec
        else:
            raise ValueError("Invalid token_postprocessing value. Must be one of ['mean', 'action_token', 'max', 'last', 'attentive'].")

    def preprocess_states(self, domain: str, data: dict) -> dict:
        """
        Pre-process proprioception-related inputs, e.g. normalizing states
        """
        # Check if stem_spec exists and has normalize_state attribute
        normalize_state = getattr(self.stem_spec, 'normalize_state', False) if hasattr(self, 'stem_spec') else False
        
        if normalize_state and "state" in data:
            data["state"] = self.normalizer[domain]["state"].normalize(data["state"])

        if "prev_actions" in data:
            data["prev_actions"] = self.normalizer[domain]["action"].normalize(data["prev_actions"])

        data["state"] = data["state"][:, :, None]
        return data

    def postprocess_actions(self, domain: str, action: torch.Tensor) -> torch.Tensor:
        """
        Postprocess output, e.g. unnormalizing actions
        """
        if domain in self.normalizer:
            action = self.normalizer[domain]["action"].unnormalize(action)
        return action

    def stem_process(self, domain: str, data: dict):
        """
        Pass through the stem to a fixed number of tokens.
        Args:
            data: dictionary of tensors of different modalities
        """
        feats = []
        # Check if modalities exists, otherwise use data keys
        modalities = self.modalities if hasattr(self, 'modalities') else list(data.keys())
        
        for modality in modalities:
            stem_key = domain + "_" + modality
            if stem_key not in self.stems:
                continue
            
            stem = self.stems[stem_key]

            if modality not in data:
                continue

            use_raw_image = "image" in modality and "image" in self.encoders
            if use_raw_image:  # finetuning with encoders
                data[modality] = self.encoders["image"](data[modality])

            # positional embedding for observations
            data_shape = data[modality].shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            # Check if stem_spec exists and has random_horizon_masking
            random_horizon_masking = getattr(self.stem_spec, 'random_horizon_masking', False) if hasattr(self, 'stem_spec') else False
            
            if self.train_mode and random_horizon_masking and data_horizon > 1:
                horizon = np.random.randint(1, data_horizon + 1)
                data[modality] = data[modality][:, data_horizon - horizon : data_horizon]

            # data is N x T x M x ... x D where M is the # of instances for that sensor
            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]).to(data[modality])
            positional_embedding = einops.repeat(positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0])
            if not use_raw_image:
                data[modality] = data[modality] + positional_embedding.view(data[modality].shape)
            stem_token = stem.compute_latent(data[modality])
            feats.append(stem_token)

        return feats
    
    def forward_features(self, domain: str, data: torch.Tensor, text_features: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the features for the given domain and data.
        Args:
            domain (str): The domain of the data.
            data (Tensor): The input data.
        """
        data = self.preprocess_states(domain, data)

        # stem pass
        self.stem_tokens = self.stem_process(domain, data)

        # combine tokens
        self.trunk_tokens, ori_tokens, proto_tokens, w_route = self.preprocess_tokens(domain, self.stem_tokens)

        # trunk pass with MoLoRA
        if not self.no_trunk:
            # Set routing weights for MoLoRA layers before trunk forward
            if w_route is not None and "fc1" in self.molora_layers and "fc2" in self.molora_layers:
                self.molora_layers["fc1"]._current_w_route = w_route
                self.molora_layers["fc2"]._current_w_route = w_route
                self.trunk_tokens = self.trunk["trunk"](self.trunk_tokens)
            else:
                print('-------------------..........use original transformer')
                self.trunk_tokens = self.trunk["trunk"](self.trunk_tokens)

        # pooling the features
        return self.postprocess_tokens(self.trunk_tokens), ori_tokens, proto_tokens, w_route

    def compute_loss(self, batch):
        """Compute the loss for the training loop forward pass.
        """
        self.train_mode = True
        domain, data = batch["domain"][0], batch["data"]
        features, ori_tokens, proto_tokens, w_route = self.forward_features(domain, data)

        # normalize the labels
        if domain in self.normalizer:
            data["action"] = self.normalizer[domain]["action"].normalize(data["action"])

        # head pass
        loss = self.heads[domain].compute_loss(features, data)
        
        # Reconstruction loss
        loss += F.mse_loss(ori_tokens, proto_tokens)
        
        # Orthogonal loss for all prototype types
        if self.agent_prototypes is not None and len(self.agent_prototypes) > 0:
            loss_ortho_agent = compute_ortho_loss(self.agent_prototypes)
            loss += 0.01 * loss_ortho_agent  # Weight for agent prototypes
        
        if self.env_prototypes is not None and len(self.env_prototypes) > 0:
            loss_ortho_env = compute_ortho_loss(self.env_prototypes)
            loss += 0.01 * loss_ortho_env  # Weight for env prototypes
        
        # 计算并加上大模型语义对齐损失 (权重设为 0.1)
        semantic_loss = self.calc_semantic_loss()
        loss = loss + 0.1 * semantic_loss
        
        return loss
    
    def calc_semantic_loss(self):
        """计算 Task 原型的语义对齐损失"""
        if not hasattr(self, 'task_encoder') or not hasattr(self, 'task_clip_anchors'):
            return torch.tensor(0.0, device=self.device if hasattr(self, 'device') else 'cuda')

        # 1. 获取合法的原型参数: 形状 [4, 1024]
        task_protos = self.task_encoder.task_prototypes

        # 2. 投影到 CLIP 维度: 形状 [4, 512]
        proj_task = self.task_encoder.semantic_proj(task_protos)

        # 3. 计算与 CLIP anchors 的余弦相似度
        # proj_task: [4, 512], task_clip_anchors: [4, 512]
        sim = F.cosine_similarity(proj_task, self.task_clip_anchors, dim=-1)
        
        # 4. 相似度越高越好，所以 loss 是 1 - 相似度
        semantic_loss = (1.0 - sim).mean()
        
        return semantic_loss

    def forward(self, domain: str, data: dict, text_features: torch.Tensor = None):
        """
        Performs a forward pass of the model.
        Args:
            domain: The domain of the data.
            data: Dictionary of observations (vision, proprioception, etc).
            text_features: Optional text embeddings for task encoding.
        """
        # pooling the features
        features, _, _, _ = self.forward_features(domain, data, text_features)

        # head pass
        action = self.heads[domain](features)

        # postprocess. unnormalize the outputs
        action = self.postprocess_actions(domain, action)
        return action

    def print_model_stats(self):
        """Prints out model parameter statistics.

        This method calculates and prints the number of total parameters in the model,
        as well as the number of parameters in each component (stem, trunk, and head).
        It also displays the list of domains associated with the model's heads.
        """
        print("==========================================")
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_parameters = sum(p.numel() for p in self.heads.parameters() if p.requires_grad)
        trunk_parameters = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        stem_parameters = sum(p.numel() for p in self.stems.parameters() if p.requires_grad)
        encoder_parameters = (
            sum(p.numel() for p in self.encoders.parameters() if p.requires_grad) if len(self.encoders) > 0 else 0
        )

        print(
            f"number of total params (M): {n_parameters / 1.0e6:.3f} stem: {stem_parameters / 1.0e6:.3f} "
            f"trunk: {trunk_parameters / 1.0e6:.3f} head: {head_parameters / 1.0e6:.3f} encoder: {encoder_parameters / 1.0e6:.3f}"
        )

    def device(self):
        """get the current device of the model"""
        return next(self.parameters()).device

    def save(self, checkpoint_path: str = "./.checkpoints/full"):
        """save the trunk part of the model"""
        try:
            torch.save(self.state_dict(), checkpoint_path)
            print('------------save success: ', checkpoint_path)
        except FileNotFoundError:
            logging.warning(f"Could not save module parameters for trunk to {checkpoint_path}.")

    def load_model(self, path: str):
        """load the trunk part of the model"""
        self.load_state_dict(torch.load(path), strict=False)

    def load_trunk(self, path: str):
        """load the trunk part of the model"""
        if "hf://" in path:
            if "output" in path:
                path = path.replace("output/", "")
            path = download_from_huggingface(path[len("hf://") :])
            self.trunk.load_state_dict(torch.load(path), strict=True)

    def freeze_trunk(self, num_layers: int = 0):
        """ freeze the trunk parameters in the last num_layers """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_trunk(self, num_layers: int = 0):
        """ unfreeze the trunk parameters in the last num_layers  """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    @classmethod
    def from_pretrained(self, checkpoint_path: str):
        """
        Load a pretrained trunk from the checkpoint and return the policy for transfering.
        """
        checkpoint_path = checkpoint_path.replace("output/", "")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface(checkpoint_path[len("hf://") :])
        cfg = OmegaConf.load(os.path.join(checkpoint_path + "/config.yaml"))
        cfg = OmegaConf.structured(cfg)
        cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(cfg.network)
        policy.load_trunk(os.path.join(checkpoint_path, "trunk.pth"))
        return policy

    @classmethod
    def from_pretrained_full_model(self, checkpoint_path: str, domain: str):
        """
        Load a pretrained full model from a checkpoint and return the policy.
        """
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface("liruiw/" + checkpoint_path)
        cfg = OmegaConf.load(os.path.join(checkpoint_path + "/config.yaml"))
        cfg = OmegaConf.structured(cfg)
        cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(cfg.network)

        # here we need to update dimensions without loading the datasets
        policy.init_domain_stem(domain, cfg.stem)
        policy.init_domain_head(domain, None, cfg.head)
        policy.finalize_modules()
        policy.load_model(os.path.join(checkpoint_path, "model.pth"))

        print("stem keys:", policy.stems.keys())
        policy.print_model_stats()

        return policy


    def _init_weights(self, m):
        """
        Weight initialization for transformer
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    ################# Policy Evaluation-Specific #################
    def reset(self):
        """
        Reset the policy's history buffer
        """
        self.history_buffer = defaultdict(list)

        # current steps in open-loop rollouts
        self.openloop_traj_step = self.action_horizon - 1
        self.language_embedding = None

    @torch.no_grad()
    def get_action(self, data: dict, domain: str = None):
        """Get action in the evaluation setup.

        Args:
            data (dict): The input data dictionary.
            domain (str, optional): The domain for which to get the action. Defaults to None.

        Returns:
            torch.Tensor: The action tensor.
        """
        self.train_mode = False

        def update_history_buffer(key: str, new_obs: torch.Tensor):
            """Update the history buffer with a new observation.

            Args:
                key (str): The key for the observation.
                new_obs (torch.Tensor): The new observation tensor.
            """
            # act like a deque
            self.history_buffer[key].append(new_obs)
            if len(self.history_buffer[key]) > self.observation_horizon:
                self.history_buffer[key].pop(0)

        if domain is None:  # default
            domain = self.domains[0]

        if not hasattr(self, "history_buffer"):
            print("should call policy reset explicitly to avoid problems for evaluation in sequence.")
            self.reset()

        # 获取 action_dim 的安全方式
        try:
            action_dim = len(self.normalizer[domain]["action"].get_input_stats()["min"])
        except (KeyError, AttributeError):
            # 如果 normalizer 未初始化，使用默认值 4（metaworld 环境的动作维度）
            action_dim = 4
        device = next(self.parameters()).device
        data_noimg = {k: v for k, v in data.items() if "image" not in k}
        data_img = {k: v for k, v in data.items() if "image" in k}

        # append batch and T dimensions
        data_th = dict_apply(data_noimg, lambda x: torch.FloatTensor(x)[None, None].to(device).float())

        # handle multi-views and history in image
        img_queue = []
        for img_key in data_img.keys():
            if self.stem_spec.precompute_feat and "image" not in self.encoders:
                # precomputed
                img_embedding = get_image_embeddings(data[img_key], self.stem_spec.image_encoder)
                img_queue.append(torch.FloatTensor(img_embedding).to(device).float())
            else:
                # raw inputs
                image = normalize_image_numpy(data[img_key])
                img_queue.append(torch.FloatTensor(image).to(device).float())

        update_history_buffer("image", torch.cat(img_queue, dim=0))  # concat in channel for views
        data_th["image"] = torch.stack(self.history_buffer["image"], dim=0).float()[None]

        # handle state and language
        for modality in data_noimg.keys():
            update_history_buffer(modality, data_th[modality])

            # language is the same for the whole trajectory
            if "language" in modality:
                if "language" in self.modalities:
                    if self.language_embedding is None:
                        self.language_embedding = get_t5_embeddings(data_th[modality], per_token=True)
                    data_th[modality] = self.language_embedding
            else:
                data_th[modality] = torch.cat(self.history_buffer[modality], dim=1).float()

        # handle previous actions
        if "prev_actions" in self.history_buffer:
            data_th["prev_actions"] = torch.cat(self.history_buffer["prev_actions"], dim=1).float()

        if self.openloop_traj_step != self.action_horizon - 1:
            # use previous predictions in open-loop execution
            self.openloop_traj_step += 1
        else:
            action_th = self(domain, data_th)  # forward pass
            self.action_traj = action_th.detach().cpu().numpy()[0]  # batch=1
            self.action_traj = self.action_traj.reshape(-1, action_dim)  # T x Da
            self.openloop_traj_step = 0  # reset steps

        # handle previous actions
        curr_action = self.action_traj[self.openloop_traj_step]
        update_history_buffer("prev_actions", torch.FloatTensor(curr_action)[None, None, None].to(device).float())
        return curr_action
