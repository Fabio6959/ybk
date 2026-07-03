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
import numpy as np
import einops
from collections import defaultdict
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


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> torch.Tensor:
    """
    Gumbel-Softmax with Straight-Through Estimator (STE).
    
    Key insight:
    - Forward pass: Uses hard (one-hot) for true sparsity
    - Backward pass: Uses soft gradients for differentiability
    
    This achieves the best of both worlds:
    - True sparse activation (only selected experts are computed)
    - Full gradient flow (all experts receive gradient signals)
    
    Args:
        logits: Unnormalized log probabilities [*, N]
        tau: Temperature for Gumbel-Softmax (lower = sharper)
        hard: If True, use STE (hard forward, soft backward)
        dim: Dimension to apply softmax
    
    Returns:
        samples: Sampled one-hot vectors (if hard=True) or soft samples
    """
    # Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    
    # Soft sample with Gumbel noise
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)
    
    if hard:
        # Straight-Through Estimator
        # Forward: argmax -> one-hot (hard, sparse)
        # Backward: y_soft gradients (soft, differentiable)
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        
        # STE trick: y_hard in forward, y_soft gradient in backward
        # (y_hard - y_soft) has zero gradient, so we add y_soft back for backward
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


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


def compute_load_balance_loss(w_route: torch.Tensor) -> torch.Tensor:
    """
    Compute load balancing loss to prevent expert collapse.
    
    Based on Switch Transformer: https://arxiv.org/abs/2101.03961
    
    L_aux = N * sum(f_i * P_i)
    where:
        f_i = fraction of tokens routed to expert i (routing frequency)
        P_i = average routing probability for expert i
    
    This loss encourages uniform routing distribution across all experts.
    
    Args:
        w_route: [B, N] routing weights (after softmax)
    
    Returns:
        loss: load balancing auxiliary loss
    """
    N = w_route.shape[-1]  # number of experts
    
    # f_i: fraction of tokens routed to expert i
    # Using soft routing weights instead of hard assignment for differentiability
    f = w_route.mean(dim=0)  # [N]
    
    # P_i: average routing probability for expert i
    # Using squared weights to emphasize confident routing
    P = (w_route ** 2).sum(dim=0) / (w_route.sum(dim=0) + 1e-8)  # [N]
    
    # Load balancing loss: minimize when f_i = P_i = 1/N for all i
    aux_loss = N * torch.sum(f * P)
    
    return aux_loss


def compute_entropy_loss(w_route: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy loss to encourage sharp (specialized) routing distributions.
    
    Key insight: 
    - Load balance loss operates at BATCH level (macro equilibrium)
    - Entropy loss operates at TOKEN level (micro specialization)
    
    Without entropy penalty, the network would converge to uniform distribution
    for every sample, achieving "balance" but losing expert specialization.
    
    Entropy minimization encourages each sample to make a confident (low-entropy)
    routing decision, while load balance ensures different samples choose different
    experts at the batch level.
    
    Args:
        w_route: [B, N] routing weights (after softmax)
    
    Returns:
        loss: entropy loss (minimize to encourage sharp distributions)
    """
    # Entropy of each sample's routing distribution
    # H(w) = -sum(w_i * log(w_i))
    # Max entropy = log(N) (uniform), Min entropy = 0 (one-hot)
    entropy = -torch.sum(w_route * torch.log(w_route + 1e-8), dim=-1)  # [B]
    
    # Average entropy across batch
    mean_entropy = entropy.mean()
    
    return mean_entropy


class SkillActionEncoder(nn.Module):
    """
    Encode an expert action window into a latent skill primitive embedding.

    The action branch is a training-only teacher. Inference uses the
    observation-conditioned skill router and does not consume future actions.
    """
    def __init__(self, one_action_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.one_action_dim = one_action_dim
        self.step_encoder = nn.Sequential(
            nn.Linear(one_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, action_window: torch.Tensor) -> torch.Tensor:
        # action_window: [B, H, A]
        x = self.step_encoder(action_window)
        x = x.mean(dim=1)
        x = self.temporal_proj(x)
        return F.normalize(x, p=2, dim=-1)


class UnifiedSemanticSpace(nn.Module):
    """
    Project Agent/Env/Skill prototypes to a unified semantic space.
    
    Key insight:
    - Agent prototypes: physical space (joint angles, end-effector pose)
    - Env prototypes: visual space (CNN features, object positions)
    - Skill prototypes: latent action primitive space
    
    These three spaces are semantically incompatible without projection.
    This module projects all three to a shared 512-dim semantic space.
    """
    def __init__(self, embed_dim=128, semantic_dim=512):
        super().__init__()
        self.semantic_dim = semantic_dim
        
        # Three independent projection heads
        self.agent_proj = nn.Sequential(
            nn.Linear(embed_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        self.env_proj = nn.Sequential(
            nn.Linear(embed_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
        self.skill_proj = nn.Sequential(
            nn.Linear(embed_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )
    
    def forward(self, agent_proto, env_proto, skill_proto):
        """
        Project prototypes to unified semantic space.
        
        Args:
            agent_proto: [P, embed_dim]
            env_proto: [P, embed_dim]
            skill_proto: [P, embed_dim]
        
        Returns:
            agent_sem, env_sem, skill_sem: all [P, semantic_dim], L2 normalized
        """
        agent_sem = F.normalize(self.agent_proj(agent_proto), p=2, dim=-1)
        env_sem = F.normalize(self.env_proj(env_proto), p=2, dim=-1)
        skill_sem = F.normalize(self.skill_proj(skill_proto), p=2, dim=-1)
        
        return agent_sem, env_sem, skill_sem


def compute_cross_modal_alignment_loss(agent_sem, env_sem, skill_sem, temperature=0.07):
    """
    Compute cross-modal alignment loss using contrastive learning.
    
    Key insight:
    - Prototypes with the same index should be close (positive pairs)
    - Prototypes with different indices should be far (negative pairs)
    
    This forces Agent/Env/Skill prototypes to align in semantic space.
    
    Args:
        agent_sem: [P, semantic_dim]
        env_sem: [P, semantic_dim]
        skill_sem: [P, semantic_dim]
        temperature: Temperature for InfoNCE loss
    
    Returns:
        loss: Cross-modal alignment loss
    """
    P = min(agent_sem.shape[0], env_sem.shape[0], skill_sem.shape[0])
    agent_sem = agent_sem[:P]
    env_sem = env_sem[:P]
    skill_sem = skill_sem[:P]
    device = agent_sem.device
    
    # Compute similarity matrices
    sim_ae = torch.matmul(agent_sem, env_sem.T) / temperature  # [P, P]
    sim_as = torch.matmul(agent_sem, skill_sem.T) / temperature
    sim_es = torch.matmul(env_sem, skill_sem.T) / temperature
    
    # Labels: diagonal should be positive pairs
    labels = torch.arange(P, device=device)
    
    # InfoNCE loss (cross-entropy with similarity as logits)
    loss_ae = (F.cross_entropy(sim_ae, labels) + F.cross_entropy(sim_ae.T, labels)) / 2
    loss_as = (F.cross_entropy(sim_as, labels) + F.cross_entropy(sim_as.T, labels)) / 2
    loss_es = (F.cross_entropy(sim_es, labels) + F.cross_entropy(sim_es.T, labels)) / 2
    
    return (loss_ae + loss_as + loss_es) / 3.0


class MoLoRALayer(nn.Module):
    """
    Mixture of LoRAs (MoLoRA) for efficient fine-tuning.
    Replaces nn.Linear layers with low-rank adapters weighted by routing scores.
    
    Supports sparse activation via Gumbel-Softmax:
    - When w_route is sparse (one-hot from STE), only activated experts are computed
    - This achieves true computational efficiency similar to Top-k MoE
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
        self.lora_B = nn.Parameter(torch.randn(num_combinations, r, out_features) * 1e-4)
        
        # Cache for routing weights (set externally by Policy)
        self._current_w_route = None
        
        # Base layer weight and bias (will be set by _replace_linear_layer)
        self.weight = None
        self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse routing support.
        
        Key optimization:
        - When w_route is one-hot (from STE), only compute the activated expert
        - When w_route is dense, fall back to weighted blending
        
        Args:
            x: [B, in_features] or [B, T, in_features] input tensor
        
        Returns:
            y: [B, out_features] or [B, T, out_features] output tensor
        """
        w = getattr(self, "_current_w_route", None)
        B_x = x.shape[0]
        
        if w is None:
            N = self.lora_A.shape[0]
            w = torch.ones(B_x, N, device=x.device, dtype=x.dtype) / N
        else:
            B_w = w.shape[0]
            if B_w != B_x:
                if B_w > B_x:
                    w = w[:B_x, :]
                else:
                    repeats = (B_x + B_w - 1) // B_w
                    w = w.repeat(repeats, 1)[:B_x, :]
        
        # Check if routing is sparse (one-hot from STE)
        # A one-hot vector has exactly one element = 1, rest = 0
        is_sparse = (w.sum(dim=-1) - 1.0).abs().max() < 1e-5 and (w.max(dim=-1)[0] - 1.0).abs().max() < 1e-5
        
        if is_sparse and not self.training:
            # Sparse path: only compute activated experts
            # Get the index of the activated expert for each sample
            expert_indices = w.argmax(dim=-1)  # [B]
            
            # Gather the LoRA matrices for the activated experts
            # lora_A: [N, in, r], lora_B: [N, r, out]
            selected_A = self.lora_A[expert_indices]  # [B, in, r]
            selected_B = self.lora_B[expert_indices]  # [B, r, out]
            
            # Compute LoRA output for each sample with its selected expert
            if x.dim() == 3:
                # x: [B, T, in], selected_A: [B, in, r]
                lora_mid = torch.einsum('bti, bir -> btr', x, selected_A)
                lora_out = torch.einsum('btr, bro -> bto', lora_mid, selected_B)
            else:
                # x: [B, in], selected_A: [B, in, r]
                lora_mid = torch.einsum('bi, bir -> br', x, selected_A)
                lora_out = torch.einsum('br, bro -> bo', lora_mid, selected_B)
        else:
            # Dense path: weighted blending (used during training for gradient flow)
            blended_A = torch.einsum('bn, nir -> bir', w, self.lora_A)
            blended_B = torch.einsum('bn, nro -> bro', w, self.lora_B)
            
            if x.dim() == 3:
                lora_mid = torch.einsum('bti, bir -> btr', x, blended_A)
                lora_out = torch.einsum('btr, bro -> bto', lora_mid, blended_B)
            elif x.dim() == 2:
                lora_mid = torch.einsum('bi, bir -> br', x, blended_A)
                lora_out = torch.einsum('br, bro -> bo', lora_mid, blended_B)
            else:
                raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Add base layer output
        if self.weight is not None:
            base_out = F.linear(x, self.weight, self.bias)
            return base_out + lora_out
        else:
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

        # Agent/Environment/Skill prototype routing. Skill routing is learned
        # from observation tokens and action-window assignments, not labels.
        self.prototype_num = int(kwargs.get("prototype_num", 6))
        self.num_skill_protos = int(kwargs.get("num_skill_prototypes", kwargs.get("num_skill_protos", 6)))
        self.lora_r = int(kwargs.get("lora_r", 64))
        self.prototype_momentum = float(kwargs.get("prototype_momentum", 0.9))
        self.tau_start = float(kwargs.get("tau_start", 1.0))
        self.tau_end = float(kwargs.get("tau_end", 0.5))
        self.tau_anneal_steps = float(kwargs.get("tau_anneal_steps", 20000.0))
        self.lambda_skill_align = float(kwargs.get("lambda_skill_align", 0.05))

        # Observation-side skill router. This is the only skill route used at inference.
        self.skill_router = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_skill_protos)
        )
        self.skill_action_encoder = None
        self.one_action_dim = None

        self.register_buffer("agent_prototypes", F.normalize(torch.randn(self.prototype_num, self.embed_dim), p=2, dim=-1))
        self.register_buffer("env_prototypes", F.normalize(torch.randn(self.prototype_num, self.embed_dim), p=2, dim=-1))
        self.register_buffer("skill_prototypes", F.normalize(torch.randn(self.num_skill_protos, self.embed_dim), p=2, dim=-1))
        
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        
        # MoLoRA layers for blocks.15.mlp.fc1 and blocks.15.mlp.fc2.
        # Total combinations: N_agent * N_env * N_skill.
        self.num_combinations = self.prototype_num * self.prototype_num * self.num_skill_protos
        self.molora_layers = nn.ModuleDict()
        self._init_molora_layers()
        
        # Store references to original layers for replacement
        self.original_fc1 = None
        self.original_fc2 = None
        
        # Routing weights storage
        self.current_w_a = None
        self.current_w_e = None
        self.current_w_s = None
        
        # Project Agent/Env/Skill prototypes into a unified semantic space.
        self.semantic_space = UnifiedSemanticSpace(
            embed_dim=embed_dim,
            semantic_dim=512
        )
        
        # Learnable fusion weights for Agent/Env/Skill branches.
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

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
            self._init_skill_action_encoder_from_normalizer(domain_name)

    def _init_skill_action_encoder_from_normalizer(self, domain_name: str):
        """Initialize the action-window skill teacher once action dimensions are known."""
        if self.skill_action_encoder is not None:
            return
        try:
            stats = self.normalizer[domain_name]["action"].get_input_stats()
            if "min" in stats:
                one_action_dim = int(stats["min"].numel())
            elif "mean" in stats:
                one_action_dim = int(stats["mean"].numel())
            else:
                return
        except (KeyError, AttributeError, RuntimeError):
            return

        self.one_action_dim = one_action_dim
        self.skill_action_encoder = SkillActionEncoder(
            one_action_dim=one_action_dim,
            embed_dim=self.embed_dim,
        )

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


    def feature2proto(self, tokens, action_window=None):
        B, L, D = tokens.shape
        pooled_tokens = tokens.mean(dim=1)

        if self.training:
            self.global_step += 1

        progress = min(1.0, self.global_step.item() / max(self.tau_anneal_steps, 1.0))
        tau = self.tau_start + (self.tau_end - self.tau_start) * progress

        pooled_tokens_norm = F.normalize(pooled_tokens, p=2, dim=-1)
        agent_protos_norm = F.normalize(self.agent_prototypes, p=2, dim=-1)
        env_protos_norm = F.normalize(self.env_prototypes, p=2, dim=-1)
        skill_protos_norm = F.normalize(self.skill_prototypes, p=2, dim=-1)

        logits_a = torch.matmul(pooled_tokens_norm, agent_protos_norm.T) / tau
        logits_e = torch.matmul(pooled_tokens_norm, env_protos_norm.T) / tau
        logits_s_obs = self.skill_router(pooled_tokens) / tau

        use_hard = self.training
        w_a = gumbel_softmax(logits_a, tau=tau, hard=use_hard)
        w_e = gumbel_softmax(logits_e, tau=tau, hard=use_hard)
        w_s = gumbel_softmax(logits_s_obs, tau=tau, hard=use_hard)

        skill_align_loss = None
        action_window = self._format_action_window(action_window)
        if self.training and action_window is not None and self.skill_action_encoder is not None:
            z_skill_action = self.skill_action_encoder(action_window)
            logits_s_action = torch.matmul(z_skill_action, skill_protos_norm.T) / tau
            w_s_action = F.softmax(logits_s_action, dim=-1)

            with torch.no_grad():
                self._update_skill_prototypes(z_skill_action.detach(), w_s_action.detach())

            skill_align_loss = F.kl_div(
                F.log_softmax(logits_s_obs, dim=-1),
                w_s_action.detach(),
                reduction="batchmean",
            )

        w_route = torch.einsum('bi, bj, bk -> bijk', w_a, w_e, w_s).reshape(B, -1)

        fusion_w = F.softmax(self.fusion_weights, dim=0)
        combined_features = (
            fusion_w[0] * torch.matmul(w_a, agent_protos_norm) +
            fusion_w[1] * torch.matmul(w_e, env_protos_norm) +
            fusion_w[2] * torch.matmul(w_s, skill_protos_norm)
        )
        combined_features = F.normalize(combined_features, p=2, dim=-1)
        proto_tokens = combined_features.unsqueeze(1).expand(-1, L, -1)

        self.current_w_a = w_a.detach()
        self.current_w_e = w_e.detach()
        self.current_w_s = w_s.detach()

        with torch.no_grad():
            self.agent_prototypes.copy_(
                self._update_prototypes(self.agent_prototypes, pooled_tokens.detach(), w_a.detach())
            )
            self.env_prototypes.copy_(
                self._update_prototypes(self.env_prototypes, pooled_tokens.detach(), w_e.detach())
            )

        return proto_tokens, w_route, skill_align_loss
    def preprocess_tokens(self, domain: str, features: List[torch.Tensor], action_window=None) -> torch.Tensor:
        """
        Shared modality layers and add modality tokens. Add positional and time embeddings.
        """
        tokens = torch.cat(features, dim=-2)
        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens[domain].repeat(len(tokens), 1, 1)
            tokens = torch.cat([tokens, action_tokens], dim=-2)
        
        ori_tokens = tokens
        proto_tokens, w_route, skill_align_loss = self.feature2proto(tokens, action_window=action_window)
        position_tokens = self.get_position_embedding(proto_tokens, self.embed_dim)
        tokens = tokens + position_tokens
        return tokens, ori_tokens, proto_tokens, w_route, skill_align_loss

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

    def _format_action_window(self, action: Optional[torch.Tensor]):
        """Return actions as [B, H, A] for the action-side skill teacher."""
        if action is None:
            return None
        if action.dim() == 3:
            return action
        if action.dim() != 2:
            raise ValueError(f"Expected action shape [B, D] or [B, H, A], got {tuple(action.shape)}")

        B, D = action.shape
        if self.one_action_dim is not None and self.one_action_dim > 0 and D % self.one_action_dim == 0:
            return action.view(B, D // self.one_action_dim, self.one_action_dim)
        return action.unsqueeze(1)

    @torch.no_grad()
    def _update_skill_prototypes(self, skill_embeddings: torch.Tensor, skill_weights: torch.Tensor):
        """EMA-update skill primitives from action-window embeddings."""
        weights_sum = skill_weights.sum(dim=0)  # [K]
        active = weights_sum > 1e-5
        if not torch.any(active):
            return

        proto_update = torch.matmul(skill_weights.T, skill_embeddings)
        proto_update = proto_update / (weights_sum.unsqueeze(-1) + 1e-6)

        updated = self.skill_prototypes.clone()
        updated[active] = (
            self.prototype_momentum * self.skill_prototypes[active]
            + (1.0 - self.prototype_momentum) * proto_update[active]
        )
        self.skill_prototypes.copy_(F.normalize(updated, p=2, dim=-1))


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
        
        Args:
            domain: Domain identifier used to select the initialized stem/normalizer.
            data: Dictionary containing state and other data
        """
        # Check if stem_spec exists and has normalize_state attribute
        normalize_state = getattr(self.stem_spec, 'normalize_state', False) if hasattr(self, 'stem_spec') else False
        
        # If a rollout-specific domain is passed, fall back to the initialized parent domain.
        normalizer_domain = domain
        if domain not in self.normalizer:
            # Try to find a parent domain that exists in self.normalizer
            for parent_domain in self.domains:
                # Check if this parent domain exists and can be used
                if parent_domain in self.normalizer:
                    normalizer_domain = parent_domain
                    break
        
        if normalize_state and "state" in data:
            data["state"] = self.normalizer[normalizer_domain]["state"].normalize(data["state"])

        if "prev_actions" in data:
            data["prev_actions"] = self.normalizer[normalizer_domain]["action"].normalize(data["prev_actions"])

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
            domain: Domain identifier used to select the initialized stem.
            data: dictionary of tensors of different modalities
        """
        feats = []
        # Check if modalities exists, otherwise use data keys
        modalities = self.modalities if hasattr(self, 'modalities') else list(data.keys())
        
        # If a rollout-specific domain is passed, fall back to the initialized parent domain.
        parent_domain = domain
        for modality in modalities:
            stem_key = domain + "_" + modality
            if stem_key not in self.stems:
                # Try to find a parent domain that has stems initialized
                for pd in self.domains:
                    parent_stem_key = pd + "_" + modality
                    if parent_stem_key in self.stems:
                        parent_domain = pd
                        break
                break
        
        for modality in modalities:
            stem_key = parent_domain + "_" + modality
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
    
    def forward_features(self, domain: str, data: dict, action_window=None) -> torch.Tensor:
        """
        Compute the features for the given domain and data.
        Args:
            domain (str): The domain of the data.
            data (dict): The input observations.
        """
        if not hasattr(self, "train_mode"):
            self.train_mode = False

        data = self.preprocess_states(domain, data)

        # stem pass
        self.stem_tokens = self.stem_process(domain, data)

        # combine tokens
        self.trunk_tokens, ori_tokens, proto_tokens, w_route, skill_align_loss = self.preprocess_tokens(
            domain, self.stem_tokens, action_window=action_window
        )

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
        return self.postprocess_tokens(self.trunk_tokens), ori_tokens, proto_tokens, w_route, skill_align_loss

    def compute_loss(self, batch):
        """Compute the loss for the training loop forward pass.
        
        Args:
            batch: Dictionary with 'domain' strings and 'data' tensors.
        """
        self.train_mode = True
        domain, data = batch["domain"][0], batch["data"]
        raw_action = data["action"].clone() if "action" in data else None
        
        # If a rollout-specific domain is passed, fall back to the initialized parent domain.
        parent_domain = domain
        if domain not in self.normalizer or domain not in self.heads:
            # Try to find a parent domain that exists
            for pd in self.domains:
                if pd in self.normalizer and pd in self.heads:
                    parent_domain = pd
                    break
        
        features, ori_tokens, proto_tokens, w_route, skill_align_loss = self.forward_features(
            domain, data, action_window=raw_action
        )

        # normalize the labels
        if parent_domain in self.normalizer:
            data["action"] = self.normalizer[parent_domain]["action"].normalize(data["action"])

        # head pass
        loss = self.heads[parent_domain].compute_loss(features, data)
        
        # Reconstruction loss
        loss += F.mse_loss(ori_tokens, proto_tokens)
        
        # Orthogonal loss for all prototype types
        if self.agent_prototypes is not None and len(self.agent_prototypes) > 0:
            loss_ortho_agent = compute_ortho_loss(self.agent_prototypes)
            loss += 0.01 * loss_ortho_agent  # Weight for agent prototypes
        
        if self.env_prototypes is not None and len(self.env_prototypes) > 0:
            loss_ortho_env = compute_ortho_loss(self.env_prototypes)
            loss += 0.01 * loss_ortho_env  # Weight for env prototypes

        if self.skill_prototypes is not None and len(self.skill_prototypes) > 0:
            loss_ortho_skill = compute_ortho_loss(self.skill_prototypes)
            loss += 0.01 * loss_ortho_skill
        
        # Load balancing loss (Batch-level) + Entropy loss (Token-level)
        # Key insight: Balance ensures macro equilibrium, Entropy ensures micro specialization
        if w_route is not None:
            # Batch-level: encourage uniform expert usage across the batch
            load_balance_loss = compute_load_balance_loss(w_route)
            loss += 0.01 * load_balance_loss
            
            # Token-level: encourage sharp (low-entropy) routing for each sample
            # This prevents the network from collapsing to uniform distribution
            entropy_loss = compute_entropy_loss(w_route)
            loss += 0.01 * entropy_loss
        
        # Align Agent/Env/Skill prototypes in the unified semantic space.
        agent_sem, env_sem, skill_sem = self.semantic_space(
            self.agent_prototypes,
            self.env_prototypes,
            self.skill_prototypes
        )
        cross_modal_loss = compute_cross_modal_alignment_loss(agent_sem, env_sem, skill_sem)
        loss += 0.01 * cross_modal_loss

        if skill_align_loss is not None:
            loss = loss + self.lambda_skill_align * skill_align_loss

        return loss

    def forward(self, domain: str, data: dict):
        """
        Performs a forward pass of the model.
        Args:
            domain: The domain of the data.
            data: Dictionary of observations (vision, proprioception, etc).
        """
        parent_domain = domain
        if domain not in self.heads:
            for pd in self.domains:
                if pd in self.heads:
                    parent_domain = pd
                    break

        # pooling the features
        features, _, _, _, _ = self.forward_features(domain, data)

        # head pass
        action = self.heads[parent_domain](features)

        # postprocess. unnormalize the outputs
        action = self.postprocess_actions(parent_domain, action)
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

        # Determine action dimension from the fitted normalizer.
        try:
            action_dim = len(self.normalizer[domain]["action"].get_input_stats()["min"])
        except (KeyError, AttributeError):
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
