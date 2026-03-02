# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PrototypeCrossDomainTransfer(nn.Module):
    """
    Prototype Cross-domain Transfer (PCT)
    
    Enables zero-shot and few-shot prototype transfer for new agents/environments.
    Addresses the issue of heavy dependence on large-scale data.
    """
    
    def __init__(
        self, 
        prototype_dim=64, 
        hidden_dim=256,
        source_prototypes: Optional[Dict[str, torch.Tensor]] = None,
        num_prototypes_per_domain=6
    ):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes_per_domain = num_prototypes_per_domain
        
        if source_prototypes is not None:
            self.register_buffer("source_agent_prototypes", source_prototypes.get('agent'))
            self.register_buffer("source_env_prototypes", source_prototypes.get('env'))
        else:
            self.source_agent_prototypes = None
            self.source_env_prototypes = None
        
        self.similarity_net = nn.Sequential(
            nn.Linear(prototype_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.interpolation_net = nn.Sequential(
            nn.Linear(prototype_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prototype_dim)
        )
        
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=prototype_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.domain_adapter = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, prototype_dim)
        )
        
    def set_source_prototypes(self, agent_prototypes: torch.Tensor, env_prototypes: torch.Tensor):
        """Set source domain prototypes for transfer."""
        self.source_agent_prototypes = agent_prototypes
        self.source_env_prototypes = env_prototypes
        
    def retrieve_similar_prototypes(
        self, 
        target_feat: torch.Tensor, 
        source_prototypes: torch.Tensor, 
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve similar prototypes from source domain prototype bank.
        
        Args:
            target_feat: [D] target domain feature
            source_prototypes: [P, T, D] source domain prototypes
            top_k: number of prototypes to retrieve
            
        Returns:
            similar_protos: [top_k, D] similar prototypes (mean pooled)
            similarities: [top_k] similarity scores
        """
        if source_prototypes is None:
            raise ValueError("Source prototypes not set. Call set_source_prototypes first.")
        
        P, T, D = source_prototypes.shape
        proto_centers = source_prototypes.mean(dim=1)
        
        if target_feat.dim() == 1:
            target_feat = target_feat.unsqueeze(0)
        
        similarities = F.cosine_similarity(
            target_feat,
            proto_centers,
            dim=-1
        )
        
        if top_k > P:
            top_k = P
        
        k = min(top_k, P)
        top_k_values, top_k_indices = torch.topk(similarities, k)
        similar_protos = proto_centers[top_k_indices]
        
        return similar_protos, top_k_values
    
    def interpolate_new_prototype(
        self, 
        target_feat: torch.Tensor, 
        similar_protos: torch.Tensor, 
        similar_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate new prototype through interpolation.
        
        Args:
            target_feat: [D] target domain feature
            similar_protos: [K, D] similar prototypes
            similar_scores: [K] similarity scores
            
        Returns:
            new_proto: [D] interpolated new prototype
        """
        weights = F.softmax(similar_scores, dim=0)
        weighted_proto = (similar_protos * weights.unsqueeze(-1)).sum(dim=0)
        
        concat_feat = torch.cat([
            weighted_proto,
            similar_protos[0],
            target_feat
        ], dim=-1)
        
        new_proto = self.interpolation_net(concat_feat)
        
        return new_proto
    
    def few_shot_finetune(
        self, 
        new_proto: torch.Tensor, 
        few_shot_data: torch.Tensor, 
        num_steps: int = 10, 
        lr: float = 0.01
    ) -> torch.Tensor:
        """
        Fast fine-tuning with few-shot data.
        
        Args:
            new_proto: [D] new prototype
            few_shot_data: [N, D] few-shot samples
            num_steps: number of fine-tuning steps
            lr: learning rate
            
        Returns:
            finetuned_proto: [D] fine-tuned prototype
        """
        if few_shot_data is None or len(few_shot_data) == 0:
            return new_proto
        
        proto = new_proto.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([proto], lr=lr)
        
        if few_shot_data.dim() == 3:
            few_shot_data = few_shot_data.mean(dim=1)
        
        for step in range(num_steps):
            similarities = F.cosine_similarity(
                proto.unsqueeze(0),
                few_shot_data,
                dim=-1
            )
            
            loss = -similarities.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return proto.detach()
    
    def transfer_prototype(
        self, 
        target_feat: torch.Tensor, 
        prototype_type: str = 'agent',
        few_shot_data: Optional[torch.Tensor] = None,
        top_k: int = 3,
        num_finetune_steps: int = 10
    ) -> torch.Tensor:
        """
        Complete prototype transfer pipeline.
        
        Args:
            target_feat: [D] target domain feature
            prototype_type: 'agent' or 'env'
            few_shot_data: [N, D] few-shot samples (optional)
            top_k: number of similar prototypes to retrieve
            num_finetune_steps: number of fine-tuning steps
            
        Returns:
            transferred_proto: [D] transferred prototype
        """
        source_protos = (self.source_agent_prototypes 
                        if prototype_type == 'agent' 
                        else self.source_env_prototypes)
        
        if source_protos is None:
            raise ValueError(f"Source {prototype_type} prototypes not set.")
        
        similar_protos, similar_scores = self.retrieve_similar_prototypes(
            target_feat, source_protos, top_k
        )
        
        new_proto = self.interpolate_new_prototype(
            target_feat, similar_protos, similar_scores
        )
        
        if few_shot_data is not None and len(few_shot_data) > 0:
            new_proto = self.few_shot_finetune(
                new_proto, few_shot_data, num_finetune_steps
            )
        
        return new_proto
    
    def transfer_domain(
        self,
        agent_feat: torch.Tensor,
        env_feat: torch.Tensor,
        few_shot_data: Optional[Dict[str, torch.Tensor]] = None,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transfer both agent and environment prototypes for a new domain.
        
        Args:
            agent_feat: [D] agent feature
            env_feat: [D] environment feature
            few_shot_data: dict with 'agent' and 'env' few-shot data
            top_k: number of similar prototypes to retrieve
            
        Returns:
            new_agent_proto: [D] new agent prototype
            new_env_proto: [D] new environment prototype
        """
        agent_few_shot = few_shot_data.get('agent') if few_shot_data else None
        env_few_shot = few_shot_data.get('env') if few_shot_data else None
        
        new_agent_proto = self.transfer_prototype(
            agent_feat, 'agent', agent_few_shot, top_k
        )
        
        new_env_proto = self.transfer_prototype(
            env_feat, 'env', env_few_shot, top_k
        )
        
        return new_agent_proto, new_env_proto


class ZeroShotPrototypeTransfer(PrototypeCrossDomainTransfer):
    """
    Zero-shot prototype transfer without any target domain data.
    Uses learned priors from source domain.
    """
    
    def __init__(self, prototype_dim=64, hidden_dim=256):
        super().__init__(prototype_dim=prototype_dim, hidden_dim=hidden_dim)
        
        self.prototype_prior_net = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prototype_dim)
        )
        
    def zero_shot_transfer(
        self, 
        domain_description: torch.Tensor,
        prototype_type: str = 'agent'
    ) -> torch.Tensor:
        """
        Generate prototype from domain description without target data.
        
        Args:
            domain_description: [D] domain description embedding
            prototype_type: 'agent' or 'env'
            
        Returns:
            generated_proto: [D] generated prototype
        """
        source_protos = (self.source_agent_prototypes 
                        if prototype_type == 'agent' 
                        else self.source_env_prototypes)
        
        if source_protos is None:
            raise ValueError(f"Source {prototype_type} prototypes not set. Call set_source_prototypes first.")
        
        similar_protos, scores = self.retrieve_similar_prototypes(
            domain_description, source_protos, top_k=5
        )
        prior_proto = (similar_protos * F.softmax(scores, dim=0).unsqueeze(-1)).sum(dim=0)
        
        generated_proto = self.prototype_prior_net(domain_description) + prior_proto
        
        return generated_proto


class MetaLearningPrototypeTransfer(nn.Module):
    """
    Meta-learning based prototype transfer for rapid adaptation.
    Uses MAML-style inner loop optimization.
    """
    
    def __init__(self, prototype_dim=64, hidden_dim=256, inner_lr=0.01):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.inner_lr = inner_lr
        
        self.encoder = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prototype_dim)
        )
        
        self.adaptation_net = nn.Sequential(
            nn.Linear(prototype_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prototype_dim)
        )
        
    def inner_loop_adapt(
        self, 
        proto: torch.Tensor, 
        support_data: torch.Tensor, 
        num_steps: int = 5
    ) -> torch.Tensor:
        """
        Inner loop adaptation for meta-learning.
        
        Args:
            proto: [D] prototype to adapt
            support_data: [N, D] support set data
            num_steps: number of inner loop steps
            
        Returns:
            adapted_proto: [D] adapted prototype
        """
        adapted_proto = proto.clone()
        
        for _ in range(num_steps):
            similarities = F.cosine_similarity(
                adapted_proto.unsqueeze(0),
                support_data,
                dim=-1
            )
            loss = -similarities.mean()
            
            grad = torch.autograd.grad(
                loss, adapted_proto, create_graph=True
            )[0]
            
            adapted_proto = adapted_proto - self.inner_lr * grad
        
        return adapted_proto
    
    def forward(
        self, 
        source_proto: torch.Tensor, 
        target_feat: torch.Tensor, 
        support_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Meta-learning based prototype transfer.
        """
        encoded_proto = self.encoder(source_proto)
        encoded_target = self.encoder(target_feat)
        
        concat_feat = torch.cat([encoded_proto, encoded_target], dim=-1)
        transferred_proto = self.adaptation_net(concat_feat)
        
        if support_data is not None:
            transferred_proto = self.inner_loop_adapt(
                transferred_proto, support_data
            )
        
        return transferred_proto
