# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeDynamicCalibrator(nn.Module):
    """
    Prototype Dynamic Calibrator (PDC)
    
    Real-time drift detection and adaptive calibration for prototypes.
    Addresses the issue of fixed prototype space unable to handle dynamic scene feature drift.
    """
    
    def __init__(self, prototype_dim=64, hidden_dim=128, threshold=0.3, history_size=100,
                 min_calibration_strength=0.0, max_calibration_strength=0.5):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.history_size = history_size
        self.min_calibration_strength = min_calibration_strength
        self.max_calibration_strength = max_calibration_strength
        
        assert 0 <= threshold <= 1, f"threshold must be between 0 and 1, got {threshold}"
        assert history_size > 0, f"history_size must be positive, got {history_size}"
        
        self.drift_detector = nn.Sequential(
            nn.Linear(prototype_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.calibration_net = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prototype_dim)
        )
        
        self.feature_history = None
        self.drift_history = []
        
    def detect_drift(self, current_feat, prototype):
        """
        Detect feature-prototype mismatch and determine if drift has occurred.
        
        Args:
            current_feat: [B, D] or [B, T, D] current features
            prototype: [P, T, D] prototypes
            
        Returns:
            drift_score: [B, P] drift score for each prototype
            is_drifted: [P] boolean mask for prototypes needing calibration
        """
        if current_feat.dim() == 3:
            current_feat_flat = current_feat.mean(dim=1)
        else:
            current_feat_flat = current_feat
            
        P, T, D = prototype.shape
        proto_centers = prototype.mean(dim=1)
        
        if current_feat_flat.dim() == 1:
            current_feat_flat = current_feat_flat.unsqueeze(0)
        
        B = current_feat_flat.shape[0]
        
        similarity = F.cosine_similarity(
            current_feat_flat.unsqueeze(1).expand(-1, P, -1),
            proto_centers.unsqueeze(0).expand(B, -1, -1),
            dim=-1
        )
        
        drift_score = 1 - similarity
        
        is_drifted = (drift_score.mean(dim=0) > self.threshold)
        
        self._update_history(current_feat_flat, drift_score.mean().item())
        
        return drift_score, is_drifted
    
    def _update_history(self, feat, drift_value):
        """Update feature history buffer for long-term drift tracking."""
        if self.feature_history is None:
            self.feature_history = feat.detach().mean(dim=0).unsqueeze(0)
        else:
            new_mean = feat.detach().mean(dim=0).unsqueeze(0)
            self.feature_history = torch.cat([
                self.feature_history, new_mean
            ], dim=0)[-self.history_size:]
        
        self.drift_history.append(drift_value)
        if len(self.drift_history) > self.history_size:
            self.drift_history = self.drift_history[-self.history_size:]
    
    def calibrate_prototype(self, prototype, is_drifted, current_feat, calibration_strength=0.1):
        """
        Perform lightweight adaptive calibration on drifted prototypes.
        
        Args:
            prototype: [P, T, D] prototypes
            is_drifted: [P] boolean mask for prototypes needing calibration
            current_feat: [B, T, D] or [B, D] current features
            calibration_strength: float, calibration step size
            
        Returns:
            calibrated_proto: [P, T, D] calibrated prototypes
        """
        P, T, D = prototype.shape
        calibrated_proto = prototype.clone()
        
        if current_feat.dim() == 3:
            current_feat_mean = current_feat.mean(dim=1)
        else:
            current_feat_mean = current_feat
        
        for idx in range(P):
            if is_drifted[idx]:
                proto_center = prototype[idx].mean(dim=0)
                
                concat_input = torch.cat([proto_center, current_feat_mean.mean(dim=0)], dim=-1)
                drift_confidence = self.drift_detector(concat_input).squeeze()
                
                delta = self.calibration_net(proto_center)
                
                adaptive_strength = calibration_strength * drift_confidence
                adaptive_strength = torch.clamp(
                    adaptive_strength,
                    min=self.min_calibration_strength,
                    max=self.max_calibration_strength
                )
                new_proto_center = proto_center + adaptive_strength * delta
                
                calibrated_proto[idx] = new_proto_center.unsqueeze(0).expand(T, -1)
        
        return calibrated_proto
    
    def forward(self, current_feat, prototype, calibration_strength=0.1):
        """
        Complete dynamic calibration pipeline.
        
        Args:
            current_feat: [B, T, D] or [B, D] current features
            prototype: [P, T, D] prototypes
            calibration_strength: float, calibration step size
            
        Returns:
            calibrated_proto: [P, T, D] calibrated prototypes
            drift_score: [B, P] drift scores
            is_drifted: [P] boolean mask for drifted prototypes
        """
        drift_score, is_drifted = self.detect_drift(current_feat, prototype)
        
        calibrated_proto = self.calibrate_prototype(
            prototype, is_drifted, current_feat, calibration_strength
        )
        
        return calibrated_proto, drift_score, is_drifted
    
    def get_drift_statistics(self):
        """Get statistics about drift history."""
        if len(self.drift_history) == 0:
            return None
        
        return {
            'mean_drift': sum(self.drift_history) / len(self.drift_history),
            'max_drift': max(self.drift_history),
            'drift_trend': self.drift_history[-10:] if len(self.drift_history) >= 10 else self.drift_history
        }


class MultiScaleDriftDetector(nn.Module):
    """
    Multi-scale drift detector for capturing both short-term and long-term drifts.
    """
    
    def __init__(self, prototype_dim=64, scales=[1, 5, 10]):
        super().__init__()
        self.scales = scales
        self.detectors = nn.ModuleList([
            PrototypeDynamicCalibrator(prototype_dim) for _ in scales
        ])
        
    def forward(self, current_feat, prototype):
        """
        Multi-scale drift detection and calibration.
        
        Returns:
            calibrated_proto: calibrated prototypes
            drift_info: dict with multi-scale drift information
        """
        calibrated_protos = []
        drift_infos = []
        
        for scale, detector in zip(self.scales, self.detectors):
            cal_proto, drift_score, is_drifted = detector(current_feat, prototype)
            calibrated_protos.append(cal_proto)
            drift_infos.append({
                'scale': scale,
                'drift_score': drift_score,
                'is_drifted': is_drifted
            })
        
        weights = torch.softmax(torch.tensor([1.0 / s for s in self.scales]), dim=0)
        final_proto = sum(w * p for w, p in zip(weights, calibrated_protos))
        
        return final_proto, drift_infos
