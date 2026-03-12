import torch
import sys
sys.path.insert(0, 'hpt')

from models.policy import Policy
from omegaconf import OmegaConf

def test_policy_initialization():
    """Test Policy initialization with MoLoRA"""
    print("=" * 50)
    print("Testing Policy Initialization")
    print("=" * 50)
    
    # Create a simple config
    config = {
        "embed_dim": 1024,
        "depth": 16,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "num_prototypes": 6,
        "prototype_dim": 64,
        "num_task_protos": 6,
        "lora_r": 64,
        "freeze_trunk": False,
        "trunk": {
            "_target_": "hpt.models.transformer.SimpleTransformer",
            "embed_dim": 1024,
            "depth": 16,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "use_molora": True
        }
    }
    
    try:
        # Create policy
        policy = Policy(**config)
        
        print(f"Policy created successfully!")
        print(f"Embed dim: {policy.embed_dim}")
        print(f"Prototype num: {policy.prototype_num}")
        print(f"Task protos: {policy.num_task_protos}")
        print(f"Num combinations: {policy.num_combinations}")
        print(f"MoLoRA layers: {list(policy.molora_layers.keys())}")
        
        # Check if MoLoRA layers are properly initialized
        if "fc1" in policy.molora_layers:
            fc1 = policy.molora_layers["fc1"]
            print(f"MoLoRA fc1 - in: {fc1.in_features}, out: {fc1.out_features}, combos: {fc1.num_combinations}")
        
        if "fc2" in policy.molora_layers:
            fc2 = policy.molora_layers["fc2"]
            print(f"MoLoRA fc2 - in: {fc2.in_features}, out: {fc2.out_features}, combos: {fc2.num_combinations}")
        
        # Test forward pass
        batch_size = 4
        seq_len = 32
        embed_dim = 1024
        text_dim = 512
        
        tokens = torch.randn(batch_size, seq_len, embed_dim)
        text_features = torch.randn(batch_size, text_dim)
        
        # Test forward
        with torch.no_grad():
            # Simulate the forward process
            B, T, _ = tokens.shape
            
            # Get routing weights
            w_a = torch.softmax(torch.randn(B, 6), dim=-1)
            w_e = torch.softmax(torch.randn(B, 6), dim=-1)
            w_t = policy.task_encoder(text_features)
            
            # 3D routing
            w_route = torch.einsum('bi, bj, bk -> bijk', w_a, w_e, w_t).reshape(B, -1)
            
            # Set routing weights
            policy.molora_layers["fc1"]._current_w_route = w_route
            policy.molora_layers["fc2"]._current_w_route = w_route
            
            print(f"\nForward pass test:")
            print(f"Tokens shape: {tokens.shape}")
            print(f"Text features shape: {text_features.shape}")
            print(f"w_route shape: {w_route.shape}")
            print(f"✓ Forward pass setup successful!")
        
        print("\n" + "=" * 50)
        print("POLICY INITIALIZATION TEST PASSED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ POLICY INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_policy_initialization()
    sys.exit(0 if success else 1)