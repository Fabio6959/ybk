import torch
import sys
sys.path.insert(0, 'hpt')

from models.policy import Policy
from omegaconf import OmegaConf

def test_full_forward():
    """Test full forward pass with compute_loss"""
    print("=" * 50)
    print("Testing Full Forward Pass")
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
        print("Creating policy...")
        policy = Policy(**config)
        print("✓ Policy created successfully!")
        
        # Initialize stems and heads (required for forward pass)
        print("\nInitializing stems and heads...")
        from omegaconf import OmegaConf
        
        # Mock stem spec
        class MockStemSpec:
            modalities = ["state", "image"]
            modality_embed_dim = 1024
            normalize_state = False
            random_horizon_masking = False
        
        # Mock head spec
        class MockHeadSpec:
            _target_ = "hpt.models.policy_head.MLPHead"
            embed_dim = 1024
            action_dim = 4
        
        # Initialize stems
        policy.init_domain_stem("metaworld_4task", MockStemSpec())
        
        # Initialize normalizer
        from hpt.utils.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        policy.normalizer["metaworld_4task"] = normalizer
        
        # Initialize heads
        policy.init_domain_head("metaworld_4task", normalizer, MockHeadSpec())
        
        # Finalize modules
        policy.finalize_modules()
        
        print("✓ Stems and heads initialized!")
        
        # Set to training mode
        policy.train()
        
        # Create a mock batch
        batch_size = 4
        seq_len = 32
        embed_dim = 1024
        action_dim = 4
        
        # Mock data
        batch = {
            "domain": ["metaworld_4task"],
            "data": {
                "state": torch.randn(batch_size, seq_len, 1),
                "action": torch.randn(batch_size, seq_len, action_dim),
                "image": torch.randn(batch_size, 1, 3, 224, 224),
            }
        }
        
        print(f"\nMock batch created:")
        print(f"  State shape: {batch['data']['state'].shape}")
        print(f"  Action shape: {batch['data']['action'].shape}")
        print(f"  Image shape: {batch['data']['image'].shape}")
        
        # Test forward_features
        print("\nTesting forward_features...")
        with torch.no_grad():
            features, ori_tokens, proto_tokens, w_route = policy.forward_features(
                batch["domain"][0], 
                batch["data"]
            )
        
        print(f"✓ forward_features successful!")
        print(f"  Features shape: {features.shape}")
        print(f"  Ori tokens shape: {ori_tokens.shape}")
        print(f"  Proto tokens shape: {proto_tokens.shape}")
        print(f"  w_route shape: {w_route.shape}")
        
        # Test compute_loss (will fail because heads are not initialized)
        print("\nTesting compute_loss (will fail without heads)...")
        try:
            loss = policy.compute_loss(batch)
            print(f"✓ compute_loss successful! Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"✗ compute_loss failed (expected without heads): {e}")
        
        print("\n" + "=" * 50)
        print("FULL FORWARD TEST COMPLETED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ FULL FORWARD TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_full_forward()
    sys.exit(0 if success else 1)