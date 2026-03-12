import torch
import sys
sys.path.insert(0, 'hpt')

from models.policy import Policy, MoLoRALayer

def test_molora_3d_input():
    """Test MoLoRALayer with 3D input (batch, seq_len, embed_dim)"""
    print("=" * 60)
    print("MoLoRA 3D Input Test")
    print("=" * 60)
    
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
        print("\n1. Creating Policy...")
        policy = Policy(**config)
        print("   ✓ Policy created successfully!")
        
        # Test MoLoRALayer with 3D input
        print("\n2. Testing MoLoRALayer with 3D input...")
        in_features = 1024
        out_features = 4096
        num_combinations = 6 * 6 * 6  # N_a * N_e * N_t
        r = 64
        batch_size = 4
        seq_len = 32
        
        molora_layer = MoLoRALayer(in_features, out_features, num_combinations, r)
        
        # Test with 2D input
        print("\n   Testing 2D input [B, D]...")
        x_2d = torch.randn(batch_size, in_features)
        w_route_2d = torch.ones(batch_size, num_combinations) / num_combinations
        molora_layer._current_w_route = w_route_2d
        y_2d = molora_layer(x_2d)
        print(f"   ✓ Input shape: {x_2d.shape}")
        print(f"   ✓ Output shape: {y_2d.shape}")
        print(f"   ✓ Expected: [{batch_size}, {out_features}]")
        assert y_2d.shape == (batch_size, out_features), "2D output shape mismatch!"
        
        # Test with 3D input
        print("\n   Testing 3D input [B, T, D]...")
        x_3d = torch.randn(batch_size, seq_len, in_features)
        w_route_3d = torch.ones(batch_size, num_combinations) / num_combinations
        molora_layer._current_w_route = w_route_3d
        y_3d = molora_layer(x_3d)
        print(f"   ✓ Input shape: {x_3d.shape}")
        print(f"   ✓ Output shape: {y_3d.shape}")
        print(f"   ✓ Expected: [{batch_size}, {seq_len}, {out_features}]")
        assert y_3d.shape == (batch_size, seq_len, out_features), "3D output shape mismatch!"
        
        print("\n" + "=" * 60)
        print("ALL 3D INPUT TESTS PASSED! ✓")
        print("=" * 60)
        print("\nMoLoRA correctly handles both 2D and 3D inputs!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_molora_3d_input()
    sys.exit(0 if success else 1)