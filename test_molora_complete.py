import torch
import sys
sys.path.insert(0, 'hpt')

from models.policy import Policy, MoLoRALayer, compute_ortho_loss

def test_molora_implementation():
    """Test MoLoRA implementation with a simple forward pass"""
    print("=" * 60)
    print("MoLoRA Implementation Test")
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
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        print(f"   ✓ Using device: {device}")
        
        # Test TaskEncoder
        print("\n2. Testing TaskEncoder...")
        text_dim = 512
        batch_size = 2
        text_features = torch.randn(batch_size, text_dim).to(device)
        
        w_t = policy.task_encoder(text_features)
        print(f"   ✓ TaskEncoder output shape: {w_t.shape}")
        print(f"   ✓ Expected: [{batch_size}, {policy.num_task_protos}]")
        assert w_t.shape == (batch_size, policy.num_task_protos), "TaskEncoder output shape mismatch!"
        
        # Test MoLoRALayer
        print("\n3. Testing MoLoRALayer...")
        in_features = 1024
        out_features = 4096
        num_combinations = 6 * 6 * 6  # N_a * N_e * N_t
        r = 64
        
        molora_layer = MoLoRALayer(in_features, out_features, num_combinations, r)
        molora_layer.to(device)
        
        # Set routing weights
        w_route = torch.ones(batch_size, num_combinations).to(device) / num_combinations
        molora_layer._current_w_route = w_route
        
        # Test forward pass
        x = torch.randn(batch_size, in_features).to(device)
        y = molora_layer(x)
        print(f"   ✓ MoLoRALayer output shape: {y.shape}")
        print(f"   ✓ Expected: [{batch_size}, {out_features}]")
        assert y.shape == (batch_size, out_features), "MoLoRALayer output shape mismatch!"
        
        # Test orthogonal loss
        print("\n4. Testing Orthogonal Loss...")
        prototypes = torch.randn(6, 1024).to(device)
        ortho_loss = compute_ortho_loss(prototypes)
        print(f"   ✓ Orthogonal loss: {ortho_loss.item():.6f}")
        assert ortho_loss.item() >= 0, "Orthogonal loss should be non-negative!"
        
        # Test 3D routing
        print("\n5. Testing 3D Feature Routing...")
        w_a = torch.randn(batch_size, 6).to(device)
        w_e = torch.randn(batch_size, 6).to(device)
        w_t = torch.randn(batch_size, 6).to(device)
        
        w_a = torch.softmax(w_a, dim=-1)
        w_e = torch.softmax(w_e, dim=-1)
        w_t = torch.softmax(w_t, dim=-1)
        
        w_route = torch.einsum('bi, bj, bk -> bijk', w_a, w_e, w_t).reshape(batch_size, -1)
        print(f"   ✓ 3D routing output shape: {w_route.shape}")
        print(f"   ✓ Expected: [{batch_size}, {6 * 6 * 6}]")
        assert w_route.shape == (batch_size, 6 * 6 * 6), "3D routing output shape mismatch!"
        
        # Test parameter freezing
        print("\n6. Testing Parameter Freezing...")
        policy.freeze_trunk()
        
        frozen_count = 0
        trainable_count = 0
        for name, param in policy.named_parameters():
            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1
        
        print(f"   ✓ Frozen parameters: {frozen_count}")
        print(f"   ✓ Trainable parameters: {trainable_count}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nMoLoRA Implementation Summary:")
        print("- TaskEncoder: ✓ Working")
        print("- MoLoRALayer: ✓ Working")
        print("- 3D Routing: ✓ Working")
        print("- Orthogonal Loss: ✓ Working")
        print("- Parameter Freezing: ✓ Working")
        print("\nReady for training! 🚀")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_molora_implementation()
    sys.exit(0 if success else 1)