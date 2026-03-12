import torch
import sys
sys.path.insert(0, 'hpt')

from models.policy import Policy, TaskEncoder, MoLoRALayer, compute_ortho_loss

def test_molora_dimensions():
    """Test MoLoRA layer dimensions"""
    print("=" * 50)
    print("Testing MoLoRA Layer Dimensions")
    print("=" * 50)
    
    # Create a simple MoLoRA layer
    in_features = 1024
    out_features = 1024
    num_combinations = 216  # 6 * 6 * 6
    r = 64
    batch_size = 64
    
    molora = MoLoRALayer(in_features, out_features, num_combinations, r)
    
    # Test forward pass
    x = torch.randn(batch_size, in_features)
    w_route = torch.randn(batch_size, num_combinations)
    w_route = torch.softmax(w_route, dim=-1)
    
    # Set routing weights
    molora._current_w_route = w_route
    
    # Forward pass
    y = molora(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Routing weights shape: {w_route.shape}")
    print(f"LoRA A shape: {molora.lora_A.shape}")
    print(f"LoRA B shape: {molora.lora_B.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (batch_size, out_features), f"Expected output shape {(batch_size, out_features)}, got {y.shape}"
    print("✓ MoLoRA dimensions are correct!")
    print()

def test_task_encoder():
    """Test TaskEncoder dimensions"""
    print("=" * 50)
    print("Testing TaskEncoder Dimensions")
    print("=" * 50)
    
    text_dim = 512
    embed_dim = 1024
    num_task_protos = 6
    batch_size = 64
    
    task_encoder = TaskEncoder(text_dim, embed_dim, num_task_protos)
    
    # Test with 2D text features
    text_features = torch.randn(batch_size, text_dim)
    w_t = task_encoder(text_features)
    
    print(f"Text features shape: {text_features.shape}")
    print(f"Task prototypes shape: {task_encoder.task_prototypes.shape}")
    print(f"Routing weights shape: {w_t.shape}")
    
    assert w_t.shape == (batch_size, num_task_protos), f"Expected {(batch_size, num_task_protos)}, got {w_t.shape}"
    print("✓ TaskEncoder dimensions are correct!")
    print()
    
    # Test with 3D text features
    text_features_3d = torch.randn(batch_size, 10, text_dim)
    w_t_3d = task_encoder(text_features_3d)
    
    print(f"3D text features shape: {text_features_3d.shape}")
    print(f"Routing weights shape: {w_t_3d.shape}")
    
    assert w_t_3d.shape == (batch_size, num_task_protos), f"Expected {(batch_size, num_task_protos)}, got {w_t_3d.shape}"
    print("✓ TaskEncoder 3D dimensions are correct!")
    print()

def test_3d_routing():
    """Test 3D tensor product routing"""
    print("=" * 50)
    print("Testing 3D Tensor Product Routing")
    print("=" * 50)
    
    batch_size = 64
    N_a = 6  # Agent prototypes
    N_e = 6  # Env prototypes
    N_t = 6  # Task prototypes
    
    # Simulate routing weights
    w_a = torch.randn(batch_size, N_a)
    w_e = torch.randn(batch_size, N_e)
    w_t = torch.randn(batch_size, N_t)
    
    # Softmax to get proper routing weights
    w_a = torch.softmax(w_a, dim=-1)
    w_e = torch.softmax(w_e, dim=-1)
    w_t = torch.softmax(w_t, dim=-1)
    
    # 3D tensor product
    w_route = torch.einsum('bi, bj, bk -> bijk', w_a, w_e, w_t).reshape(batch_size, -1)
    
    print(f"w_a shape: {w_a.shape}")
    print(f"w_e shape: {w_e.shape}")
    print(f"w_t shape: {w_t.shape}")
    print(f"w_route shape: {w_route.shape}")
    print(f"Expected w_route shape: ({batch_size}, {N_a * N_e * N_t})")
    
    assert w_route.shape == (batch_size, N_a * N_e * N_t), f"Expected {(batch_size, N_a * N_e * N_t)}, got {w_route.shape}"
    print("✓ 3D routing dimensions are correct!")
    print()

def test_ortho_loss():
    """Test orthogonal loss computation"""
    print("=" * 50)
    print("Testing Orthogonal Loss")
    print("=" * 50)
    
    num_protos = 6
    embed_dim = 1024
    
    # Test with 2D prototypes
    prototypes_2d = torch.randn(num_protos, embed_dim)
    loss_2d = compute_ortho_loss(prototypes_2d)
    
    print(f"2D prototypes shape: {prototypes_2d.shape}")
    print(f"Orthogonal loss (2D): {loss_2d.item():.4f}")
    
    # Test with 3D prototypes (time series)
    T = 32
    prototypes_3d = torch.randn(num_protos, T, embed_dim)
    loss_3d = compute_ortho_loss(prototypes_3d)
    
    print(f"3D prototypes shape: {prototypes_3d.shape}")
    print(f"Orthogonal loss (3D): {loss_3d.item():.4f}")
    print("✓ Orthogonal loss computation works!")
    print()

def test_integration():
    """Test full integration"""
    print("=" * 50)
    print("Testing Full Integration")
    print("=" * 50)
    
    batch_size = 64
    seq_len = 32
    embed_dim = 1024
    text_dim = 512
    
    # Simulate tokens
    tokens = torch.randn(batch_size, seq_len, embed_dim)
    
    # Simulate text features
    text_features = torch.randn(batch_size, text_dim)
    
    # Create task encoder
    task_encoder = TaskEncoder(text_dim, embed_dim, num_task_protos=6)
    
    # Get task routing weights
    w_t = task_encoder(text_features)
    
    # Simulate agent and env routing weights
    w_a = torch.softmax(torch.randn(batch_size, 6), dim=-1)
    w_e = torch.softmax(torch.randn(batch_size, 6), dim=-1)
    
    # 3D routing
    w_route = torch.einsum('bi, bj, bk -> bijk', w_a, w_e, w_t).reshape(batch_size, -1)
    
    # Create MoLoRA layer
    molora = MoLoRALayer(embed_dim, embed_dim, 216, r=64)
    molora._current_w_route = w_route
    
    # Test on each token
    for i in range(seq_len):
        token = tokens[:, i, :]  # [batch_size, embed_dim]
        output = molora(token)
        assert output.shape == (batch_size, embed_dim), f"Token {i}: Expected {(batch_size, embed_dim)}, got {output.shape}"
    
    print(f"Tokens shape: {tokens.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"w_route shape: {w_route.shape}")
    print(f"MoLoRA output shape per token: ({batch_size}, {embed_dim})")
    print("✓ Full integration works correctly!")
    print()

if __name__ == "__main__":
    try:
        test_molora_dimensions()
        test_task_encoder()
        test_3d_routing()
        test_ortho_loss()
        test_integration()
        
        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nDimensions are correctly aligned:")
        print(f"- MoLoRA: [B, in_features] -> [B, out_features]")
        print(f"- TaskEncoder: [B, text_dim] -> [B, N_t]")
        print(f"- 3D Routing: [B, N_a], [B, N_e], [B, N_t] -> [B, N_a*N_e*N_t]")
        print(f"- Orthogonal Loss: [N, D] or [N, T, D] -> scalar")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)