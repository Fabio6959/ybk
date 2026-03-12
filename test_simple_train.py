import torch
import sys
sys.path.insert(0, 'hpt')

from models.policy import Policy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

def simple_train_test():
    """Simple training test without wandb"""
    print("=" * 50)
    print("Simple Training Test")
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
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        print(f"Using device: {device}")
        
        # Create dummy data
        batch_size = 4
        seq_len = 32
        embed_dim = 1024
        action_dim = 4
        
        # Create a simple dataset
        num_samples = 100
        states = torch.randn(num_samples, seq_len, 1)
        actions = torch.randn(num_samples, seq_len, action_dim)
        images = torch.randn(num_samples, 1, 3, 224, 224)
        
        dataset = TensorDataset(states, actions, images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"\nDataset created: {num_samples} samples")
        print(f"Batch size: {batch_size}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        
        # Training loop
        print("\nStarting training loop...")
        for epoch in range(3):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (state, action, image) in enumerate(dataloader):
                # Move to device
                state = state.to(device)
                action = action.to(device)
                image = image.to(device)
                
                # Create batch dict
                batch = {
                    "domain": ["test"],
                    "data": {
                        "state": state,
                        "action": action,
                        "image": image,
                    }
                }
                
                # Forward pass
                try:
                    loss = policy.compute_loss(batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 5 == 0:
                        print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"  ❌ Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
        
        print("\n" + "=" * 50)
        print("TRAINING TEST COMPLETED SUCCESSFULLY! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TRAINING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = simple_train_test()
    sys.exit(0 if success else 1)