"""
Test and validation script for DTM implementation.

This script performs sanity checks to ensure:
1. Backbone is frozen (no gradients)
2. Only Head parameters are trainable
3. Forward pass works with dummy data
4. Loss can be computed and backpropagated
5. Inference works with different T values
"""

import torch
import torch.nn.functional as F

from f5_tts.model import DiT, DTM, DTMHead


def test_dtm_head():
    """Test DTM Head module."""
    print("=" * 60)
    print("Testing DTM Head Module")
    print("=" * 60)
    
    # Create head
    head = DTMHead(
        backbone_dim=1024,
        mel_dim=100,
        hidden_dim=512,
        num_layers=6,
        ff_mult=4,
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    h_t = torch.randn(batch_size, seq_len, 1024)
    y_s = torch.randn(batch_size, seq_len, 100)
    s = torch.rand(batch_size)
    
    # Forward
    v = head(h_t, y_s, s)
    
    # Check output shape
    assert v.shape == (batch_size, seq_len, 100), f"Expected shape {(batch_size, seq_len, 100)}, got {v.shape}"
    
    # Check all parameters are trainable
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in head.parameters())
    assert trainable == total, "All head parameters should be trainable"
    
    print(f"✓ Head forward pass works")
    print(f"✓ Output shape correct: {v.shape}")
    print(f"✓ All parameters trainable: {trainable:,}")
    print()
    
    return head


def test_dtm_frozen_backbone():
    """Test that DTM properly freezes the backbone."""
    print("=" * 60)
    print("Testing DTM with Frozen Backbone")
    print("=" * 60)
    
    # Create small backbone for testing
    backbone = DiT(
        dim=256,
        depth=4,
        heads=4,
        dim_head=64,
        mel_dim=100,
        text_num_embeds=256,
    )
    
    # Create head
    head = DTMHead(
        backbone_dim=256,
        mel_dim=100,
        hidden_dim=128,
        num_layers=2,
    )
    
    # Create DTM
    dtm = DTM(
        backbone=backbone,
        head=head,
        global_timesteps=4,
        mel_spec_kwargs={"n_mel_channels": 100},
    )
    
    # Check backbone is frozen
    backbone_trainable = sum(p.numel() for p in dtm.backbone.parameters() if p.requires_grad)
    assert backbone_trainable == 0, f"Backbone should have 0 trainable params, got {backbone_trainable}"
    
    # Check head is trainable
    head_trainable = sum(p.numel() for p in dtm.head.parameters() if p.requires_grad)
    assert head_trainable > 0, "Head should have trainable parameters"
    
    # Check backbone is in eval mode
    assert not dtm.backbone.training, "Backbone should be in eval mode"
    
    print(f"✓ Backbone frozen: 0 trainable parameters")
    print(f"✓ Head trainable: {head_trainable:,} parameters")
    print(f"✓ Backbone in eval mode")
    print()
    
    return dtm


def test_dtm_training_forward():
    """Test DTM training forward pass (Algorithm 3)."""
    print("=" * 60)
    print("Testing DTM Training Forward (Algorithm 3)")
    print("=" * 60)
    
    # Create small DTM for testing
    backbone = DiT(
        dim=256,
        depth=4,
        heads=4,
        dim_head=64,
        mel_dim=100,
        text_num_embeds=256,
    )
    
    head = DTMHead(
        backbone_dim=256,
        mel_dim=100,
        hidden_dim=128,
        num_layers=2,
    )
    
    dtm = DTM(
        backbone=backbone,
        head=head,
        global_timesteps=4,
        mel_spec_kwargs={"n_mel_channels": 100},
    )
    
    # Create dummy data
    batch_size = 2
    seq_len = 64
    
    mel = torch.randn(batch_size, seq_len, 100)
    text = torch.randint(0, 256, (batch_size, 32))
    lens = torch.tensor([seq_len, seq_len - 10])
    
    # Forward pass
    loss, cond, pred = dtm(mel, text, lens=lens)
    
    # Check loss is valid
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is inf"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Check shapes
    assert pred.shape == mel.shape, f"Prediction shape {pred.shape} != mel shape {mel.shape}"
    
    # Test backward pass
    loss.backward()
    
    # Check gradients
    for name, param in dtm.head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
    
    # Check backbone has no gradients
    for param in dtm.backbone.parameters():
        assert param.grad is None, "Backbone should not have gradients"
    
    print(f"✓ Training forward pass works")
    print(f"✓ Loss value: {loss.item():.6f}")
    print(f"✓ Prediction shape: {pred.shape}")
    print(f"✓ Gradients computed correctly")
    print(f"✓ Backbone has no gradients")
    print()
    
    return dtm


def test_dtm_inference():
    """Test DTM inference (Algorithm 4)."""
    print("=" * 60)
    print("Testing DTM Inference (Algorithm 4)")
    print("=" * 60)
    
    # Create small DTM for testing
    backbone = DiT(
        dim=256,
        depth=4,
        heads=4,
        dim_head=64,
        mel_dim=100,
        text_num_embeds=256,
    )
    
    head = DTMHead(
        backbone_dim=256,
        mel_dim=100,
        hidden_dim=128,
        num_layers=2,
    )
    
    dtm = DTM(
        backbone=backbone,
        head=head,
        global_timesteps=4,
        ode_solver_steps=1,
        mel_spec_kwargs={"n_mel_channels": 100},
    )
    
    # Test inference
    batch_size = 1
    cond_len = 32
    target_len = 64
    
    cond = torch.randn(batch_size, cond_len, 100)
    text = torch.randint(0, 256, (batch_size, 16))
    lens = torch.tensor([cond_len])
    
    # Sample with different T values
    for T in [2, 4, 8]:
        print(f"Testing with T={T} global timesteps...")
        
        out, trajectory = dtm.sample(
            cond=cond,
            text=text,
            duration=target_len,
            lens=lens,
            steps=T,
            seed=42,
        )
        
        # Check output shape
        assert out.shape == (batch_size, target_len, 100), f"Output shape {out.shape} incorrect"
        
        # Check trajectory
        assert trajectory.shape[0] == T + 1, f"Trajectory should have {T+1} steps, got {trajectory.shape[0]}"
        
        # Check conditioning is preserved
        cond_match = torch.allclose(out[0, :cond_len], cond[0], atol=1e-5)
        assert cond_match, "Conditioning should be preserved in output"
        
        print(f"  ✓ Output shape: {out.shape}")
        print(f"  ✓ Trajectory shape: {trajectory.shape}")
        print(f"  ✓ Conditioning preserved")
    
    print(f"\n✓ Inference works with different T values")
    print()
    
    return dtm


def test_memory_usage():
    """Test memory usage fits in RTX 4090 (24GB)."""
    print("=" * 60)
    print("Testing Memory Usage")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        print()
        return
    
    device = torch.device("cuda")
    
    # Create full-size model
    backbone = DiT(
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        mel_dim=100,
        text_num_embeds=1024,
        ff_mult=2,
    ).to(device)
    
    head = DTMHead(
        backbone_dim=1024,
        mel_dim=100,
        hidden_dim=512,
        num_layers=6,
        ff_mult=4,
    ).to(device)
    
    dtm = DTM(
        backbone=backbone,
        head=head,
        global_timesteps=8,
        mel_spec_kwargs={"n_mel_channels": 100},
    ).to(device)
    
    # Test with realistic batch
    batch_size = 4
    seq_len = 512
    
    mel = torch.randn(batch_size, seq_len, 100, device=device)
    text = torch.randint(0, 1024, (batch_size, 128), device=device)
    lens = torch.tensor([seq_len] * batch_size, device=device)
    
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    loss, _, _ = dtm(mel, text, lens=lens)
    loss.backward()
    
    # Check memory
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"✓ Forward + backward pass completed")
    print(f"✓ Peak memory usage: {peak_memory:.2f} GB")
    
    if peak_memory < 24:
        print(f"✓ Fits in RTX 4090 (24 GB)")
    else:
        print(f"⚠ Exceeds RTX 4090 memory ({peak_memory:.2f} GB > 24 GB)")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DTM Implementation Validation")
    print("=" * 60 + "\n")
    
    try:
        # Test individual components
        test_dtm_head()
        test_dtm_frozen_backbone()
        test_dtm_training_forward()
        test_dtm_inference()
        test_memory_usage()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nDTM implementation is ready for training.")
        print("\nTo start training, run:")
        print("  python -m f5_tts.train.train_dtm --config-name DTM_F5TTS_Base")
        print("\nMake sure to:")
        print("  1. Set the correct backbone_checkpoint_path in the config")
        print("  2. Prepare your dataset")
        print("  3. Adjust batch_size_per_gpu based on your GPU memory")
        print()
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print("✗ Tests failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()

