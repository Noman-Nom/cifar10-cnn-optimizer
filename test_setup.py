"""
Quick test script to verify setup is working correctly.
"""
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing CIFAR-10 HPO Framework Setup...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from data.loaders import get_cifar10_loaders
    from models.cnn import create_cnn_from_config
    from utils.trainer import ModelTrainer, EarlyStopping
    from optimizers.random_search import RandomSearch
    from optimizers.pso import ParticleSwarmOptimizer
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test device
print("\n2. Testing device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
if device.type == 'cuda':
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠ Warning: CUDA not available. Training will be slow on CPU.")

# Test data loader
print("\n3. Testing data loader...")
try:
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=32, num_workers=2, augment=True
    )
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches: {len(val_loader)}")
    print(f"   ✓ Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"   ✓ Batch shape: {images.shape}, labels: {labels.shape}")
        break
except Exception as e:
    print(f"   ✗ Data loader error: {e}")
    sys.exit(1)

# Test model creation
print("\n4. Testing model creation...")
try:
    config = {
        'num_conv_blocks': 3,
        'conv_channels_base': 64,
        'fc_hidden': 256,
        'dropout': 0.3
    }
    model = create_cnn_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created: {num_params:,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"   ✓ Forward pass OK: {output.shape}")
except Exception as e:
    print(f"   ✗ Model error: {e}")
    sys.exit(1)

# Test optimizers
print("\n5. Testing optimizers...")
try:
    search_space = {
        'learning_rate': {'type': 'float', 'min': 0.0001, 'max': 0.01, 'scale': 'log'},
        'batch_size': {'type': 'int', 'min': 32, 'max': 64, 'scale': 'linear'}
    }
    
    rs = RandomSearch(search_space, seed=42)
    config = rs.sample_config()
    print(f"   ✓ Random Search: {config}")
    
    pso = ParticleSwarmOptimizer(search_space, population_size=5, seed=42)
    print(f"   ✓ PSO initialized")
except Exception as e:
    print(f"   ✗ Optimizer error: {e}")
    sys.exit(1)

# Test config loading
print("\n6. Testing config loading...")
try:
    import yaml
    config_path = Path(__file__).parent / 'experiments' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Config loaded: {len(config)} sections")
    else:
        print(f"   ⚠ Config file not found: {config_path}")
except Exception as e:
    print(f"   ✗ Config error: {e}")

print("\n" + "=" * 60)
print("✓ Setup test completed successfully!")
print("=" * 60)
print("\nYou can now run experiments with:")
print("  cd experiments")
print("  python run_experiment.py")

