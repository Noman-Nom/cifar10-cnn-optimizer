"""
Configurable CNN architecture for CIFAR-10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModernCNN(nn.Module):
    """
    Modern CNN architecture for CIFAR-10 with configurable hyperparameters.
    
    Architecture:
    - Variable number of convolutional blocks
    - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
    - Adaptive pooling to handle variable sizes
    - Fully connected layers with dropout
    """
    
    def __init__(self, num_conv_blocks=3, conv_channels=[64, 128, 256], 
                 fc_hidden=256, dropout=0.3, num_classes=10):
        """
        Initialize CNN model.
        
        Args:
            num_conv_blocks (int): Number of convolutional blocks
            conv_channels (list): List of channel sizes for each conv block
            fc_hidden (int): Hidden size for fully connected layers
            dropout (float): Dropout probability
            num_classes (int): Number of output classes (10 for CIFAR-10)
        """
        super(ModernCNN, self).__init__()
        
        self.num_conv_blocks = num_conv_blocks
        self.conv_channels = conv_channels[:num_conv_blocks]  # Trim to num_conv_blocks
        
        # Build convolutional layers
        self.features = nn.ModuleList()
        in_channels = 3  # CIFAR-10 has 3 RGB channels
        
        for i, out_channels in enumerate(self.conv_channels):
            self.features.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size after conv layers
        # After 3 conv blocks with MaxPool2d(2): 32 -> 16 -> 8 -> 4
        # With adaptive pooling, we ensure 4x4 output
        flattened_size = self.conv_channels[-1] * 4 * 4
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He/Kaiming initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        for feature_block in self.features:
            x = feature_block(x)
        
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def create_cnn_from_config(config):
    """
    Create CNN model from hyperparameter configuration dictionary.
    
    Args:
        config (dict): Hyperparameter configuration with keys:
            - num_conv_blocks (int): Number of conv blocks
            - conv_channels_base (int): Base channel size (will create [base, base*2, base*4, ...])
            - fc_hidden (int): FC hidden size
            - dropout (float): Dropout probability
    
    Returns:
        ModernCNN: Configured model
    """
    num_conv_blocks = config.get('num_conv_blocks', 3)
    conv_channels_base = config.get('conv_channels_base', 64)
    fc_hidden = config.get('fc_hidden', 256)
    dropout = config.get('dropout', 0.3)
    
    # Create channel list: [base, base*2, base*4, ...]
    conv_channels = [conv_channels_base * (2 ** i) for i in range(num_conv_blocks)]
    
    # Cap channels at reasonable maximum
    conv_channels = [min(c, 512) for c in conv_channels]
    
    model = ModernCNN(
        num_conv_blocks=num_conv_blocks,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        dropout=dropout,
        num_classes=10
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = {
        'num_conv_blocks': 3,
        'conv_channels_base': 64,
        'fc_hidden': 256,
        'dropout': 0.3
    }
    
    model = create_cnn_from_config(config)
    print(f"Model: {model}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR-10 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

