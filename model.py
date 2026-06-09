import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False, pool_type=None):
        """
        pool_type: None, 'standard' (2x2), or 'strip' (2x1)
        """
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Setup residual connection mapping if channels mismatch
        if self.use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()
        
        # Pooling options
        if pool_type == 'standard':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == 'strip':
            self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        res = x
        out = self.conv(x)
        out = self.bn(out)
        
        if self.use_residual:
            out = self.relu(out + self.shortcut(res))
        else:
            out = self.relu(out)
            
        out = self.pool(out)
        return out


class CRNN(nn.Module):
    def __init__(self, num_classes):
        """
        num_classes: Vocabulary Size + 1 (blank token)
        """
        super().__init__()
        
        # CNN Backbone
        # Channels: 64 -> 128 -> 256 -> 256 -> 512 -> 512 -> 512
        self.block1 = ConvBlock(1, 64, use_residual=False, pool_type='standard')      # (64, 32, 128)
        self.block2 = ConvBlock(64, 128, use_residual=False, pool_type='standard')   # (128, 16, 64)
        self.block3 = ConvBlock(128, 256, use_residual=True, pool_type=None)         # (256, 16, 64)
        self.block4 = ConvBlock(256, 256, use_residual=True, pool_type='strip')      # (256, 8, 64)
        self.block5 = ConvBlock(256, 512, use_residual=True, pool_type=None)         # (512, 8, 64)
        self.block6 = ConvBlock(512, 512, use_residual=True, pool_type='strip')      # (512, 4, 64)
        
        # Block 7: Kernel 2x2 with no padding to reduce height further
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)                                            # (512, 3, 63)
        
        # Force the output representation to be exactly shape (Batch, 512, 1, 64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 64))
        
        # Sequence Model
        # Input size: 512 channels
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )
        
        # Output project layer (bidirectional LSTM has output shape 2 * hidden_size = 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 64, 256)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)
        
        # Reduce to height=1, width=64
        out = self.adaptive_pool(out)
        
        # Squeeze height dimension: (Batch, 512, 1, 64) -> (Batch, 512, 64)
        out = out.squeeze(2)
        
        # Permute to (SeqLen=64, Batch, Features=512) for PyTorch LSTM
        out = out.permute(2, 0, 1)
        
        # Pass to bidirectional LSTM
        out, _ = self.lstm(out)  # Output: (64, Batch, 512)
        
        # Final linear projection
        logits = self.fc(out)    # Output: (64, Batch, num_classes)
        return logits
