import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseEstimator(nn.Module):
    
    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super(NoiseEstimator, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False)

        
        self.out_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=1, bias=True)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(f1)) + f1          
        f3 = self.act(self.conv3(f2))
        noise_map = self.sigmoid(self.out_conv(f3))  
        return noise_map
