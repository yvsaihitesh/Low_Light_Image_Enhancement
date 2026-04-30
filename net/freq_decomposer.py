import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecomposer(nn.Module):
    

    def __init__(self, channels: int = 3, kernel_size: int = 15, sigma_init: float = 3.0):
        super(FrequencyDecomposer, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = kernel_size // 2

        # Learnable Gaussian blur kernel is shared across channels
        kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma_init)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]          
        kernel_2d = kernel_2d / kernel_2d.sum()                       
        kernel_init = kernel_2d.unsqueeze(0).unsqueeze(0)             
        kernel_init = kernel_init.repeat(channels, 1, 1, 1)          

        # learnable parameter
        self.blur_kernel = nn.Parameter(kernel_init)

        
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self._init_gate_weights()

    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _init_gate_weights(self):
        for m in self.freq_gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)   # start near neutral blend

    def forward(self, img: torch.Tensor):
        
        kernel = self.blur_kernel
        kernel_sum = kernel.sum(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
        kernel_normed = kernel / kernel_sum

        low_freq = F.conv2d(
            img,
            kernel_normed,
            padding=self.padding,
            groups=self.channels          # here depth wise each channel is blurred independently
        )
        low_freq = torch.clamp(low_freq, 0.0, 1.0)

        high_freq = img - low_freq

        gate_input = torch.cat([img, low_freq, high_freq], dim=1)   
        alpha = self.freq_gate(gate_input)                           

        
        blend_img = alpha * low_freq + (1.0 - alpha) * img
        blend_img = torch.clamp(blend_img, 0.0, 1.0)

        return low_freq, high_freq, blend_img, alpha
