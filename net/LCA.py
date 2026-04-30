import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import *



# Deformable Cross-Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias, max_offset=3.0):
        super(CAB, self).__init__()
        self.num_heads   = num_heads
        self.max_offset  = max_offset
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q        = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                  groups=dim, bias=bias)
        self.kv           = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.offset_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.offset_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.offset_proj  = nn.Conv2d(dim, num_heads * 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _deformable_sample(self, feat, offsets):
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype),
            indexing='ij')
        base_grid     = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        mean_offset   = offsets.view(B, self.num_heads, 2, H, W).mean(dim=1)
        norm_offset   = torch.cat([mean_offset[:, 0:1] / (W / 2.0),
                                   mean_offset[:, 1:2] / (H / 2.0)], dim=1).permute(0, 2, 3, 1)
        deformed_grid = torch.clamp(base_grid + norm_offset, -1, 1)
        return F.grid_sample(feat, deformed_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def forward(self, x, y):
        b, c, h, w  = x.shape
        q           = self.q_dwconv(self.q(x))
        raw         = F.relu(self.offset_conv1(y))
        raw         = F.relu(self.offset_conv2(raw))
        offsets     = torch.tanh(self.offset_proj(raw)) * self.max_offset
        y_def       = self._deformable_sample(y, offsets)
        k, v        = self.kv(y_def).chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        out  = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h, w=w)
        return self.project_out(out)


# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        mid = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, kernel_size=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.ones_(self.fc[2].bias)   

    def forward(self, x):
        scale = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * scale



# Spatial Attention (SA)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)   

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale      = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in  = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv      = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                     stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv1     = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                     stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2     = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                     stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh        = nn.Tanh()

        
        self.ca = ChannelAttention(hidden_features, reduction=4)

    def forward(self, x):
        x       = self.project_in(x)
        x1, x2  = self.dwconv(x).chunk(2, dim=1)
        x1      = self.Tanh(self.dwconv1(x1)) + x1
        x2      = self.Tanh(self.dwconv2(x2)) + x2
        x       = self.ca(x1 * x2)      
        return self.project_out(x)



class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, max_offset=3.0):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn  = CAB(dim, num_heads, bias, max_offset=max_offset)
        
        self.sa   = SpatialAttention(kernel_size=7)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return self.sa(x)



class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, max_offset=3.0):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn  = CAB(dim, num_heads, bias=bias, max_offset=max_offset)
        
        self.sa   = SpatialAttention(kernel_size=7)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return self.sa(x)
