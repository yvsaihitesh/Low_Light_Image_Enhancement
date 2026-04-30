import torch
import torch.nn as nn

pi = 3.141592653589793


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # base k (reciprocal to paper)
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

        self.noise_gain = nn.Parameter(torch.tensor(0.1))   # this gets how much noise raises k
        self.bright_gain = nn.Parameter(torch.tensor(0.05)) # this gets how much brightness lowers k

        self._high_freq = None    
        self._freq_alpha = None   
        self.hue_bias_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           
            nn.Flatten(),                       
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Tanh()                          
        )
        
        nn.init.zeros_(self.hue_bias_net[4].weight)
        nn.init.zeros_(self.hue_bias_net[4].bias)

        
        self._hue_bias = None   
    # NA-HVI forward transform  (Low → HVI space)
    
    def HVIT(self, img, noise_map=None):
        
        eps = 1e-8
        device = img.device
        dtypes = img.dtype

        
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)

        hue[img[:, 2] == value] = (4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps)))[img[:, 2] == value]
        hue[img[:, 1] == value] = (2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps)))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6
        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        # Learnable Hue Rotation (LHR)
        hue_bias = self.hue_bias_net(img) * 0.5          
        self._hue_bias = hue_bias                         
        
        hue_bias_spatial = hue_bias.view(hue_bias.shape[0], 1, 1, 1)  
        hue = (hue + hue_bias_spatial) % 1.0              
        

        # Computing adaptive k

        base_k = self.density_k  
        if noise_map is not None:
            
            noise_map = noise_map.to(dtypes)
            k_map = base_k + self.noise_gain * noise_map - self.bright_gain * value
            k_map = torch.clamp(k_map, min=0.05, max=2.0)
            self.this_k = base_k.item()          
            self._k_map = k_map                  
        else:
            
            k_map = base_k
            self.this_k = base_k.item()
            self._k_map = None

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k_map)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value

        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    
    # NA-HVI inverse transform  (HVI → RGB)
    
    def PHVIT(self, img):
        
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I

        
        if self._k_map is not None:
            k = self._k_map.squeeze(1)           
        else:
            k = self.this_k                      

        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1

       
        if self._hue_bias is not None:
            
            hue_bias_spatial = self._hue_bias.view(self._hue_bias.shape[0], 1, 1)
            h = (h - hue_bias_spatial) % 1.0
            self._hue_bias = None    
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0; hi1 = hi == 1; hi2 = hi == 2
        hi3 = hi == 3; hi4 = hi == 4; hi5 = hi == 5

        r[hi0] = v[hi0]; g[hi0] = t[hi0]; b[hi0] = p[hi0]
        r[hi1] = q[hi1]; g[hi1] = v[hi1]; b[hi1] = p[hi1]
        r[hi2] = p[hi2]; g[hi2] = v[hi2]; b[hi2] = t[hi2]
        r[hi3] = p[hi3]; g[hi3] = q[hi3]; b[hi3] = v[hi3]
        r[hi4] = t[hi4]; g[hi4] = p[hi4]; b[hi4] = v[hi4]
        r[hi5] = v[hi5]; g[hi5] = p[hi5]; b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)

        if self.gated2:
            rgb = rgb * self.alpha

       
        if self._high_freq is not None and self._freq_alpha is not None:
            reinject_weight = 1.0 - self._freq_alpha         
            rgb = rgb + reinject_weight * self._high_freq
            rgb = torch.clamp(rgb, 0.0, 1.0)
            # Clear stored state after use
            self._high_freq = None
            self._freq_alpha = None
        

        return rgb
