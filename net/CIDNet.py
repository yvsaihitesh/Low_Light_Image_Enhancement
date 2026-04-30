import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.noise_estimator import NoiseEstimator
from net.freq_decomposer import FrequencyDecomposer
from huggingface_hub import PyTorchModelHubMixin


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 max_offset=3.0):
        
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # noise estimator (runs on raw RGB before transform) its a basic CNN that predicts per-pixel noise confidence
        self.noise_estimator = NoiseEstimator(in_channels=3, base_channels=16)

        # frequency decomposer (runs on raw RGB before transform) to spilt image into low-freq (structure) + high-freq (texture/noise).
        self.freq_decomposer = FrequencyDecomposer(channels=3, kernel_size=15, sigma_init=3.0)

        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2, max_offset=max_offset)
        self.HV_LCA2 = HV_LCA(ch3, head3, max_offset=max_offset)
        self.HV_LCA3 = HV_LCA(ch4, head4, max_offset=max_offset)
        self.HV_LCA4 = HV_LCA(ch4, head4, max_offset=max_offset)
        self.HV_LCA5 = HV_LCA(ch3, head3, max_offset=max_offset)
        self.HV_LCA6 = HV_LCA(ch2, head2, max_offset=max_offset)

        self.I_LCA1 = I_LCA(ch2, head2, max_offset=max_offset)
        self.I_LCA2 = I_LCA(ch3, head3, max_offset=max_offset)
        self.I_LCA3 = I_LCA(ch4, head4, max_offset=max_offset)
        self.I_LCA4 = I_LCA(ch4, head4, max_offset=max_offset)
        self.I_LCA5 = I_LCA(ch3, head3, max_offset=max_offset)
        self.I_LCA6 = I_LCA(ch2, head2, max_offset=max_offset)

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype

        #NA-HVI Step 1: estimate noise map from raw RGB input (B,1,H,W)
        noise_map = self.noise_estimator(x)                          

        # FG-HVI Step 2: decompose into low / high frequency
        low_freq, high_freq, blend_img, freq_alpha = self.freq_decomposer(x)

        self.trans._high_freq  = high_freq
        self.trans._freq_alpha = freq_alpha

        # FG-HVI + NA-HVI Step 3: transform blend_img to HVI
        hvi = self.trans.HVIT(blend_img, noise_map=noise_map)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)

        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self, x, noise_map=None):

        # Clearing FG-HVI state so that the loss computation calls wont be reused
        self.trans._high_freq  = None
        self.trans._freq_alpha = None
        hvi = self.trans.HVIT(x, noise_map=noise_map)
        return hvi

    def get_noise_map(self, x):
        return self.noise_estimator(x)
