
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops.layers.torch import Rearrange
from collections import OrderedDict

class TwoScaleMix(nn.Module):  # 192-512-1024
    def __init__(self, in_dim_m, in_dim_s, out_dim):        # middle + small == layer2+layer3
        super().__init__()
        self.up_sample = nn.Sequential(Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2),
                                       nn.LayerNorm(in_dim_s // 4),
                                       )  # 96
        self.channel_proj = nn.Sequential(nn.Linear(in_dim_m + in_dim_s // 4, 512),
                                          nn.Linear(512, 1024),
                                          nn.LayerNorm(1024)
                                          )
        # self.input_proj = nn.Conv2d(1024, out_dim, kernel_size=1)

    def forward(self, xs):
        output = OrderedDict()
        m, s = xs['layer2'], xs['layer3']
        s = self.up_sample(s.permute(0, 2, 3, 1))  # 384-96
        out = torch.cat([m.permute(0, 2, 3, 1), s], dim=3)  # 192+96=288
        backbone_out = self.channel_proj(out).permute(0, 3, 1, 2)
        # out = self.input_proj(backbone_out)  # 288
        output['layer3'] = backbone_out
        return out



class TwoScaleMixV2(nn.Module):  # 192-512-1024
    def __init__(self, in_dim_m, in_dim_s, out_dim):        # middle + small == layer2+layer3
        super().__init__()
        self.down_sample = nn.Sequential(Rearrange('b c (neiw h) (neih w) -> b h w (neiw neih c)', neih=2, neiw=2),
                                       nn.LayerNorm(in_dim_m*4),
                                       )  # 192--768
        self.channel_proj = nn.Sequential(nn.Linear(in_dim_s + in_dim_m * 4, 1024),
                                          # nn.Linear(512, 1024),
                                          nn.LayerNorm(1024)
                                          )
        # self.input_proj = nn.Conv2d(1024, out_dim, kernel_size=1)

    def forward(self, xs):
        output = OrderedDict()
        xs['layer2'] = self.down_sample(xs['layer2'])  # 192-768
        out = torch.cat([xs['layer3'].permute(0, 2, 3, 1), xs['layer2']], dim=3)  # 768+384=1152
        out = self.channel_proj(out).permute(0, 3, 1, 2)
        # out = self.input_proj(backbone_out)  # 288
        output['layer3'] = out
        return output
