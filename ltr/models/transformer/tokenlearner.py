import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)

        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))

        return out, x * weight_map


class TokenLearner(nn.Module):
    def __init__(self, C, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        B, _, _, C = x.shape
        Z = torch.zeros(B, self.S, C, device=x.device)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x) # [B, C]
            Z[:, i, :] = Ai
        return Z


class TokenFuser(nn.Module):
    def __init__(self, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S

    def forward(self, y, x):
        B, S, C = y.shape
        B, H, W, C = x.shape

        Y = self.projection(y.view(B, C, S)).view(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).view(B, H*W, S) # [B, HW, S]
        BwY = torch.matmul(Bw, Y)

        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H*W, C)

        out = (BwY + xj).view(B, H, W, C)

        return out


if __name__ == '__main__':
    # B, H, W, C
    # img = torch.Tensor(4, 32, 32, 3)

    x = torch.rand(3, 18, 18, 768)          # torch.Size([4, 64, 48, 96])
    tklr = TokenLearner(S=8)
    tklr_res = tklr(x)                  # torch.Size([4, 8, 96]) B, N, C
    print('tklr_res shape: ', tklr_res.shape)

    tkfr = TokenFuser(768, 8)
    tkfr_res = tkfr(tklr_res, x)      # torch.Size([4, 64, 48, 3])
    print('tkfr_res shape: ', tkfr_res.shape)

    n_parameters = sum(p.numel() for p in tklr.parameters() if p.requires_grad)
    print('----------------------number of params: %d,  %.3f -----------------' % (n_parameters, n_parameters))
