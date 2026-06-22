from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 base: int = 32, depth: int = 4) -> None:
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]

        self.downs = nn.ModuleList()
        for i in range(depth):
            in_c = in_ch if i == 0 else chs[i - 1]
            self.downs.append(DoubleConv(in_c, chs[i]))

        bottleneck_ch = chs[-1] * 2
        self.bottleneck = DoubleConv(chs[-1], bottleneck_ch)

        self.ups = nn.ModuleList()
        cur = bottleneck_ch
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(cur, chs[i], 2, stride=2))
            self.ups.append(DoubleConv(chs[i] * 2, chs[i]))
            cur = chs[i]

        self.out_conv = nn.Conv2d(base, out_ch, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat([x, skips[i // 2]], dim=1)
            x = self.ups[i + 1](x)

        return self.tanh(self.out_conv(x))
