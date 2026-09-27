"""Small U-Net trained from scratch -- the reference the frozen encoders are compared to.

Our own compact implementation, modelled on the CaFFe baseline's choices
(Gourmelon et al. 2022: U-Net, 256 px patches, leaky ReLU with slope 0.1). Their
ASPP bottleneck is left out to keep this a plain reference; that difference is
stated wherever the numbers are compared.

Single-channel input (SAR amplitude), three zone classes out (stone, glacier,
ocean); the no-information zone and padding are ignored by the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1, inplace=True),
    )


class UNet(nn.Module):
    """4 down-sampling levels, so inputs must be multiples of 16 px."""

    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([_block(in_ch, c[0])] + [_block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([skips.pop(), x], dim=1))
        return self.head(x)
