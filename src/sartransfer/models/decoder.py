"""Small learned heads on frozen features, for the calving-front task (rows 3, 4, 6).

Both heads get the raw grey tile as a second input ("image skip"), because a
1 px front needs pixel-level evidence the 16 px feature grid cannot hold.

GridDecoder  (row 3): frozen (D, 32, 32) grid -> 1x1 reduce -> four x2
             upsampling stages, each concatenating a same-scale map from a tiny
             image encoder -> 4-class logits at tile resolution.
PixelHead    (row 4): learned 1x1 reduction (D -> 64) on the grid, upsampled to
             pixels by AnyUp's attention (computed from the frozen features and
             the image, applied to the reduced values -- exact, see
             anyup_loader.upsample_with_value), concatenated with the image
             encoder's full-resolution map, then two 3x3 convs -> logits.
AnyUpDecoder (row 6): GridDecoder with its two learned stages between the grid
             and 1/4 resolution replaced by AnyUp's attention (to 1/4, guided
             by the image resized to 1/4); the last three stages are row 3's.
GridDecoder on four layers concatenated is row 5 (no new module).

Trainable parameters at D = 1024 / 1280: GridDecoder 0.54 / 0.57 M (0.84 /
0.94 M on four layers), PixelHead 0.11 / 0.13 M, AnyUpDecoder 0.24 / 0.26 M,
against a 7.8 M U-Net.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.LeakyReLU(0.1, inplace=True))


class ImageStem(nn.Module):
    """Tiny image encoder: maps at strides 1, 2, 4, 8, 16 (channels 16, 24, 32, 48, 64).

    `levels` builds only the first maps a head uses (GridDecoder 5, AnyUpDecoder 3,
    PixelHead 1); the maps it does build are unchanged.
    """

    CH = (16, 24, 32, 48, 64)

    def __init__(self, in_ch: int = 1, levels: int = 5):
        super().__init__()
        if not 1 <= levels <= len(self.CH):
            raise ValueError(f"levels must be 1-{len(self.CH)}, got {levels}")
        self.blocks = nn.ModuleList()
        c_prev = in_ch
        for c in self.CH[:levels]:
            self.blocks.append(_conv(c_prev, c))
            c_prev = c

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        maps, x = [], img
        for i, blk in enumerate(self.blocks):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = blk(x)
            maps.append(x)
        return maps                              # [s1, s2, s4, s8, s16][:levels]


_STEM_KEY = re.compile(r"stem\.blocks\.(\d+)\.")


def load_head_state(head: nn.Module, state_dict: dict) -> nn.Module:
    """Load a saved head (rows 3-6), also one saved before the stems were lean.

    Heads saved up to 2026-09-25 hold all five ImageStem levels, also PixelHead
    (uses 1) and AnyUpDecoder (uses 3), whose unused levels never got a gradient.
    Only the keys `stem.blocks.<k>.*` with k >= the head's stem levels are
    dropped; any other missing or unexpected key still raises (strict=True).
    A module without an ImageStem (e.g. a row 1/2 nn.Linear) loads as is.
    """
    stem = getattr(head, "stem", None)
    levels = len(stem.blocks) if isinstance(stem, ImageStem) else None
    if levels is not None:
        state_dict = {k: v for k, v in state_dict.items()
                      if not ((m := _STEM_KEY.match(k)) and int(m.group(1)) >= levels)}
    head.load_state_dict(state_dict, strict=True)
    return head


class GridDecoder(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = 4, width: int = 96):
        super().__init__()
        self.stem = ImageStem()
        self.reduce = nn.Conv2d(feat_dim, width, 1)
        ch = ImageStem.CH
        self.up = nn.ModuleList([
            _conv(width + ch[4], width),          # 1/16: grid + s16
            _conv(width + ch[3], width),          # 1/8
            _conv(width + ch[2], 64),             # 1/4
            _conv(64 + ch[1], 48),                # 1/2
            _conv(48 + ch[0], 32),                # 1/1
        ])
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, feats: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """feats (B, D, gh, gw) float; img (B, 1, H, W) in [0, 1]."""
        maps = self.stem(img)
        x = self.reduce(feats)
        x = self.up[0](torch.cat([x, maps[4]], 1))
        for i, m in zip(range(1, 5), (maps[3], maps[2], maps[1], maps[0])):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.up[i](torch.cat([x, m], 1))
        return self.head(x)


class PixelHead(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = 4, reduced: int = 64):
        super().__init__()
        self.stem = ImageStem(levels=1)          # uses the full-resolution map only
        self.reduce = nn.Conv2d(feat_dim, reduced, 1)
        self.mix = nn.Sequential(_conv(reduced + ImageStem.CH[0], 48), _conv(48, 32))
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, feats: torch.Tensor, img: torch.Tensor, guide: torch.Tensor,
                anyup) -> torch.Tensor:
        """feats (B, D, gh, gw); img (B, 1, H, W) in [0, 1]; guide = ImageNet-normalised RGB."""
        from .anyup_loader import upsample_with_value

        value = self.reduce(feats)
        up = upsample_with_value(anyup, guide, feats, value, out_size=tuple(img.shape[-2:]))
        s1 = self.stem(img)[0]
        return self.head(self.mix(torch.cat([up, s1], 1)))


class AnyUpDecoder(nn.Module):
    """Row 6: AnyUp (frozen) does grid -> 1/4 resolution, row 3's last three
    stages do the rest. Same stem (first three levels), same 1x1 reduction
    width as GridDecoder."""

    def __init__(self, feat_dim: int, n_classes: int = 4, width: int = 96, up_stride: int = 4):
        super().__init__()
        self.stem = ImageStem(levels=3)          # s1, s2, s4
        self.reduce = nn.Conv2d(feat_dim, width, 1)
        self.up_stride = up_stride
        ch = ImageStem.CH
        self.up = nn.ModuleList([
            _conv(width + ch[2], 64),             # 1/4: AnyUp output + s4
            _conv(64 + ch[1], 48),                # 1/2
            _conv(48 + ch[0], 32),                # 1/1
        ])
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, feats: torch.Tensor, img: torch.Tensor, guide: torch.Tensor,
                anyup) -> torch.Tensor:
        """feats (B, D, gh, gw); img (B, 1, H, W) in [0, 1]; guide = ImageNet-normalised RGB."""
        from .anyup_loader import upsample_with_value

        maps = self.stem(img)
        size = (img.shape[-2] // self.up_stride, img.shape[-1] // self.up_stride)
        g = F.interpolate(guide, size=size, mode="bilinear", align_corners=False)
        x = upsample_with_value(anyup, g, feats, self.reduce(feats), out_size=size)
        x = self.up[0](torch.cat([x, maps[2]], 1))
        for i, m in zip((1, 2), (maps[1], maps[0])):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.up[i](torch.cat([x, m], 1))
        return self.head(x)
