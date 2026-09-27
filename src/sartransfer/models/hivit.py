"""HiViT encoder for the SARATR-X checkpoints -- our own compact implementation.

SARATR-X (Li et al., arXiv 2405.09365; github.com/waterdisappear/SARATR-X)
pretrains a HiViT-Base with masked image modelling on SAR target chips. Only the
encoder is needed here, as a frozen feature extractor, so this module rebuilds
just that, with parameter names matching the released checkpoint so its weights
load strictly. Behaviour follows the published architecture:

    patch_embed  4x4 conv (stride 4) -> 128 ch; each 16 px patch = 4x4 sub-patches
    stage 1      MLP-only blocks at 128        (blocks 0..3 in the v1 checkpoint)
    merge        2x2 sub-patches -> 256        (block 4)
    stage 2      MLP-only blocks at 256        (blocks 5..8)
    merge        2x2 sub-patches -> 512        (block 9)
    + absolute position embedding (14 x 14 at 224 px; interpolated for larger input)
    main stage   attention blocks at 512       (blocks 10..29), 8 heads

Layout verified against the v1 checkpoint `weight/186K_all/checkpoint-800.pth`
on 2026-09-23 (pos embed (1,196,512), no relative position bias, 30 blocks).
The stage layout is read from the checkpoint keys, not assumed.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

LN = lambda d: nn.LayerNorm(d, eps=1e-6)       # noqa: E731  (hivit_base uses eps 1e-6)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)            # softmax(q k^T / sqrt(d)) v
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """Pre-norm block; MLP-only when heads == 0 (the early stages)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = LN(dim) if heads else None
        self.attn = Attention(dim, heads) if heads else None
        self.norm2 = LN(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchMerge(nn.Module):
    """Merge 2x2 sub-patches inside each 16 px patch: C -> 2C."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = LN(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):                       # (B, N, h, w, C)
        x = torch.cat([x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :],
                       x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :]], dim=-1)
        return self.reduction(self.norm(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, in_chans: int = 3, inner: int = 4, patch: int = 16):
        super().__init__()
        self.inner, self.patch = inner, patch
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch // inner, stride=patch // inner)
        self.norm = LN(dim)

    def forward(self, x):                       # (B, 3, H, W) -> (B, N, inner, inner, C)
        B, _, H, W = x.shape
        hp, wp = H // self.patch, W // self.patch
        x = self.proj(x).view(B, -1, hp, self.inner, wp, self.inner)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, hp * wp, self.inner, self.inner, -1)
        return self.norm(x)


class HiViTEncoder(nn.Module):
    """Frozen-feature HiViT: image -> (B, H/16 * W/16, dim) patch tokens, row-major."""

    def __init__(self, layout: list[tuple[str, int]], stem_dim: int, pos_grid: int,
                 heads: int = 8, stem_mlp_ratio: float = 3.0, mlp_ratio: float = 4.0,
                 final_norm: bool = False):
        super().__init__()
        self.patch_embed = PatchEmbed(stem_dim)
        blocks, dim = [], stem_dim
        for kind, n in layout:                  # ("stem", 4), ("merge", 1), ... ("main", 20)
            for _ in range(n):
                if kind == "stem":
                    blocks.append(Block(dim, 0, stem_mlp_ratio))
                elif kind == "merge":
                    blocks.append(PatchMerge(dim)); dim *= 2
                else:
                    blocks.append(Block(dim, heads, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)
        self.n_main = sum(n for k, n in layout if k == "main")
        self.dim = dim
        self.pos_grid = pos_grid
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, pos_grid * pos_grid, dim))
        self.norm = LN(dim) if final_norm else None

    def _pos(self, hp: int, wp: int) -> torch.Tensor:
        pe = self.absolute_pos_embed
        if (hp, wp) == (self.pos_grid, self.pos_grid):
            return pe
        g = self.pos_grid
        pe = pe.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe.float(), size=(hp, wp), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, hp * wp, -1).to(self.absolute_pos_embed.dtype)

    def forward(self, x):
        hp, wp = x.shape[-2] // 16, x.shape[-1] // 16
        x = self.patch_embed(x)
        for blk in self.blocks[:-self.n_main]:
            x = blk(x)
        x = x[..., 0, 0, :] + self._pos(hp, wp)
        for blk in self.blocks[-self.n_main:]:
            x = blk(x)
        return self.norm(x) if self.norm is not None else x


def layout_from_state_dict(sd: dict) -> tuple[list[tuple[str, int]], int, int, bool]:
    """Read (stage layout, stem width, pos grid, has final norm) off checkpoint keys."""
    idx = sorted({int(m.group(1)) for k in sd for m in [re.match(r"blocks\.(\d+)\.", k)] if m})
    kinds = []
    for i in idx:
        if f"blocks.{i}.reduction.weight" in sd:
            kinds.append("merge")
        elif f"blocks.{i}.attn.qkv.weight" in sd:
            kinds.append("main")
        else:
            kinds.append("stem")
    layout: list[tuple[str, int]] = []
    for k in kinds:
        if layout and layout[-1][0] == k:
            layout[-1] = (k, layout[-1][1] + 1)
        else:
            layout.append((k, 1))
    stem_dim = sd["patch_embed.proj.weight"].shape[0]
    n = sd["absolute_pos_embed"].shape[1]
    grid = int(round(math.sqrt(n)))
    assert grid * grid == n, f"position embedding length {n} is not square"
    # A top-level "norm" only counts as the encoder's final norm if its width is the
    # encoder width. The v1 checkpoint has a 3584-wide "norm" that belongs to the
    # pretraining targets (found 2026-09-23 when a strict load failed).
    out_dim = sd["absolute_pos_embed"].shape[-1]
    final_norm = "norm.weight" in sd and tuple(sd["norm.weight"].shape) == (out_dim,)
    return layout, stem_dim, grid, final_norm


def load_hivit(path: str, heads: int = 8) -> HiViTEncoder:
    """Build the encoder from a SARATR-X checkpoint and load it strictly."""
    # weights_only=True refuses any code in the pickle. The SARATR-X file also
    # holds an argparse.Namespace of training args, which is a plain data class,
    # so it is allow-listed explicitly rather than opening the door to everything.
    import argparse
    with torch.serialization.safe_globals([argparse.Namespace]):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    layout, stem_dim, grid, final_norm = layout_from_state_dict(sd)
    model = HiViTEncoder(layout, stem_dim, grid, heads=heads, final_norm=final_norm)
    wanted = set(model.state_dict())
    enc_sd = {k: v for k, v in sd.items() if k in wanted}
    missing = sorted(wanted - set(enc_sd))
    if missing:
        raise RuntimeError(f"checkpoint lacks encoder weights: {missing[:10]}")
    model.load_state_dict(enc_sd, strict=True)
    skipped = sorted(set(sd) - wanted)
    print(f"HiViT loaded: layout {layout}, stem {stem_dim}, pos grid {grid}x{grid}, "
          f"final norm {final_norm}; {len(enc_sd)} tensors used, {len(skipped)} skipped "
          f"(decoder / pretraining targets, e.g. {skipped[:3]})")
    return model
