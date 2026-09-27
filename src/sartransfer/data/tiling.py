"""Tiling and normalisation for variable-sized SAR scenes.

CaFFe image heights run 405-3,561 px and widths 382-4,476 px, so every encoder
needs fixed-size tiles.
Tiles are padded at the right and bottom edges only, and the padding is tracked
so it can be excluded from loss and metrics alongside the dataset's own
"no information" zone.

Normalisation constants are taken from the official CaFFe code, not recomputed:
`data_processing/glacier_data.py` line 94 uses
`Normalize(mean=0.3047126829624176, std=0.32187142968177795)` over the training
images, on grayscale scaled to [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Official CaFFe training-set statistics, grayscale in [0, 1].
# Source: Nora-Go/Calving_Fronts_and_Where_to_Find_Them,
#         data_processing/glacier_data.py L93-94.
CAFFE_MEAN = 0.3047126829624176
CAFFE_STD = 0.32187142968177795

# ImageNet statistics, what the RGB encoders were pretrained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TileSpec:
    size: int = 512
    overlap: float = 0.0

    @property
    def stride(self) -> int:
        return max(1, int(round(self.size * (1 - self.overlap))))


def tile_positions(h: int, w: int, spec: TileSpec) -> list[tuple[int, int]]:
    """Top-left corners covering an h x w image, padded at right and bottom."""
    s = spec.stride
    rows = max(1, math.ceil(max(0, h - spec.size) / s) + 1)
    cols = max(1, math.ceil(max(0, w - spec.size) / s) + 1)
    return [(r * s, c * s) for r in range(rows) for c in range(cols)]


def n_tiles(h: int, w: int, spec: TileSpec) -> int:
    return len(tile_positions(h, w, spec))


def tile_image(img: np.ndarray, spec: TileSpec,
               fill: int = 0) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Cut an image into tiles.

    Returns (tiles, valid, positions):
      tiles  (N, size, size) same dtype as `img`
      valid  (N, size, size) bool, False where the tile is padding
      positions  top-left corner of each tile in the original image
    """
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {img.shape}")
    h, w = img.shape
    pos = tile_positions(h, w, spec)
    need_h = max(h, max(t for t, _ in pos) + spec.size)
    need_w = max(w, max(l for _, l in pos) + spec.size)

    padded = np.full((need_h, need_w), fill, dtype=img.dtype)
    padded[:h, :w] = img
    real = np.zeros((need_h, need_w), dtype=bool)
    real[:h, :w] = True

    tiles = np.stack([padded[t:t + spec.size, l:l + spec.size] for t, l in pos])
    valid = np.stack([real[t:t + spec.size, l:l + spec.size] for t, l in pos])
    return tiles, valid, pos


NORM_MODES = ("per_image", "caffe", "unit", "per_tile")


def normalise_image(img: np.ndarray, mode: str) -> np.ndarray:
    """Scale a whole uint8 image to float32 BEFORE tiling.

    `per_image`  z-score with the mean and std of the whole image -- the study's
                 "per-image z-score". Tiles keep their brightness differences,
                 so a mostly-ocean tile stays darker than a mostly-glacier one.
    `caffe`      the official fixed CaFFe training statistics (the study's
                 "one fixed global scaling").

    SkyCap reported results to be very sensitive to matching pretraining
    statistics, which is why both settings are carried. Statistics use every
    pixel of the image, as the official CaFFe mean/std script does.
    """
    x = img.astype(np.float32) / 255.0
    if mode == "caffe":
        return (x - CAFFE_MEAN) / CAFFE_STD
    if mode == "per_image":
        mu, sd = float(x.mean()), float(x.std())
        return (x - mu) / (sd if sd > 1e-6 else 1.0)
    if mode == "unit":
        # plain [0, 1] scaling: what SARATR-X v1 saw in pretraining (no Normalize)
        return x
    raise ValueError(f"normalise_image: unknown mode {mode!r}; use 'per_image', 'caffe' or 'unit'")


def normalise(tiles: np.ndarray, mode: str, valid: np.ndarray | None = None) -> np.ndarray:
    """Scale uint8 TILES to float32. Only for `per_tile` and `caffe`.

    `per_tile`   z-score with each 512 px tile's own mean and std. This is what
                 the first probe run (2026-09-22, 0.578 mIoU on SI) actually
                 used, although it was labelled "per_image" at the time. Kept so
                 that result stays reproducible; not one of the study's settings.
    `caffe`      the official fixed CaFFe statistics (same result as
                 `normalise_image(..., "caffe")` followed by tiling).

    A true per-image z-score needs the whole image: use `normalise_image`
    before `tile_image`. Padding is excluded from per-tile statistics when
    `valid` is given.
    """
    x = tiles.astype(np.float32) / 255.0
    if mode == "caffe":
        return (x - CAFFE_MEAN) / CAFFE_STD
    if mode == "per_image":
        raise ValueError("per_image needs the whole image: call normalise_image() "
                         "before tile_image(), not normalise() on tiles")
    if mode != "per_tile":
        raise ValueError(f"unknown mode {mode!r}; use 'per_tile' or 'caffe'")

    out = np.empty_like(x)
    for i in range(len(x)):
        sel = x[i][valid[i]] if valid is not None else x[i]
        if sel.size == 0:
            out[i] = 0.0
            continue
        mu, sd = float(sel.mean()), float(sel.std())
        out[i] = (x[i] - mu) / (sd if sd > 1e-6 else 1.0)
    return out


def to_rgb(tiles: np.ndarray) -> np.ndarray:
    """Replicate a single channel to three, for encoders pretrained on RGB.

    Applied after `normalise`, so both channels-first copies carry the same
    statistics. Shape (N, H, W) -> (N, 3, H, W).
    """
    return np.repeat(tiles[:, None, :, :], 3, axis=1)
