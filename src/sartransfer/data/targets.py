"""Turn pixel-level CaFFe labels into patch-level targets for a linear probe.

The probe reads one vector per encoder patch (16x16 px for every DINOv3 ViT), so
each patch needs a single label. Zones are area-shaped and survive this; calving
fronts are 1 px lines and do not, which is why the feature-ceiling measurement
probes zones and the front task gets an upsampling head instead.

Zone coding, verified against the official code
(`data_processing/glacier_zones_data.py` L12-17):

    0 = NA (no information)   64 = stone   127 = glacier   254 = ocean w/ melange

`0` is not a class. It marks radar shadow, layover and area outside the swath,
and the paper excludes those pixels from loss and metrics. Here they are folded
into the valid mask, alongside tile padding.
"""

from __future__ import annotations

import numpy as np

# grey value -> class id, official mapping
ZONE_VALUES = (0, 64, 127, 254)
ZONE_NAMES = ("na", "stone", "glacier", "ocean")
VALUE_TO_CLASS = {64: 0, 127: 1, 254: 2}      # NA deliberately absent
CLASS_NAMES = ("stone", "glacier", "ocean")
N_CLASSES = 3

# 4-class variant for the fronts (notebook 06 onward), in the official order: the model must
# predict NA itself, like the CaFFe baseline, so it never draws fronts inside
# radar shadow / outside the swath.
VALUE_TO_CLASS4 = {0: 0, 64: 1, 127: 2, 254: 3}
CLASS_NAMES4 = ("na", "stone", "glacier", "ocean")
N_CLASSES4 = 4

IGNORE = 255                                   # target value for excluded patches


def class_names(n: int) -> tuple[str, ...]:
    """Names for a 3-class (stone/glacier/ocean) or 4-class (+ na) problem."""
    return {N_CLASSES: CLASS_NAMES, N_CLASSES4: CLASS_NAMES4}[n]


def zones_to_class(mask: np.ndarray, with_na: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw zone mask to class ids plus a validity mask.

    Returns (classes, valid) at pixel resolution. Default: 0..2 (stone, glacier,
    ocean) where valid, IGNORE on NA. `with_na=True`: 0..3 (na, stone, glacier,
    ocean), every known pixel valid. Unknown grey values raise.
    """
    out = np.full(mask.shape, IGNORE, dtype=np.uint8)
    valid = np.zeros(mask.shape, dtype=bool)
    for value, cls in (VALUE_TO_CLASS4 if with_na else VALUE_TO_CLASS).items():
        hit = mask == value
        out[hit] = cls
        valid |= hit
    known = valid | (mask == 0)
    if not known.all():
        stray = np.unique(mask[~known])
        raise ValueError(f"unexpected zone values {stray.tolist()}; expected {ZONE_VALUES}")
    return out, valid


def patch_labels(classes: np.ndarray, valid: np.ndarray, patch: int = 16,
                 min_purity: float = 0.0, n_classes: int = N_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """Reduce pixel labels to one label per patch by majority vote.

    classes, valid : (H, W), H and W already multiples of `patch`
    min_purity     : 0.0 keeps every patch that has any valid pixel (majority
                     vote). Raise it to drop mixed patches -- e.g. 0.75 keeps
                     only patches where one class holds 75 % of valid pixels.
                     Note that discarding mixed patches removes exactly the zone
                     boundaries the task is about, so the default keeps them.

    Returns (labels, patch_valid), both (H//patch, W//patch).
    """
    h, w = classes.shape
    if h % patch or w % patch:
        raise ValueError(f"shape {classes.shape} is not a multiple of patch {patch}")
    gh, gw = h // patch, w // patch

    # (gh, gw, patch*patch) blocks
    blocks = (classes.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
              .reshape(gh, gw, patch * patch))
    vblocks = (valid.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
               .reshape(gh, gw, patch * patch))

    counts = np.zeros((gh, gw, n_classes), dtype=np.int32)
    for cls in range(n_classes):
        counts[..., cls] = ((blocks == cls) & vblocks).sum(axis=-1)

    n_valid = counts.sum(axis=-1)
    labels = counts.argmax(axis=-1).astype(np.uint8)
    keep = n_valid > 0
    if min_purity > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = counts.max(axis=-1) / np.maximum(n_valid, 1)
        keep &= purity >= min_purity

    labels[~keep] = IGNORE
    return labels, keep


def tile_patch_targets(zone_tiles: np.ndarray, pad_valid: np.ndarray,
                       patch: int = 16, min_purity: float = 0.0, with_na: bool = False):
    """Patch targets for a stack of zone tiles.

    zone_tiles : (N, tile, tile) raw grey values
    pad_valid  : (N, tile, tile) False where the tile is padding

    Returns (labels, valid) of shape (N, tile//patch, tile//patch).
    """
    labels, valids = [], []
    for z, pv in zip(zone_tiles, pad_valid):
        cls, ok = zones_to_class(z, with_na=with_na)
        ok &= pv                                # padding is not a label either
        lab, keep = patch_labels(cls, ok, patch=patch, min_purity=min_purity,
                                 n_classes=N_CLASSES4 if with_na else N_CLASSES)
        labels.append(lab)
        valids.append(keep)
    return np.stack(labels), np.stack(valids)
