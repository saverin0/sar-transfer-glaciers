"""Calving fronts from zone predictions, and the CaFFe mean distance error (MDE).

Our own implementation of the official CaFFe evaluation route for zone models
(Nora-Go/Calving_Fronts_and_Where_to_Find_Them, read 2026-09-24), step by step:

1. `postprocess_zones`   fill gaps inside the ocean, keep only the largest
                         ocean component (data_postprocessing.postprocess_zone_segmenation)
2. `front_from_zones`    front = ocean pixels with a glacier 4-neighbour; drop
                         8-connected pieces with <= 750 m / resolution pixels
                         (extract_front_from_zones, meter_threshold = 750)
3. `mask_with_bbox`      zero everything outside the image's bounding box
                         (validate_or_test.mask_prediction_with_bounding_box,
                         including its clipping)
4. `front_distances`     for every predicted front pixel the distance to the
                         nearest true front pixel and vice versa (front_error);
                         KD-tree instead of a full distance matrix, same values
5. MDE = mean of ALL distances x resolution, pooled over images
                         (calculate_front_delineation_metric); images without a
                         predicted front are left out and counted.

Zone maps use the official grey values: 0 NA, 64 stone, 127 glacier, 254 ocean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GREY_OF_CLASS4 = np.array([0, 64, 127, 254], dtype=np.uint8)   # na, stone, glacier, ocean
METER_THRESHOLD = 750


# ------------------------------------------------------------------ zone -> front

def postprocess_zones(zone: np.ndarray) -> np.ndarray:
    """Fill gaps in the ocean and keep only its largest connected component."""
    from skimage.measure import label

    mask = zone.copy()
    # gaps: components of "not ocean" other than the largest one become ocean
    lab, n = label(mask != 254, connectivity=2, return_num=True)
    if n > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        biggest = sizes.argmax()
        mask[(lab >= 1) & (lab != biggest)] = 254
    # largest ocean component stays ocean, other ocean pieces become glacier
    lab, n = label(mask >= 254, connectivity=2, return_num=True)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = lab == sizes.argmax()
    mask[mask == 254] = 127
    mask[keep] = 254
    return mask


def front_from_zones(zone: np.ndarray, resolution_m: float,
                     meter_threshold: float = METER_THRESHOLD) -> np.ndarray:
    """Ocean pixels touching glacier (4-neighbourhood), short pieces removed."""
    from skimage.measure import label

    ocean, glacier = zone == 254, zone == 127
    g = np.pad(glacier, 1)
    touch = g[:-2, 1:-1] | g[2:, 1:-1] | g[1:-1, :-2] | g[1:-1, 2:]
    front = ocean & touch
    lab, n = label(front, connectivity=2, return_num=True)
    if n == 0:
        return front
    sizes = np.bincount(lab.ravel())
    too_short = sizes <= meter_threshold / resolution_m
    too_short[0] = True
    return ~too_short[lab] & front


def read_bbox(path: str | Path) -> list[tuple[int, int]]:
    """The four (x, y) corners, rounded, in file order (header line skipped)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    return [tuple(round(float(v)) for v in ln.split(",")) for ln in lines[1:5]]


def mask_with_bbox(front: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    """Official box masking, clipping quirks included."""
    (lu_x, lu_y), (ll_x, ll_y), (rl_x, rl_y), (ru_x, ru_y) = corners
    H, W = front.shape
    lu_x, ll_x = max(lu_x, 0), max(ll_x, 0)
    if ru_x > W: ru_x = W - 1
    if rl_x > W: rl_x = W - 1
    if lu_y > H: lu_y = H - 1
    ll_y, rl_y = max(ll_y, 0), max(rl_y, 0)
    if ru_y > H: ru_y = H - 1
    out = front.copy()
    out[:rl_y, :] = 0
    out[lu_y:, :] = 0
    out[:, :lu_x] = 0
    out[:, rl_x:] = 0
    return out


def front_distances(pred: np.ndarray, true: np.ndarray) -> np.ndarray | None:
    """Pixel distances pred->true and true->pred; None if either front is empty."""
    from scipy.spatial import cKDTree

    p, t = np.argwhere(pred), np.argwhere(true)
    if len(p) == 0 or len(t) == 0:
        return None
    d1, _ = cKDTree(t).query(p)
    d2, _ = cKDTree(p).query(t)
    return np.concatenate([d1, d2])


def zone_front_record(zone_classes4: np.ndarray, true_front: np.ndarray, bbox_path: str | Path,
                      resolution_m: float, meta: dict) -> dict:
    """Full official route for one image, from a 4-class id map (0..3)."""
    zone = GREY_OF_CLASS4[zone_classes4]
    front = front_from_zones(postprocess_zones(zone), resolution_m)
    front = mask_with_bbox(front, read_bbox(bbox_path))
    d = front_distances(front, true_front > 0)
    rec = {**meta, "resolution_m": resolution_m, "front_px": int(front.sum()),
           "has_front": d is not None}
    if d is not None:
        dm = d * resolution_m
        rec.update(n_dist=int(dm.size), sum_m=float(dm.sum()), mean_m=float(dm.mean()))
    else:
        rec.update(n_dist=0, sum_m=0.0, mean_m=float("nan"))
    return rec


def mde_table(records: list[dict] | "pd.DataFrame", groupings=("band", "satellite", "resolution_m", "sensor_res"),
              run: str = "") -> "pd.DataFrame":
    """Pooled MDE (official: all distances pooled) overall and per group."""
    import pandas as pd

    df = pd.DataFrame(records)
    rows = []

    def one(g, key, sub):
        ok = sub[sub["has_front"]]
        rows.append({"run": run, "grouping": g, "group": key,
                     "MDE_m": ok["sum_m"].sum() / max(ok["n_dist"].sum(), 1) if len(ok) else float("nan"),
                     "images": len(sub), "no_front": int((~sub["has_front"]).sum()),
                     "n_dist": int(ok["n_dist"].sum())})

    one("all", "all", df)
    for g in groupings:
        for key, sub in df.groupby(g):
            one(g, str(key), sub)
    return pd.DataFrame(rows)


# ------------------------------------------------------- probe logits -> zone map

def probe_zone_map(features: np.ndarray, positions: np.ndarray, shape: tuple[int, int],
                   weight: np.ndarray, bias: np.ndarray, patch: int = 16) -> np.ndarray:
    """Dense class map for one image from per-patch probe logits.

    features (T, g, g, D) and tile top-left positions (T, 2) as saved by
    features.extract_dataset. Logits are assembled on the patch grid and
    bilinearly upsampled x16 before the argmax, so boundaries are smoother than
    blocky 16 px patches. The patch grid still limits front precision to a few
    pixels -- the gap the AnyUp heads (rows 2, 4, 6 in heads.py) target.
    """
    import torch
    import torch.nn.functional as F

    h, w = shape
    T, g = features.shape[0], features.shape[1]
    logits = features.reshape(-1, features.shape[-1]).astype(np.float32) @ weight.T + bias
    logits = logits.reshape(T, g, g, -1)
    tile = g * patch
    H = max(int(positions[:, 0].max()) + tile, h)
    W = max(int(positions[:, 1].max()) + tile, w)
    grid = np.zeros((H // patch + 1, W // patch + 1, logits.shape[-1]), np.float32)
    for (t, l), lg in zip(positions, logits):
        grid[t // patch:t // patch + g, l // patch:l // patch + g] = lg
    gh, gw = int(np.ceil(h / patch)), int(np.ceil(w / patch))
    x = torch.from_numpy(grid[:gh, :gw]).permute(2, 0, 1)[None]
    up = F.interpolate(x, size=(gh * patch, gw * patch), mode="bilinear", align_corners=False)
    return up[0].argmax(0).numpy()[:h, :w].astype(np.uint8)
