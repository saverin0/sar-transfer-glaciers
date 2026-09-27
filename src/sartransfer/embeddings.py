"""Look at the frozen encoder features directly: nothing is trained and no result file is touched.

For a few test images (default: the two images of the README animations, one hard,
one easy) and the three encoders at the layers the study used, the patch features
are made exactly as in the study: `features._prepare` (caffe normalisation, 512 px
tiles, overlap 0), then `FrozenEncoder.encode(tiles, layer=...)`. Each image's
per-tile patch grids are stitched into ONE patch map with the tile positions
(16 px per patch) and cropped to the image. The 4-class patch labels (majority
vote, `with_na=True`: 0 NA, 1 stone, 2 glacier, 3 ocean) and the validity mask
come from the same `_prepare` call and are stitched the same way, so labels and
features line up patch for patch. Valid = inside the image; NA patches are valid
and labelled NA. The pixel zones are stitched too; the query rule counts them.

Three descriptive views:

1. `plot_embedding_maps`: PCA 1-3 as RGB and a k-means map (k = 4), per image and encoder.
2. `plot_similarity`: cosine similarity from one query patch picked by a fixed rule
   (`query_patch`): the purest (>= 90 %) ocean patch closest to the true front that
   does not itself contain the front; ties toward the middle of the front. The top-100
   matches are boxed and their class shares given; also the shares of the top
   k20 = max(10, round(0.2 * n)), n = the image's valid patches of the query's class,
   the like-for-like comparison between images (the same 20 % of that class on each).
   With the ocean query this is the direct check of the melange explanation (icy
   water that looks like glacier); a glacier query, same rule, is the contrast.
3. `plot_margins`: per patch cos(ocean centroid) - cos(glacier centroid), histograms of
   glacier vs ocean patches, and the rank AUC of that margin.

With AnyUp (`extract(..., anyup=...)`), the same features are also upsampled by AnyUp, the frozen
image-guided upsampler of heads 2, 4 and 6: `upsample_with_value(anyup, guide, feats, feats)`
under bf16 autocast, exactly as the reported runs (`anyup_tiles`).

A. `anyup_q`: AnyUp at 1/4 resolution over the whole image, as head 6 (per 512 px tile the guide
   resized bilinearly to 128 x 128, output 128 x 128), stitched into one grid of 4 px blocks.
   `quarter_view` turns it into an image entry (block zone = majority of its pixels), so views
   1 and 3 run on blocks (`features="anyup_quarter"`); view 2 is `plot_similarity_anyup`: the
   SAME query patch, its 16 blocks averaged and left out, top 16 x 100 and 16 x k20 blocks (the
   same area as the raw lists).
B. `crop_full`: AnyUp at full resolution (as heads 2 and 4) on the tile holding the ocean query's
   centre, cut to a 256 px window around the query (`crop_window`). `plot_zoom` compares raw,
   AnyUp 1/4 and AnyUp full in that window, pixel by pixel (`crop_views`).

`save_gif_arrays` keeps the drawn maps (uint8 / float16, no features) for a GIF made later on the PC.

Everything describes single images. It explains; it scores nothing and changes no choice.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENCODER_LAYERS = {"cradio-v4-h": 32, "dinov3-l-sat": 21,     # the tested heads (predmaps.PRED_RUNS)
                  # arm-2 layer sweep choice (08_heads step 1, docs/results.md "Arm 2: layer sweep")
                  "dinov3-l-photo": 18}
STEMS = ("COL_2020-03-08_S1_20_3_094",                        # hard: Columbia, Sentinel-1
         "Mapple_2017-09-08_S1_20_2_009")                     # easy: Mapple, Sentinel-1
NORM, TILE = "caffe", 512                                     # as in the reported test runs
CLASSES4 = ("na", "stone", "glacier", "ocean")                # patch label ids 0..3 (with_na=True)
NA, STONE, GLACIER, OCEAN = range(4)
TOP_K, N_CLUSTERS, SEED = 100, 4, 0
QUERY_PURITY = 0.9                                            # query patch: >= 90 % of its pixels in its class
QUERY_RULE = ("the purest (>= {pct:.0f} %) {cls} patch closest to the true front that does not itself contain "
              "the front; ties toward the middle of the front")

# Zones in the colours of the GIFs and the viewer (scripts/make_gifs.py, view_results.py).
# Clusters get a separate set, so a cluster colour never reads as a zone.
ZONE_COLOURS = ("#111111", "#9e9e9e", "#bfe6ff", "#1f3b8c")   # na, stone, glacier, ocean
ZONE_LABELS = ("NA (no data)", "stone", "glacier", "ocean incl. melange")
CLUSTER_COLOURS = ("#eda100", "#e87ba4", "#008300", "#4a3aa7")
MATCH_COLOUR, QUERY_COLOUR = "#ff33cc", "#ffd400"
SIM_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")   # one hue, light -> dark
HIST_ALPHA, HIST_LINES = 0.45, {"glacier": "#bfe6ff", "ocean": "#1f3b8c"}   # margin histograms
FRONT_COLOUR = "#ff2020"                                      # true front, as the viewer (view_results.py)

# AnyUp views (A: whole image at 1/4, B: full resolution in one window)
BLOCK = 4                                                     # A: 4 px blocks (128 x 128 per 512 px tile, head 6)
CROP = 256                                                    # B: window size, px
KMEANS_MAX_FIT = 20_000                                       # A: k-means fitted on <= this many blocks (fixed seed)
FEATURES = ("raw", "anyup_quarter", "anyup_full_crop")        # the CSV's `features` column
ZOOM_VERSIONS = {"raw": "raw patches", "anyup_quarter": "AnyUp 1/4", "anyup_full": "AnyUp full"}   # B, per pixel


# ------------------------------------------------------------------ features and labels

def stitch(grids: np.ndarray, positions: np.ndarray, shape: tuple[int, int], patch: int) -> np.ndarray:
    """Per-tile grids (T, g, g, ...) -> one map (ceil(h / patch), ceil(w / patch), ...) of the image.

    `positions` are the tiles' top-left pixels, as `tile_image` returns them. `patch` = 16
    for patch grids (features, labels, validity), 1 for pixel tiles (image, zones). Every
    cell must come from exactly one tile (overlap 0, as in the study); the tiles' padding
    beyond the image is cropped away.
    """
    h, w = shape
    g = grids.shape[1]
    gh, gw = -(-h // patch), -(-w // patch)
    out = np.zeros((gh, gw) + grids.shape[3:], grids.dtype)
    hits = np.zeros((gh, gw), np.int32)
    for (top, left), grid in zip(np.asarray(positions), grids):
        if top % patch or left % patch:
            raise ValueError(f"tile at ({top}, {left}) is not on the {patch} px grid")
        r, c = top // patch, left // patch
        nr, nc = min(g, gh - r), min(g, gw - c)
        out[r:r + nr, c:c + nc] = grid[:nr, :nc]
        hits[r:r + nr, c:c + nc] += 1
    if (hits != 1).any():
        raise ValueError(f"{int((hits == 0).sum())} cells not covered, {int((hits > 1).sum())} covered "
                         "more than once: stitching needs tiles with overlap 0")
    return out


def front_patches(front: np.ndarray, patch: int) -> np.ndarray:
    """Pixel front mask (h, w) -> patch mask: True where the patch holds a front pixel."""
    h, w = front.shape
    gh, gw = -(-h // patch), -(-w // patch)
    padded = np.zeros((gh * patch, gw * patch), bool)
    padded[:h, :w] = front > 0
    return padded.reshape(gh, patch, gw, patch).any(axis=(1, 3))


def pixel_shares(zones: np.ndarray, patch: int) -> np.ndarray:
    """Pixel zones (h, w), ids 0..3 -> (ceil(h / patch), ceil(w / patch), 4): each patch's share
    of each class. Padded like `front_patches`; only the pixels inside the image are counted."""
    h, w = zones.shape
    gh, gw = -(-h // patch), -(-w // patch)
    padded = np.full((gh * patch, gw * patch), -1, np.int16)
    padded[:h, :w] = zones
    blocks = padded.reshape(gh, patch, gw, patch)
    counts = np.stack([(blocks == k).sum(axis=(1, 3)) for k in range(len(CLASSES4))], -1)
    n_in = (blocks >= 0).sum(axis=(1, 3))
    if (counts.sum(-1) != n_in).any():
        raise ValueError("zone ids outside 0..3 inside the image")
    return counts / n_in[..., None]


def _entry(it: dict, sar_path: Path, data_root: Path, patch: int) -> dict:
    """Everything of one image except the features, from what `_prepare` returned."""
    from PIL import Image

    from .data.targets import zones_to_class

    pos, shape = it["positions"], tuple(it["shape"])
    front = np.asarray(Image.open(data_root / "fronts" / sar_path.parent.name / f"{sar_path.stem}_front.png"))
    if front.shape != shape:
        raise ValueError(f"{sar_path.stem}: front {front.shape} vs image {shape}")
    return {"stem": it["stem"], "shape": shape, "patch": patch,
            "image": stitch(it["raw"], pos, shape, 1),
            "zones": zones_to_class(stitch(it["zone_tiles"], pos, shape, 1), with_na=True)[0],
            "labels": stitch(it["labels"], pos, shape, patch),
            "valid": stitch(it["valid"], pos, shape, patch),
            "front": front_patches(front, patch), "front_px": front > 0, "feats": {}, "layers": {}}


def _frozen_encoder(name: str, token: str | None):
    from .models.encoders import ENCODERS, FrozenEncoder
    from .runs import ensure_packages

    spec = ENCODERS[name]
    ensure_packages(spec.kind)
    return FrozenEncoder(spec, token=token)


def extract(df: pd.DataFrame, data_root: str | Path, stems=STEMS, encoders: dict = ENCODER_LAYERS,
            token: str | None = None, make_encoder=None, anyup=None, device: str | None = None) -> dict:
    """{stem: image entry} with `feats` {encoder: (gh, gw, D) float16} for every encoder.

    One encoder at a time: all images, then the encoder is freed before the next.
    Same preparation as the study (`_prepare`, caffe, 512 px, overlap 0, `encode(layer=)`).
    `make_encoder(name, token)` is normally left as None: the real frozen encoder is loaded.

    With `anyup` (the loaded AnyUp model), while an encoder's tiles are in memory, also
    `anyup_q` {encoder: (ceil(h / 4), ceil(w / 4), D) float16} (option A) and `crop_full`
    {encoder: (window h, window w, D) float16} for the window `crop` (option B); see `_add_anyup`.
    AnyUp runs on `device` (None: CUDA when there is a GPU).
    """
    import torch

    from .data.tiling import TileSpec
    from .features import _prepare

    data_root, make_encoder = Path(data_root), make_encoder or _frozen_encoder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = {Path(p).stem: Path(p) for p in df["path"]}
    missing = [s for s in stems if s not in paths]
    if missing or not stems:
        raise KeyError(f"not among the SAR images: {missing}" if missing else "no stems given")
    spec, data = TileSpec(size=TILE, overlap=0.0), {}
    for name, layer in encoders.items():
        t0, t_up = time.perf_counter(), 0.0
        enc = make_encoder(name, token)
        try:
            for stem in stems:
                it = _prepare(paths[stem], data_root, spec, NORM, enc.patch, with_na=True)
                tiles = enc.encode(it["x"], layer=layer)
                feats = stitch(tiles, it["positions"], it["shape"], enc.patch)
                if stem not in data:
                    data[stem] = _entry(it, paths[stem], data_root, enc.patch)
                e = data[stem]
                labels = stitch(it["labels"], it["positions"], it["shape"], enc.patch)
                if feats.shape[:2] != e["labels"].shape or not np.array_equal(labels, e["labels"]):
                    raise RuntimeError(f"{stem}: {name} patch grid {feats.shape[:2]} does not match the labels "
                                       f"{e['labels'].shape} of the first encoder")
                e["feats"][name], e["layers"][name] = feats, layer
                if anyup is not None:
                    t1 = time.perf_counter()
                    _add_anyup(e, name, tiles, it, anyup, device)
                    t_up += time.perf_counter() - t1
        finally:
            del enc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        up = "" if anyup is None else f" (of it AnyUp 1/4 + full-resolution window: {t_up / 60:.1f} min)"
        print(f"{name} layer {layer}: {len(stems)} images, {feats.shape[-1]}-d, "
              f"{(time.perf_counter() - t0) / 60:.1f} min{up}")
    return data


def anyup_tiles(anyup, tile_feats: np.ndarray, raw_tiles: np.ndarray, out_size: tuple[int, int],
                device: str = "cuda", q_chunk_size: int | None = None) -> np.ndarray:
    """AnyUp on each tile: (T, g, g, D) patch features + (T, t, t) raw uint8 tiles -> (T, *out_size, D) float16.

    As the study ran it: guide = `guide_image` of the raw tile (what `heads._to_guide` returns),
    resized bilinearly to `out_size` when that differs from the tile (as `AnyUpDecoder`, head 6);
    `upsample_with_value(anyup, guide, feats, feats, out_size)`: value = the features, which is
    `anyup(guide, feats)`; under bf16 autocast on CUDA, as every reported run (`fp32_positions`
    not used). One tile at a time; `q_chunk_size` only bounds memory (same result).
    """
    import torch
    import torch.nn.functional as F

    from .models.anyup_loader import guide_image, upsample_with_value

    size, out = (int(out_size[0]), int(out_size[1])), []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for f, raw in zip(tile_feats, raw_tiles):
            ft = torch.from_numpy(np.asarray(f, np.float32)).permute(2, 0, 1)[None].to(device)
            guide = guide_image(torch.from_numpy(np.ascontiguousarray(raw))[None, None].to(device))
            if tuple(guide.shape[-2:]) != size:
                guide = F.interpolate(guide, size=size, mode="bilinear", align_corners=False)
            up = upsample_with_value(anyup, guide, ft, ft, out_size=size, q_chunk_size=q_chunk_size)
            out.append(up[0].permute(1, 2, 0).to(torch.float16).cpu().numpy())
    return np.stack(out)


def crop_window(query: tuple[int, int], patch: int, shape: tuple[int, int], positions: np.ndarray,
                size: int = CROP, tile: int = TILE) -> dict:
    """Option B's window: `size` x `size` px centred on the centre of query patch (row, col), inside
    the tile that holds that centre and inside the image (moved inward at their edges; cut to what
    the tile holds of the image when that is less than `size`). The query patch's image pixels are
    always inside. Returns {tile (index), tile_top, tile_left, y0, y1, x0, x1 (end exclusive),
    query_row, query_col}."""
    r, c = int(query[0]), int(query[1])
    cy, cx = r * patch + patch // 2, c * patch + patch // 2
    pos = np.asarray(positions)
    hit = np.flatnonzero((pos[:, 0] <= cy) & (cy < pos[:, 0] + tile) & (pos[:, 1] <= cx) & (cx < pos[:, 1] + tile))
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} tiles hold pixel ({cy}, {cx}): needs tiles with overlap 0")
    t = int(hit[0])
    top, left = int(pos[t, 0]), int(pos[t, 1])
    box = []
    for centre, start, n in ((cy, top, shape[0]), (cx, left, shape[1])):
        lo, hi = start, min(start + tile, n)
        s = min(size, hi - lo)
        a = min(max(centre - size // 2, lo), hi - s)
        box += [a, a + s]
    return {"tile": t, "tile_top": top, "tile_left": left, "y0": box[0], "y1": box[1], "x0": box[2], "x1": box[3],
            "query_row": r, "query_col": c}


def _add_anyup(e: dict, name: str, tile_feats: np.ndarray, it: dict, anyup, device: str) -> None:
    """Options A and B of one encoder on one image, from its tile features (T, g, g, D).

    A: every tile at 1/4 (128 x 128 out of 512 px), stitched into 4 px blocks, cut to the image.
    B: the tile holding the ocean query's centre at full resolution, cut to `crop_window`. The
    window depends only on the labels, so it is the same for every encoder."""
    pos, t_px = it["positions"], it["raw"].shape[-1]
    if "crop" not in e:
        q = query_patch(e["labels"], e["valid"], e["front"], pixel_shares(e["zones"], e["patch"]), OCEAN)
        e["crop"] = crop_window((q["row"], q["col"]), e["patch"], e["shape"], pos, CROP, t_px)
        e["anyup_q"], e["crop_full"] = {}, {}
    quarter = anyup_tiles(anyup, tile_feats, it["raw"], (t_px // BLOCK, t_px // BLOCK), device)
    e["anyup_q"][name] = stitch(quarter, pos, e["shape"], BLOCK)
    del quarter
    cr = e["crop"]
    t, top, left = cr["tile"], cr["tile_top"], cr["tile_left"]
    full = anyup_tiles(anyup, tile_feats[t:t + 1], it["raw"][t:t + 1], (t_px, t_px), device)[0]
    e["crop_full"][name] = np.ascontiguousarray(full[cr["y0"] - top:cr["y1"] - top, cr["x0"] - left:cr["x1"] - left])


# ------------------------------------------------------------------ the measures

def _unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to length 1 (float32)."""
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pca3(X: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, 3): the first three principal components, exact SVD of the centred rows.

    Each component's sign is fixed so its largest loading is positive (as sklearn does),
    so a re-run gives the same colours.
    """
    import torch

    x = torch.from_numpy(np.asarray(X, np.float32))
    x = x - x.mean(0)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    v = vh[:3]
    v = v * torch.sign(v.gather(1, v.abs().argmax(1, keepdim=True)))
    return (x @ v.T).numpy()


def pca_rgb(feats: np.ndarray, valid: np.ndarray, pct: tuple[float, float] = (1, 99)) -> np.ndarray:
    """PCA 1-3 of this image's valid patches as RGB in [0, 1]; invalid patches black.

    Each component is scaled from its 1st to its 99th percentile and clipped.
    """
    p = pca3(feats[valid])
    lo, hi = np.percentile(p, pct, axis=0)
    rgb = np.zeros(valid.shape + (3,), np.float32)
    rgb[valid] = np.clip((p - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    return rgb


def kmeans_map(feats: np.ndarray, valid: np.ndarray, k: int = N_CLUSTERS, seed: int = SEED,
               max_fit: int | None = None) -> np.ndarray:
    """k-means on the valid patches (raw features, as PCA); -1 on invalid patches.

    `max_fit` (AnyUp blocks): with more valid cells than that, fitted on `max_fit` of them drawn
    with the fixed seed, then every valid cell gets its nearest centre. None = fit on all."""
    from sklearn.cluster import KMeans

    X = feats[valid]
    if max_fit is None or len(X) <= max_fit:
        km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X.astype(np.float32))
        labels = km.labels_
    else:
        pick = np.sort(np.random.default_rng(seed).choice(len(X), max_fit, replace=False))
        km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X[pick].astype(np.float32))
        labels = km.predict(X.astype(np.float32))
    out = np.full(valid.shape, -1, np.int16)
    out[valid] = labels
    return out


def query_patch(labels: np.ndarray, valid: np.ndarray, front: np.ndarray, shares: np.ndarray,
                cls: int, min_share: float = QUERY_PURITY) -> dict:
    """The fixed query rule (`QUERY_RULE`): the purest (>= 90 %) patch of class `cls` closest to
    the true front that does not itself contain the front; ties toward the middle of the front.

    Candidates: valid patches labelled `cls`, holding no front pixel (`front`, patch mask), with
    at least `min_share` of their in-image pixels in `cls` (`shares` from `pixel_shares`). Among
    them the one with the smallest Euclidean distance, in patches, to the nearest front patch.
    Ties -> the one closest to the middle of the front (median row, median column of the front
    patches), then the first in raster order (row by row, left to right). If no candidate reaches
    `min_share`, the purest ones are used instead and `fallback` is True.
    Returns {row, col, front_dist (patches), shares {class: pixel share of the query}, fallback, rule}.
    """
    from scipy.ndimage import distance_transform_edt

    name = CLASSES4[cls]
    if not front.any():
        raise ValueError("no true front in this image")
    base = valid & (labels == cls) & ~front
    if not base.any():
        raise ValueError(f"no valid patch labelled {name} off the true front")
    share = shares[..., cls]
    cand = base & (share >= min_share)
    fallback = not cand.any()
    if fallback:
        cand = base & (share == share[base].max())
    dist = distance_transform_edt(~front)
    rows, cols = np.nonzero(cand)                             # raster order
    d = dist[rows, cols]
    near = d == d.min()
    rows, cols = rows[near], cols[near]
    fr, fc = np.nonzero(front)
    mid = (rows - np.median(fr)) ** 2 + (cols - np.median(fc)) ** 2
    i = int(np.argmin(mid))                                   # argmin = first minimum in raster order
    r, c = int(rows[i]), int(cols[i])
    rule = QUERY_RULE.format(pct=100 * min_share, cls=name)
    if fallback:
        rule = (f"FALLBACK (no {name} patch off the front has >= {100 * min_share:.0f} % {name} pixels): "
                f"the purest ({100 * share[r, c]:.1f} % {name}) {name} patch closest to the true front that "
                "does not itself contain the front; ties toward the middle of the front")
    return {"row": r, "col": c, "front_dist": float(dist[r, c]), "fallback": fallback, "rule": rule,
            "shares": {n: float(shares[r, c, k]) for k, n in enumerate(CLASSES4)}}


def fair_k(n_query_class: int) -> int:
    """k20 = max(10, round(0.2 * n)), n = the image's valid patches of the query's class: the same
    20 % of that class on every image, so shares compare like-for-like between images."""
    return max(10, round(0.2 * n_query_class))


def _class_shares(labels: np.ndarray, flat_idx: np.ndarray) -> dict:
    counts = np.bincount(labels.ravel()[flat_idx], minlength=len(CLASSES4))
    if len(counts) > len(CLASSES4):
        raise ValueError("labels outside 0..3 among the valid patches")
    return {n: float(v / len(flat_idx)) for n, v in zip(CLASSES4, counts)}


def similarity_search(feats: np.ndarray, valid: np.ndarray, labels: np.ndarray,
                      query: tuple[int, int], k: int = TOP_K, k20: int | None = None) -> dict:
    """Cosine similarity of the query patch to every valid patch, and the class shares of
    the k most similar ones. The query itself is left out; ties -> raster order.
    With `k20` (see `fair_k`) also the shares of the k20 most similar (`k20`, `shares20`)."""
    gh, gw = valid.shape
    qi = query[0] * gw + query[1]
    if not valid.flat[qi]:
        raise ValueError(f"query patch {query} is not valid")
    idx = np.flatnonzero(valid)
    X = _unit(feats.reshape(gh * gw, -1)[idx])
    sim = X @ X[np.searchsorted(idx, qi)]
    sim_map = np.full(gh * gw, np.nan, np.float32)
    sim_map[idx] = sim
    order = idx[np.argsort(-sim, kind="stable")]
    ranked = order[order != qi]
    top = ranked[:k]
    out = {"sim": sim_map.reshape(gh, gw), "top": np.column_stack(np.divmod(top, gw)), "k": len(top),
           "shares": _class_shares(labels, top)}
    if k20 is not None:
        out.update(k20=len(ranked[:k20]), shares20=_class_shares(labels, ranked[:k20]))
    return out


def quarter_view(e: dict) -> dict:
    """Option A's features (`anyup_q`) as an image entry on 4 px blocks, for the patch-map views.

    Block zone = the majority of its in-image pixels (`pixel_shares`; ties -> the lower class id,
    as the patch labels); valid = every block (the grid is cut at the image edge, so each block
    holds image pixels); front block = holds a true-front pixel. `patch` is the block size."""
    if "anyup_q" not in e:
        raise KeyError(f"{e['stem']}: no AnyUp features, run extract(..., anyup=...)")
    shares = pixel_shares(e["zones"], BLOCK)
    return {"stem": e["stem"], "shape": e["shape"], "patch": BLOCK, "image": e["image"], "zones": e["zones"],
            "labels": shares.argmax(-1).astype(np.uint8), "valid": np.ones(shares.shape[:2], bool),
            "front": front_patches(e["front_px"], BLOCK), "feats": e["anyup_q"], "layers": e["layers"]}


def patch_footprint(query: tuple[int, int], patch: int, unit: int, grid: tuple[int, int],
                    origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Mask on a grid of `unit` px cells whose first cell starts at pixel `origin`: the cells inside
    patch (row, col). Blocks: unit 4 -> the patch's 4 x 4 blocks; pixels of a window: unit 1."""
    r, c = query
    ys = origin[0] + np.arange(grid[0]) * unit
    xs = origin[1] + np.arange(grid[1]) * unit
    return (((ys >= r * patch) & (ys < (r + 1) * patch))[:, None]
            & ((xs >= c * patch) & (xs < (c + 1) * patch))[None, :])


def similarity_search_area(feats: np.ndarray, valid: np.ndarray, labels: np.ndarray, footprint: np.ndarray,
                           k: int, k20: int | None = None) -> dict:
    """`similarity_search` from a query AREA (`footprint`, a mask): query vector = the mean of the
    L2-normalised features of its valid cells, normalised; its cells are left out of the ranking.
    One cell -> the same as `similarity_search`. Returns the same keys."""
    gh, gw = valid.shape
    fp = footprint & valid
    if not fp.any():
        raise ValueError("the query area has no valid cell")
    idx = np.flatnonzero(valid)
    X = _unit(feats.reshape(gh * gw, -1)[idx])
    sim = X @ _unit(X[fp.ravel()[idx]].mean(0))
    sim_map = np.full(gh * gw, np.nan, np.float32)
    sim_map[idx] = sim
    order = idx[np.argsort(-sim, kind="stable")]
    ranked = order[~footprint.ravel()[order]]
    top = ranked[:k]
    out = {"sim": sim_map.reshape(gh, gw), "top": np.column_stack(np.divmod(top, gw)), "k": len(top),
           "shares": _class_shares(labels, top)}
    if k20 is not None:
        out.update(k20=len(ranked[:k20]), shares20=_class_shares(labels, ranked[:k20]))
    return out


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random `pos` value > a random `neg` value), ties counting half, from average
    ranks (Mann-Whitney U). 1.0 = fully separated, 0.5 = no separation. No model, no threshold."""
    from scipy.stats import rankdata

    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def class_margins(feats: np.ndarray, valid: np.ndarray, labels: np.ndarray) -> dict:
    """margin = cos(patch, ocean centroid) - cos(patch, glacier centroid) for every valid
    glacier and ocean patch. Centroid = mean of the class's L2-normalised features, from
    this image's own labels. `auc` = how well the margin tells ocean (higher) from glacier."""
    X, lab = _unit(feats[valid]), labels[valid]
    g, o = lab == GLACIER, lab == OCEAN
    if not g.any() or not o.any():
        raise ValueError("the image needs glacier and ocean patches")
    cg, co = _unit(X[g].mean(0)), _unit(X[o].mean(0))
    margin = X @ co - X @ cg
    return {"glacier": margin[g], "ocean": margin[o], "auc": auc_rank(margin[o], margin[g])}


def patch_counts(e: dict) -> dict:
    n = np.bincount(e["labels"][e["valid"]], minlength=len(CLASSES4))
    return {"n_patches": int(e["valid"].sum()), **{f"n_{c}": int(v) for c, v in zip(CLASSES4, n)}}


def crop_views(e: dict, name: str) -> dict:
    """Option B for encoder `name` in the window `e["crop"]`, pixel by pixel, for each of
    `ZOOM_VERSIONS`: raw (each pixel takes its patch's vector), anyup_quarter (its block's),
    anyup_full (its own). Per version {pca: PCA-RGB fitted on the window's pixels, sim: cosine to
    the ocean query (mean of the L2-normalised vectors of the query patch's pixels, normalised),
    auc: `class_margins` AUC over the window's pixels with the pixel zones as labels (NaN if the
    window lacks glacier or ocean)}."""
    cr, p = e["crop"], e["patch"]
    y0, y1, x0, x1 = cr["y0"], cr["y1"], cr["x0"], cr["x1"]
    ys, xs = np.arange(y0, y1)[:, None], np.arange(x0, x1)[None, :]
    zones = e["zones"][y0:y1, x0:x1]
    valid = np.ones(zones.shape, bool)
    foot = patch_footprint((cr["query_row"], cr["query_col"]), p, 1, zones.shape, (y0, x0))
    both = bool((zones == GLACIER).any() and (zones == OCEAN).any())
    out = {}
    for key in ZOOM_VERSIONS:
        if key == "raw":
            f = e["feats"][name][ys // p, xs // p]
        elif key == "anyup_quarter":
            f = e["anyup_q"][name][ys // BLOCK, xs // BLOCK]
        else:
            f = e["crop_full"][name]
        if f.shape[:2] != zones.shape:
            raise ValueError(f"{name} {key}: window features {f.shape[:2]} vs window {zones.shape}")
        X = _unit(f.reshape(-1, f.shape[-1]))
        sim = (X @ _unit(X[foot.ravel()].mean(0))).reshape(zones.shape)
        del X
        out[key] = {"pca": pca_rgb(f, valid), "sim": sim,
                    "auc": class_margins(f, valid, zones)["auc"] if both else float("nan")}
    return out


# ------------------------------------------------------------------ figures

def _figure(e: dict, n_rows: int, n_cols: int, panel: float = 3.4):
    import matplotlib.pyplot as plt

    h, w = e["shape"]
    ratio = min(max(h / w, 0.4), 2.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * panel, n_rows * panel * ratio + 1.2),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def _fit_to_image(ax, e: dict) -> None:
    """Axis limits = the image (pixel centres at whole numbers, as `imshow` draws it). Called
    after boxes and circles are drawn, so nothing reaching past the edge adds white strips."""
    h, w = e["shape"]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def _patch_map(ax, arr, e: dict, **kw):
    """A patch-grid map drawn in the image's pixel coordinates (patch edges on pixel edges)."""
    gh, gw = arr.shape[:2]
    p = e["patch"]
    im = ax.imshow(arr, extent=(-0.5, gw * p - 0.5, gh * p - 0.5, -0.5), interpolation="nearest", **kw)
    _fit_to_image(ax, e)
    return im


def _image_and_zones(ax_img, ax_zones, e: dict) -> None:
    from matplotlib.colors import ListedColormap

    ax_img.imshow(e["image"], cmap="gray", vmin=0, vmax=255)
    ax_zones.imshow(e["zones"], cmap=ListedColormap(ZONE_COLOURS), vmin=-0.5, vmax=3.5,
                    interpolation="nearest")
    for ax in (ax_img, ax_zones):
        _fit_to_image(ax, e)


def _zone_handles() -> list:
    from matplotlib.patches import Patch

    return [Patch(facecolor=c, edgecolor="#333333", label=n) for c, n in zip(ZONE_COLOURS, ZONE_LABELS)]


def _boxes(ax, cells: np.ndarray, patch: int) -> None:
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    ax.add_collection(PatchCollection([Rectangle((c * patch - 0.5, r * patch - 0.5), patch, patch)
                                       for r, c in cells],
                                      facecolor="none", edgecolor=MATCH_COLOUR, linewidth=0.8))


def _query_mark(ax, rc: tuple[int, int], e: dict) -> None:
    """Query box plus a circle around it so it can be found; dark halo under yellow."""
    from matplotlib.patches import Circle, Rectangle

    (r, c), p = rc, e["patch"]
    for colour, lw in (("black", 3.5), (QUERY_COLOUR, 2.0)):
        ax.add_patch(Rectangle((c * p - 0.5, r * p - 0.5), p, p, fill=False, edgecolor=colour, linewidth=lw))
        ax.add_patch(Circle(((c + 0.5) * p - 0.5, (r + 0.5) * p - 0.5), 0.05 * max(e["shape"]), fill=False,
                            edgecolor=colour, linewidth=lw))


def _save(fig, out_dir: str | Path, name: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=150)
    print(f"saved {out / name}")
    return fig


def _label(e: dict, name: str) -> str:
    return f"{name}\nlayer {e['layers'][name]}"


def _features_view(e: dict, features: str) -> tuple[dict, dict]:
    """(entry to draw, wording) for `features`: "raw" (16 px patches) or "anyup_quarter" (option A's 4 px blocks)."""
    if features == "raw":
        return e, {"max_fit": None, "tag": "", "what": "", "km_note": "", "unit": ("patch", "patches"), "name": ""}
    if features == "anyup_quarter":
        return quarter_view(e), {"max_fit": KMEANS_MAX_FIT, "tag": "_anyup", "what": " + AnyUp to 1/4 (4 px blocks)",
                                 "km_note": f"\n(fitted on <= {KMEANS_MAX_FIT:,} blocks)", "unit": ("block", "blocks"),
                                 "name": ", AnyUp 1/4"}
    raise ValueError(f"features {features!r}: 'raw' or 'anyup_quarter'")


def plot_embedding_maps(e: dict, out_dir: str | Path, features: str = "raw"):
    """One figure per image, one row per encoder: image | true zones | PCA 1-3 as RGB | k-means.

    `features="anyup_quarter"`: the same on option A's 4 px blocks (`quarter_view`), k-means fitted
    on at most `KMEANS_MAX_FIT` blocks (fixed seed); saved as embedding_maps_anyup_<stem>.png."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    v, how = _features_view(e, features)
    names = list(v["feats"])
    fig, axes = _figure(v, len(names), 4)
    for row, name in zip(axes, names):
        f = v["feats"][name]
        _image_and_zones(row[0], row[1], v)
        _patch_map(row[2], pca_rgb(f, v["valid"]), v)
        _patch_map(row[3], np.ma.masked_less(kmeans_map(f, v["valid"], max_fit=how["max_fit"]), 0), v,
                   cmap=ListedColormap(CLUSTER_COLOURS[:N_CLUSTERS]), vmin=-0.5, vmax=N_CLUSTERS - 0.5)
        row[0].set_ylabel(_label(v, name))
        for ax in row:
            _fit_to_image(ax, v)
    for ax, title in zip(axes[0], ("SAR image", "true zones", "PCA 1-3 as RGB",
                                   f"k-means, k = {N_CLUSTERS}{how['km_note']}")):
        ax.set_title(title)
    handles = _zone_handles() + [Patch(facecolor=c, label=f"cluster {i}")
                                 for i, c in enumerate(CLUSTER_COLOURS[:N_CLUSTERS])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"{e['stem']}: frozen features{how['what']}, no training "
                 "(cluster numbers and PCA colours are arbitrary)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, out_dir, f"embedding_maps{how['tag']}_{e['stem']}.png")


def plot_similarity(e: dict, query: str, out_dir: str | Path, k: int = TOP_K):
    """Similarity search from the fixed query patch (`query_patch`) of class `query`, one
    row per encoder: image | true zones (query and top-k matches boxed on both) | cosine
    similarity map. Shares of the top k and of the top k20 (`fair_k`). Returns (figure, one
    record per encoder) and prints the query (place, distance, pixel shares, rule) and the shares."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    cls, p = CLASSES4.index(query), e["patch"]
    q = query_patch(e["labels"], e["valid"], e["front"], pixel_shares(e["zones"], p), cls)
    r, c = q["row"], q["col"]
    n_q = int((e["valid"] & (e["labels"] == cls)).sum())
    k20 = fair_k(n_q)
    px = ", ".join(f"{n} {q['shares'][n]:.2f}" for n in ("glacier", "ocean", "stone", "na"))
    print(f"{e['stem']}, {query} query: patch (row {r}, col {c}) = pixels (y {r * p}, x {c * p}), "
          f"{q['front_dist']:.1f} patches from the true front | its pixels: {px}\n  rule: {q['rule']}\n"
          f"  {n_q:,} valid {query} patches in the image -> k20 = {k20}")
    cmap = LinearSegmentedColormap.from_list("similarity", SIM_RAMP)
    cmap.set_bad("white")
    names = list(e["feats"])
    fig, axes = _figure(e, len(names), 3)
    recs = []
    for row, name in zip(axes, names):
        s = similarity_search(e["feats"][name], e["valid"], e["labels"], (r, c), k, k20)
        _image_and_zones(row[0], row[1], e)
        im = _patch_map(row[2], s["sim"], e, cmap=cmap)
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.02, label="cosine similarity")
        for ax in row[:2]:
            _boxes(ax, s["top"], p)
        for ax in row:
            _query_mark(ax, (r, c), e)
            _fit_to_image(ax, e)                      # last: boxes and circles past the edge add no strips
        text, text20 = (", ".join(f"{n} {sh[n]:.2f}" for n in ("glacier", "ocean", "stone", "na"))
                        for sh in (s["shares"], s["shares20"]))
        row[1].set_xlabel(f"top {s['k']}: {text}\ntop {s['k20']} (k20): {text20}", fontsize=8)
        row[0].set_ylabel(_label(e, name))
        print(f"  {name} layer {e['layers'][name]}: top {s['k']} = {text} | top {s['k20']} (k20) = {text20}")
        recs.append({"stem": e["stem"], "encoder": name, "layer": e["layers"][name], "query": query,
                     "query_row": r, "query_col": c, "query_front_dist_patches": q["front_dist"],
                     **{f"query_px_share_{n}": q["shares"][n] for n in CLASSES4},
                     "query_fallback": q["fallback"], "query_rule": q["rule"], "n_query_class": n_q,
                     "top_k": s["k"], **{f"share_{n}": s["shares"][n] for n in CLASSES4},
                     "k20": s["k20"], **{f"k20_share_{n}": s["shares20"][n] for n in CLASSES4}})
    for ax, title in zip(axes[0], ("SAR image", "true zones", f"similarity to the {query} query")):
        ax.set_title(title)
    purity = f"{query}, below 90 %" if q["fallback"] else f"pure {query}"      # never call a fallback 'pure'
    marks = [Line2D([], [], color=QUERY_COLOUR, lw=2, label=f"query ({purity} next to the front)"),
             Line2D([], [], color=MATCH_COLOUR, lw=1, label=f"top {k} most similar patches")]
    fig.legend(handles=marks + _zone_handles(), loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"{e['stem']}: which patches look like the {query} patch next to the front?")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=2.5)      # room for the two lines of shares
    return _save(fig, out_dir, f"similarity_{query}_query_{e['stem']}.png"), recs


def plot_margins(e: dict, out_dir: str | Path, features: str = "raw"):
    """Histograms of the ocean-minus-glacier centroid margin, glacier vs ocean patches, one
    panel per encoder, with the AUC. Returns (figure, one record per encoder) and prints the AUC.

    `features="anyup_quarter"`: the same over option A's 4 px blocks (`quarter_view`; the counts in
    the records are then blocks); saved as class_margins_anyup_<stem>.png."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    v, how = _features_view(e, features)
    one, many = how["unit"]
    names = list(v["feats"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8), squeeze=False)
    counts, recs = patch_counts(v), []
    colours = {key: ZONE_COLOURS[CLASSES4.index(key)] for key in ("glacier", "ocean")}
    for ax, name in zip(axes[0], names):
        m = class_margins(v["feats"][name], v["valid"], v["labels"])
        both = np.concatenate([m["glacier"], m["ocean"]])
        bins = np.linspace(both.min(), both.max(), 61) if both.max() > both.min() else 10
        for key, colour in colours.items():           # both fills first, see-through ...
            ax.hist(m[key], bins=bins, density=True, histtype="stepfilled", color=colour, alpha=HIST_ALPHA,
                    linewidth=0)
        for key in colours:                           # ... then both outlines on top: both stay visible
            ax.hist(m[key], bins=bins, density=True, histtype="step", color=HIST_LINES[key], linewidth=1.5)
        ax.axvline(0, color="#52514e", linewidth=1, linestyle="--")
        ax.set_title(f"{name}, layer {e['layers'][name]}\nAUC ocean vs glacier = {m['auc']:.3f}", fontsize=9)
        ax.set_xlabel(f"cos({one}, ocean centroid) - cos({one}, glacier centroid)", fontsize=8)
        ax.legend(handles=[Patch(facecolor=to_rgba(colour, HIST_ALPHA), edgecolor=HIST_LINES[key], linewidth=1.5,
                                 label=f"{key} {many} (n = {len(m[key]):,})") for key, colour in colours.items()],
                  fontsize=8, frameon=False)
        print(f"{e['stem']}{how['name']}  {name} layer {e['layers'][name]}: AUC ocean vs glacier {m['auc']:.3f} "
              f"({len(m['ocean']):,} ocean, {len(m['glacier']):,} glacier {many})")
        recs.append({"stem": e["stem"], "encoder": name, "layer": e["layers"][name],
                     "auc_ocean_vs_glacier": m["auc"], **counts})
    axes[0][0].set_ylabel("density")
    fig.suptitle(f"{e['stem']}{how['name']}: closer to the ocean or the glacier centroid? (centroids from this image's "
                 "labels)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, f"class_margins{how['tag']}_{e['stem']}.png"), recs


def _outline(ax, cells: np.ndarray, grid: tuple[int, int], unit: int) -> None:
    """The outline of a set of grid cells (the top AnyUp blocks) as one contour: 1,600 boxes of
    4 px would hide the zones under them."""
    gh, gw = grid
    m = np.zeros((gh + 2, gw + 2), np.float32)             # a free border closes the outline at the edge
    m[cells[:, 0] + 1, cells[:, 1] + 1] = 1
    ax.contour((np.arange(-1, gw + 1) + 0.5) * unit - 0.5, (np.arange(-1, gh + 1) + 0.5) * unit - 0.5, m,
               levels=[0.5], colors=[MATCH_COLOUR], linewidths=0.8)


def plot_similarity_anyup(e: dict, query: str, out_dir: str | Path, k: int = TOP_K):
    """`plot_similarity` on option A's features (AnyUp 1/4, 4 px blocks) from the SAME query patch.

    Query vector = the mean of the L2-normalised features of the patch's 16 blocks, normalised
    (`patch_footprint`, `similarity_search_area`); those blocks are left out of the ranking. The
    lists cover the same area as the raw ones: 16 x k blocks and 16 x k20 blocks, k20 from the raw
    patch count of the query's class. Shares count blocks (block zone: `quarter_view`). The top
    blocks are outlined. Returns (figure, one record per encoder with the keys of `plot_similarity`)."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    v, cls, p = quarter_view(e), CLASSES4.index(query), e["patch"]
    q = query_patch(e["labels"], e["valid"], e["front"], pixel_shares(e["zones"], p), cls)
    r, c = q["row"], q["col"]
    n_q = int((e["valid"] & (e["labels"] == cls)).sum())
    per = (p // BLOCK) ** 2
    kb, k20b = per * k, per * fair_k(n_q)
    foot = patch_footprint((r, c), p, BLOCK, v["valid"].shape)
    print(f"{e['stem']}, {query} query, AnyUp 1/4: the same patch (row {r}, col {c}) = its {int(foot.sum())} blocks | "
          f"top {kb:,} blocks = the area of {k} patches | {per} x k20 = {k20b:,} blocks "
          f"({n_q:,} valid {query} patches -> k20 = {fair_k(n_q)})")
    cmap = LinearSegmentedColormap.from_list("similarity", SIM_RAMP)
    cmap.set_bad("white")
    names = list(v["feats"])
    fig, axes = _figure(v, len(names), 3)
    recs = []
    for row, name in zip(axes, names):
        s = similarity_search_area(v["feats"][name], v["valid"], v["labels"], foot, kb, k20b)
        _image_and_zones(row[0], row[1], v)
        im = _patch_map(row[2], s["sim"], v, cmap=cmap)
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.02, label="cosine similarity")
        for ax in row[:2]:
            _outline(ax, s["top"], v["valid"].shape, BLOCK)
        for ax in row:
            _query_mark(ax, (r, c), e)
            _fit_to_image(ax, e)
        text, text20 = (", ".join(f"{n} {sh[n]:.2f}" for n in ("glacier", "ocean", "stone", "na"))
                        for sh in (s["shares"], s["shares20"]))
        row[1].set_xlabel(f"top {s['k']:,} blocks: {text}\ntop {s['k20']:,} blocks ({per} x k20): {text20}", fontsize=8)
        row[0].set_ylabel(_label(e, name))
        print(f"  {name} layer {e['layers'][name]}: top {s['k']:,} blocks = {text} | "
              f"top {s['k20']:,} blocks ({per} x k20) = {text20}")
        recs.append({"stem": e["stem"], "encoder": name, "layer": e["layers"][name], "query": query,
                     "query_row": r, "query_col": c, "query_front_dist_patches": q["front_dist"],
                     **{f"query_px_share_{n}": q["shares"][n] for n in CLASSES4},
                     "query_fallback": q["fallback"], "query_rule": q["rule"], "n_query_class": n_q,
                     "top_k": s["k"], **{f"share_{n}": s["shares"][n] for n in CLASSES4},
                     "k20": s["k20"], **{f"k20_share_{n}": s["shares20"][n] for n in CLASSES4}})
    for ax, title in zip(axes[0], ("SAR image", "true zones", f"similarity to the {query} query (AnyUp 1/4)")):
        ax.set_title(title)
    purity = f"{query}, below 90 %" if q["fallback"] else f"pure {query}"
    marks = [Line2D([], [], color=QUERY_COLOUR, lw=2, label=f"query ({purity} next to the front, its 16 blocks)"),
             Line2D([], [], color=MATCH_COLOUR, lw=1, label=f"top {kb:,} most similar blocks (area of {k} patches)")]
    fig.legend(handles=marks + _zone_handles(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{e['stem']}: AnyUp 1/4 features: which 4 px blocks look like the {query} patch next to the front?")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=2.5)
    return _save(fig, out_dir, f"similarity_{query}_query_anyup_{e['stem']}.png"), recs


def plot_zoom(e: dict, out_dir: str | Path):
    """Option B: the window `e["crop"]` around the ocean query, one row per encoder (`crop_views`).

    zoom_front_<stem>.png: SAR | true zones with the true front (red) | PCA 1-3 of raw patches,
    AnyUp 1/4 and AnyUp full (each fitted on the window). zoom_front_similarity_<stem>.png: SAR |
    true zones | similarity to the ocean query, same three versions, one colour scale per row.
    The window AUC of each version is written under its panels. Returns (figure, figure, one
    record per encoder) and prints the window and the AUCs."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    cr, p = e["crop"], e["patch"]
    y0, y1, x0, x1 = cr["y0"], cr["y1"], cr["x0"], cr["x1"]
    q = query_patch(e["labels"], e["valid"], e["front"], pixel_shares(e["zones"], p), OCEAN)
    if (q["row"], q["col"]) != (cr["query_row"], cr["query_col"]):
        raise RuntimeError(f"{e['stem']}: window at patch {cr['query_row'], cr['query_col']}, ocean query at "
                           f"{q['row'], q['col']}")
    win = {"shape": (y1 - y0, x1 - x0), "image": e["image"][y0:y1, x0:x1], "zones": e["zones"][y0:y1, x0:x1]}
    front = e["front_px"][y0:y1, x0:x1]
    n_g, n_o = int((win["zones"] == GLACIER).sum()), int((win["zones"] == OCEAN).sum())
    print(f"{e['stem']}: window y {y0}-{y1 - 1}, x {x0}-{x1 - 1} ({y1 - y0} x {x1 - x0} px) around the ocean query "
          f"(row {q['row']}, col {q['col']}), in the tile at (y {cr['tile_top']}, x {cr['tile_left']}) | "
          f"{n_g:,} glacier, {n_o:,} ocean pixels")
    names, panel = list(e["feats"]), 2.6
    h, w = win["shape"]
    ratio = min(max(h / w, 0.4), 2.5)
    fig_p, ax_p = _figure(win, len(names), 5, panel)
    fig_s, ax_s = plt.subplots(len(names), 6, figsize=(5 * panel + 0.6, len(names) * panel * ratio + 1.2),
                               squeeze=False, gridspec_kw={"width_ratios": [1] * 5 + [0.06]})
    for ax in ax_s[:, :5].ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    cmap = LinearSegmentedColormap.from_list("similarity", SIM_RAMP)
    recs = []
    for i, name in enumerate(names):
        cv = crop_views(e, name)
        lo = min(float(np.min(cv[key]["sim"])) for key in ZOOM_VERSIONS)
        hi = max(float(np.max(cv[key]["sim"])) for key in ZOOM_VERSIONS)
        for axes in (ax_p[i], ax_s[i]):
            _image_and_zones(axes[0], axes[1], win)
            axes[1].imshow(np.ma.masked_where(~front, front.astype(np.uint8)), cmap=ListedColormap([FRONT_COLOUR]),
                           vmin=0, vmax=1, interpolation="nearest")
            axes[0].set_ylabel(_label(e, name))
        for j, key in enumerate(ZOOM_VERSIONS):
            ax_p[i, 2 + j].imshow(cv[key]["pca"], interpolation="nearest")
            im = ax_s[i, 2 + j].imshow(cv[key]["sim"], cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            for ax in (ax_p[i, 2 + j], ax_s[i, 2 + j]):
                ax.set_xlabel(f"AUC in the window {cv[key]['auc']:.3f}", fontsize=8)
        fig_s.colorbar(im, cax=ax_s[i, 5], label="cosine similarity")
        for ax in (*ax_p[i], *ax_s[i, :5]):
            for colour, lw in (("black", 3.5), (QUERY_COLOUR, 2.0)):
                ax.add_patch(Rectangle((q["col"] * p - x0 - 0.5, q["row"] * p - y0 - 0.5), p, p, fill=False,
                                       edgecolor=colour, linewidth=lw))
            _fit_to_image(ax, win)
        aucs = {key: cv[key]["auc"] for key in ZOOM_VERSIONS}
        print(f"  {name} layer {e['layers'][name]}: AUC ocean vs glacier in the window: "
              + " | ".join(f"{ZOOM_VERSIONS[key]} {a:.3f}" for key, a in aucs.items()))
        recs.append({"stem": e["stem"], "encoder": name, "layer": e["layers"][name], "query": "ocean",
                     "query_row": q["row"], "query_col": q["col"], "query_front_dist_patches": q["front_dist"],
                     **{f"query_px_share_{n}": q["shares"][n] for n in CLASSES4},
                     "query_fallback": q["fallback"], "query_rule": q["rule"],
                     "crop_y0": y0, "crop_x0": x0, "crop_height": y1 - y0, "crop_width": x1 - x0,
                     "crop_n_glacier_px": n_g, "crop_n_ocean_px": n_o,
                     **{f"crop_auc_{key}": a for key, a in aucs.items()}})
    lead = ("SAR window", "true zones + true front")
    for ax, title in zip(ax_p[0], lead + tuple(f"{t}: PCA 1-3" for t in ZOOM_VERSIONS.values())):
        ax.set_title(title, fontsize=10)
    for ax, title in zip(ax_s[0], lead + tuple(f"{t}: similarity" for t in ZOOM_VERSIONS.values())):
        ax.set_title(title, fontsize=10)
    marks = [Line2D([], [], color=FRONT_COLOUR, lw=2, label="true front"),
             Line2D([], [], color=QUERY_COLOUR, lw=2, label="ocean query (step 3)")]
    where = f"{h} x {w} px window at y {y0}, x {x0}"
    for fig, what in ((fig_p, "PCA fitted per version on the window, colours arbitrary"),
                      (fig_s, "cosine similarity to the ocean query")):
        fig.legend(handles=marks + _zone_handles(), loc="lower center", ncol=6, frameon=False, fontsize=8)
        fig.suptitle(f"{e['stem']}: {where}: raw vs AnyUp 1/4 vs AnyUp full ({what})", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=2.0)     # room for the AUC line under each row
    return (_save(fig_p, out_dir, f"zoom_front_{e['stem']}.png"),
            _save(fig_s, out_dir, f"zoom_front_similarity_{e['stem']}.png"), recs)


def _rgb8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(rgb, np.float32) * 255).astype(np.uint8)


def save_gif_arrays(e: dict, out_dir: str | Path) -> Path:
    """Write `embeddings_gif_<stem>.npz` (compressed): the maps the figures draw, compact (uint8 /
    float16, no features), for a GIF made later on the PC. Needs `extract(..., anyup=...)`.

    Keys (read by a PC-side script):
      stem                                  str
      image_shape                           int32 (2,)  image height, width (px)
      patch, block                          int32       16 (raw patch) and 4 (AnyUp 1/4 block), px
      encoders                              str (E,)    encoder names, in the figures' row order
      layers                                int32 (E,)  their layers
      query_ocean_rc, query_glacier_rc      int32 (2,)  query patch row, col (step 3 rule)
      query_ocean_box, query_glacier_box    int32 (4,)  its pixels y0, x0, y1, x1 (end exclusive, cut at the image edge)
      crop_box                              int32 (4,)  option B window y0, x0, height, width (px)
    per encoder, `<enc>/<map>` with <enc> from `encoders`:
      pca_raw                               uint8 (gh, gw, 3)  PCA 1-3 as RGB on 16 px patches (black = invalid)
      pca_quarter                           uint8 (bh, bw, 3)  the same on AnyUp 1/4 blocks
      pca_crop_raw, pca_crop_quarter, pca_crop_full   uint8 (ch, cw, 3)  option B window, per pixel
      kmeans_raw                            uint8 (gh, gw)     cluster 0-3, 255 = invalid
      kmeans_quarter                        uint8 (bh, bw)
      sim_ocean_raw, sim_glacier_raw        float16 (gh, gw)   cosine similarity to the query, NaN = invalid
      sim_ocean_quarter, sim_glacier_quarter   float16 (bh, bw)
      sim_crop_raw, sim_crop_quarter, sim_crop_full   float16 (ch, cw)  to the ocean query only (the
                                            window is built around it; the glacier query may lie outside)
    gh, gw = ceil(h / 16), ceil(w / 16); bh, bw = ceil(h / 4), ceil(w / 4) (the last row and column may
    reach past the image, as in the figures); ch, cw = crop_box height, width. PCA-RGB = round(255 x the
    figure's colours); same functions and seeds as the figures, so the same maps.
    """
    p, v, cr = e["patch"], quarter_view(e), e["crop"]
    h, w = e["shape"]
    shares = pixel_shares(e["zones"], p)
    qs = {n: query_patch(e["labels"], e["valid"], e["front"], shares, CLASSES4.index(n)) for n in ("ocean", "glacier")}
    names = list(e["feats"])
    arr = {"stem": np.array(e["stem"]), "image_shape": np.array([h, w], np.int32), "patch": np.array(p, np.int32),
           "block": np.array(BLOCK, np.int32), "encoders": np.array(names),
           "layers": np.array([e["layers"][n] for n in names], np.int32),
           "crop_box": np.array([cr["y0"], cr["x0"], cr["y1"] - cr["y0"], cr["x1"] - cr["x0"]], np.int32)}
    for qn, q in qs.items():
        r, c = q["row"], q["col"]
        arr[f"query_{qn}_rc"] = np.array([r, c], np.int32)
        arr[f"query_{qn}_box"] = np.array([r * p, c * p, min((r + 1) * p, h), min((c + 1) * p, w)], np.int32)
    for name in names:
        f, fq = e["feats"][name], v["feats"][name]
        arr[f"{name}/pca_raw"] = _rgb8(pca_rgb(f, e["valid"]))
        arr[f"{name}/pca_quarter"] = _rgb8(pca_rgb(fq, v["valid"]))
        for key, km in (("raw", kmeans_map(f, e["valid"])),
                        ("quarter", kmeans_map(fq, v["valid"], max_fit=KMEANS_MAX_FIT))):
            arr[f"{name}/kmeans_{key}"] = np.where(km < 0, 255, km).astype(np.uint8)
        for qn, q in qs.items():
            rc = (q["row"], q["col"])
            arr[f"{name}/sim_{qn}_raw"] = similarity_search(f, e["valid"], e["labels"], rc)["sim"].astype(np.float16)
            foot = patch_footprint(rc, p, BLOCK, v["valid"].shape)
            arr[f"{name}/sim_{qn}_quarter"] = similarity_search_area(fq, v["valid"], v["labels"], foot, TOP_K)[
                "sim"].astype(np.float16)
        cv = crop_views(e, name)
        for key, short in (("raw", "raw"), ("anyup_quarter", "quarter"), ("anyup_full", "full")):
            arr[f"{name}/pca_crop_{short}"] = _rgb8(cv[key]["pca"])
            arr[f"{name}/sim_crop_{short}"] = cv[key]["sim"].astype(np.float16)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"embeddings_gif_{e['stem']}.npz"
    np.savez_compressed(path, **arr)
    print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB, {len(arr)} arrays)")
    return path


def numbers_table(sim_records: list[dict], margin_records: list[dict], anyup_sim_records=(),
                  anyup_margin_records=(), crop_records=()) -> pd.DataFrame:
    """One row per image, encoder, query and feature version (column `features`, see `FEATURES`).

    raw: top-k class shares, AUC and patch counts. anyup_quarter: the same over option A's 4 px
    blocks (top_k, k20 and the n_* counts in blocks; n_query_class stays in patches).
    anyup_full_crop: one row per image and encoder, option B's window and its three AUCs (crop_*
    columns, empty in the other rows; the whole-image columns are empty in these rows)."""
    keys, parts = ["stem", "encoder", "layer"], []
    for feat, sims, margins in (("raw", sim_records, margin_records),
                                ("anyup_quarter", anyup_sim_records, anyup_margin_records)):
        if len(sims):
            t = pd.DataFrame(sims).merge(pd.DataFrame(margins), on=keys, how="left")
            t.insert(3, "features", feat)
            parts.append(t)
    if len(crop_records):
        t = pd.DataFrame(crop_records)
        t.insert(3, "features", FEATURES[2])
        parts.append(t)
    ints = {c for t in parts for c in t.columns if pd.api.types.is_integer_dtype(t[c])}
    tab = pd.concat(parts, ignore_index=True)
    for c in ints:
        if tab[c].isna().any():
            tab[c] = tab[c].astype("Int64")                 # integers stay integers next to empty cells
    return tab
