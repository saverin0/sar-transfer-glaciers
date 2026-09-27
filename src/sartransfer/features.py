"""Extract frozen features for the whole dataset to local disk.

Measured 2026-09-22: 146 tiles/s on an A100, so all 17,076 tiles take about
2 minutes per (encoder, normalisation). That is why nothing is cached to Drive —
features go to `/content`, the probe trains, and only probe weights and metric
CSVs are kept (features for every encoder and setting would need ~286 GB; recomputing is minutes).

One `.npz` per image holds features, patch targets and the patch validity mask,
so a probe can stream over files without loading 36 GB at once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data.inventory import parse_name
from .data.targets import tile_patch_targets
from .data.tiling import TileSpec, normalise, normalise_image, tile_image, to_rgb

Image.MAX_IMAGE_PIXELS = None

TEST_GLACIERS = {"COL", "Mapple"}
VAL_GLACIERS = {"SI"}


def split_of(glacier: str) -> str:
    if glacier in TEST_GLACIERS:
        return "test"
    return "val" if glacier in VAL_GLACIERS else "train"


def zone_path_for(sar_path: str | Path, data_root: Path) -> Path:
    """The zones file matching a sar_images file.

    Layout confirmed by the inventory: `zones/<split>/<stem>_zones.png` next to
    `sar_images/<split>/<stem>.png`.
    """
    sar_path = Path(sar_path)
    split = sar_path.parent.name
    return data_root / "zones" / split / f"{sar_path.stem}_zones.png"


def _prepare(sar_path: Path, data_root: Path, spec: TileSpec, norm: str, patch: int,
             min_purity: float = 0.0, with_na: bool = False) -> dict:
    """Everything one image needs before the encoder: normalised RGB tiles, patch
    labels and validity, tile positions, metadata. CPU only, so it can run in a
    background thread while the GPU encodes the previous image. Shared by the
    disk and in-memory extraction paths, so both see identical inputs."""
    sar_path = Path(sar_path)
    zone_path = zone_path_for(sar_path, data_root)
    if not zone_path.is_file():
        raise FileNotFoundError(f"no zones file for {sar_path.name}: {zone_path}")
    img = np.asarray(Image.open(sar_path))
    zones = np.asarray(Image.open(zone_path))
    if img.shape != zones.shape:
        raise ValueError(f"{sar_path.name}: image {img.shape} vs zones {zones.shape}")

    tiles, pad_valid, positions = tile_image(img, spec)
    ztiles, _, _ = tile_image(zones, spec, fill=0)     # pad zones as NA
    labels, patch_valid = tile_patch_targets(
        ztiles, pad_valid, patch=patch, min_purity=min_purity, with_na=with_na)
    labels3, valid3 = (tile_patch_targets(ztiles, pad_valid, patch=patch, min_purity=min_purity)
                       if with_na else (labels, patch_valid))   # 3-class set, same definition as notebooks 03-05
    if norm == "per_tile":             # legacy: the first probe run
        x = to_rgb(normalise(tiles, norm, pad_valid))
    else:                               # per_image / caffe / unit: whole image first
        xt, _, _ = tile_image(normalise_image(img, norm), spec, fill=0.0)
        x = to_rgb(xt)
    meta = parse_name(sar_path.name)
    glacier = meta.get("glacier") or sar_path.stem.split("_")[0]
    return {"stem": sar_path.stem, "x": x, "labels": labels, "valid": patch_valid,
            "labels3": labels3, "valid3": valid3,
            "raw": tiles, "pad_valid": pad_valid, "zone_tiles": ztiles,
            "positions": np.array(positions), "shape": img.shape, "glacier": glacier,
            "split": split_of(glacier), "satellite": meta.get("satellite"),
            "band": meta.get("band"), "date": meta.get("date"),
            "resolution_m": meta.get("resolution_m")}


def _select(df: pd.DataFrame, splits, limit) -> list:
    work = df.copy()
    work["glacier"] = work["path"].map(lambda p: Path(p).stem.split("_")[0])
    if splits is not None:
        work = work[work["glacier"].map(split_of).isin(splits)]
        print(f"restricted to splits {splits}: {len(work)} of {len(df)} images")
    return list(work["path"])[:limit]


def _prefetched(paths, fn, workers: int = 4, depth: int = 8):
    """Yield fn(path) in order, computing up to `depth` items ahead in threads."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    it = iter(paths)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        q = deque(ex.submit(fn, p) for _, p in zip(range(depth), it))
        while q:
            item = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(fn, nxt))
            yield item


def _tile_copy(feats: np.ndarray, it: dict, t: int) -> tuple:
    """One kept training tile, copied: a view would keep the whole image's
    feature array alive, i.e. every training tile instead of the reservoir."""
    return (np.copy(feats[t]), np.copy(it["raw"][t]), np.copy(it["zone_tiles"][t]),
            np.copy(it["pad_valid"][t]))


def _reservoir_slot(rng: np.random.Generator, n_kept: int, n_seen: int, cap: int | None,
                    legacy: bool = False) -> int | None:
    """Where candidate training tile number `n_seen` (1-based, counting every
    candidate so far across images, this one included) goes in a list that holds
    `n_kept` tiles, at most `cap`: `n_kept` = append, a smaller index = replace
    that slot, None = drop. Nothing is drawn until the list is full, then exactly
    one `rng.integers` per tile.

    Default: Algorithm R, a uniform sample of all tiles seen. `legacy=True`
    repeats the draw of the runs reported up to 2026-09-25, j in [0, 4 * cap):
    every later tile replaced a slot with probability 1/4 however many tiles came
    before, so early tiles in file order were kept more often (code review
    2026-09-25; docs/results.md, caveats).
    """
    if cap is None or n_kept < cap:
        return n_kept
    j = int(rng.integers(0, n_kept * 4)) if legacy else int(rng.integers(0, n_seen))
    return j if j < cap else None


def extract_in_memory(df: pd.DataFrame, data_root: Path, encoder, spec: TileSpec, norm: str,
                      splits: tuple[str, ...] = ("train", "val"), with_na: bool = False,
                      train_frac: float = 0.21, seed: int = 0, batch: int | None = None,
                      limit: int | None = None, eval_split: str = "val",
                      layer: int | None = None, keep_tiles: bool = False,
                      max_train_tiles: int | None = None, legacy_tile_sampler: bool = False) -> dict:
    """Extract without touching the disk (speed-up, 2026-09-24).

    Keeps only what the probe uses: a random `train_frac` of the valid training
    patches (0.21 of 9.5 M ~= the 2 M subsample used so far) and every patch of
    the validation images (needed for dense zone maps). CPU preparation of the
    next images overlaps with GPU encoding. Returns
    {"train_X" (N, D) fp16, "train_y" (N,), "train_y3" (N,) 3-class labels with
    IGNORE on NA patches, "eval" {stem: arrays} for `eval_split`, "manifest" df}.

    `keep_tiles` also keeps whole training tiles for the dense heads, at most
    `max_train_tiles` of them as a uniform reservoir sample (`_reservoir_slot`);
    `legacy_tile_sampler=True` reproduces the tilted sample of the reported runs.
    """
    rng = np.random.default_rng(seed)
    paths = _select(df, splits, limit)
    t0 = time.perf_counter()
    xs, ys, y3s, val, rows = [], [], [], {}, []
    train_tiles: list = []                   # (features, raw, zone_tiles, pad_valid) per kept tile
    n_seen = 0                               # candidate training tiles so far, across images
    prep = lambda p: _prepare(p, data_root, spec, norm, encoder.patch, with_na=with_na)  # noqa: E731
    gen, i, t_wait, t_enc = _prefetched(paths, prep), 0, 0.0, 0.0
    while True:
        tw = time.perf_counter()
        it = next(gen, None)                 # time spent here = waiting for CPU preparation
        t_wait += time.perf_counter() - tw
        if it is None:
            break
        i += 1
        te = time.perf_counter()
        feats = encoder.encode(it["x"], batch=batch, layer=layer)   # GPU forward + copy back
        t_enc += time.perf_counter() - te
        ok = it["valid"]
        if it["split"] == "train":
            fv, lv, l3 = feats[ok], it["labels"][ok], it["labels3"][ok]
            keep = rng.random(len(fv)) < train_frac
            xs.append(fv[keep]); ys.append(lv[keep]); y3s.append(l3[keep])
            if keep_tiles:
                # whole tiles for the dense heads (rows 2-6), capped at max_train_tiles:
                # uniform reservoir sample, or the reported runs' tilted draw if legacy
                for t in range(len(feats)):
                    n_seen += 1
                    slot = _reservoir_slot(rng, len(train_tiles), n_seen, max_train_tiles,
                                           legacy=legacy_tile_sampler)
                    if slot is None:
                        continue
                    if slot == len(train_tiles):
                        train_tiles.append(_tile_copy(feats, it, t))
                    else:
                        train_tiles[slot] = _tile_copy(feats, it, t)
        elif it["split"] == eval_split:
            val[it["stem"]] = {"features": feats, "labels": it["labels"], "valid": ok,
                               "labels3": it["labels3"], "valid3": it["valid3"],
                               "positions": it["positions"], "shape": it["shape"]}
            if keep_tiles:
                val[it["stem"]].update(raw=it["raw"], pad_valid=it["pad_valid"])
        rows.append({k: it[k] for k in ("stem", "glacier", "satellite", "band", "date",
                                        "resolution_m", "split")} |
                    {"n_tiles": len(it["x"]), "grid": feats.shape[1], "dim": feats.shape[-1],
                     "valid_patches": int(ok.sum())})
        if i % 50 == 0 or i == len(paths):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(paths)}  {el/60:.1f} min elapsed, {el/i:.2f} s/image, "
                  f"ETA {(len(paths)-i)*el/i/60:.1f} min")
    X = np.concatenate(xs) if xs else np.empty((0, 0), np.float16)
    y = np.concatenate(ys) if ys else np.empty((0,), np.uint8)
    y3 = np.concatenate(y3s) if y3s else np.empty((0,), np.uint8)
    total = time.perf_counter() - t0
    print(f"in memory: train X {X.shape} ({X.nbytes/1e9:.1f} GB fp16), {len(val)} {eval_split} images")
    print(f"extraction time: waiting for CPU preparation {t_wait/60:.1f} min, encoder {t_enc/60:.1f} min, "
          f"keeping/copying {(total - t_wait - t_enc)/60:.1f} min, total {total/60:.1f} min")
    return {"train_X": X, "train_y": y, "train_y3": y3, "eval": val,
            "train_tiles": train_tiles, "layer": layer, "manifest": pd.DataFrame(rows)}


def extract_dataset(df: pd.DataFrame, data_root: Path, out_dir: Path, encoder,
                    spec: TileSpec, norm: str, batch: int | None = None,
                    min_purity: float = 0.0, limit: int | None = None,
                    splits: tuple[str, ...] | None = None, with_na: bool = False) -> pd.DataFrame:
    """Extract features + patch targets for every image in `df`.

    `df` is the `sar_images` rows from `inventory.scan`. Writes one .npz per
    image into `out_dir` and returns a manifest DataFrame, also written as
    `manifest.csv`.

    `splits` restricts the work, e.g. `("train", "val")` to leave the test
    glaciers untouched until final evaluation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    paths = _select(df, splits, limit)
    prep = lambda p: _prepare(p, data_root, spec, norm, encoder.patch,  # noqa: E731
                              min_purity=min_purity, with_na=with_na)

    for i, it in enumerate(_prefetched(paths, prep), 1):
        sar_path = Path(paths[i - 1])
        tiles, labels, patch_valid, positions = it["x"], it["labels"], it["valid"], it["positions"]
        feats = encoder.encode(it["x"], batch=batch)
        meta = {"satellite": it["satellite"], "band": it["band"], "date": it["date"],
                "resolution_m": it["resolution_m"]}
        glacier = it["glacier"]
        # Uncompressed on purpose: float16 activations barely compress, and
        # zlib on ~36 GB would cost far more than the 2 min of extraction.
        dest = out_dir / f"{sar_path.stem}.npz"
        np.savez(dest, features=feats, labels=labels,
                 valid=patch_valid, positions=np.array(positions))

        rows.append({"stem": sar_path.stem, "glacier": glacier,
                     "satellite": meta.get("satellite"), "band": meta.get("band"),
                     "date": meta.get("date"), "resolution_m": meta.get("resolution_m"),
                     "split": split_of(glacier), "n_tiles": len(tiles),
                     "grid": feats.shape[1], "dim": feats.shape[-1],
                     "valid_patches": int(patch_valid.sum()),
                     "bytes": dest.stat().st_size, "file": dest.name})

        if i % 50 == 0 or i == len(paths):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(paths)}  {el/60:.1f} min elapsed, "
                  f"{el/i:.2f} s/image, ETA {(len(paths)-i)*el/i/60:.1f} min")

    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "encoder": encoder.spec.name, "model_id": encoder.spec.model_id,
        "tile": spec.size, "overlap": spec.overlap, "normalisation": norm,
        "patch": encoder.patch, "dim": encoder.dim, "n_prefix": encoder._n_prefix,
        "min_purity": min_purity, "with_na": with_na, "images": len(man),
        "tiles": int(man["n_tiles"].sum()),
        "gb": round(float(man["bytes"].sum()) / 1e9, 2),
    }, indent=2), encoding="utf-8")
    return man


def rebuild_manifest(out_dir: Path) -> pd.DataFrame:
    """Re-derive the manifest from .npz files already on disk.

    Cheaper than re-extracting when only the manifest columns changed.
    """
    out_dir = Path(out_dir)
    rows = []
    for f in sorted(out_dir.glob("*.npz")):
        with np.load(f) as z:
            feats, valid = z["features"], z["valid"]
        meta = parse_name(f.stem + ".png")
        glacier = meta.get("glacier") or f.stem.split("_")[0]
        rows.append({"stem": f.stem, "glacier": glacier,
                     "satellite": meta.get("satellite"), "band": meta.get("band"),
                     "date": meta.get("date"), "resolution_m": meta.get("resolution_m"),
                     "split": split_of(glacier), "n_tiles": feats.shape[0],
                     "grid": feats.shape[1], "dim": feats.shape[-1],
                     "valid_patches": int(valid.sum()),
                     "bytes": f.stat().st_size, "file": f.name})
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    return man


def load_split(out_dir: Path, split: str, manifest: pd.DataFrame | None = None,
               max_patches: int | None = None, seed: int = 0):
    """Stack valid patch vectors of one split into (X, y).

    `max_patches` subsamples **per file as it reads**, never materialising the
    full split. The training split holds 9.5 M patches x 1024 dims, which would
    be ~39 GB as float32 -- far more than a 3 k-parameter linear probe needs.
    """
    out_dir = Path(out_dir)
    man = manifest if manifest is not None else pd.read_csv(out_dir / "manifest.csv")
    rows = man[man["split"] == split]
    rng = np.random.default_rng(seed)

    frac = None
    if max_patches:
        total = int(rows["valid_patches"].sum())
        if total > max_patches:
            frac = max_patches / total
            print(f"subsampling {split}: {total:,} valid patches -> ~{max_patches:,} "
                  f"({frac:.1%} per file)")

    xs, ys = [], []
    for stem in rows["stem"]:
        with np.load(out_dir / f"{stem}.npz") as z:
            f, lab, ok = z["features"], z["labels"], z["valid"]
        fv, lv = f[ok], lab[ok]
        if frac is not None and len(fv):
            keep = rng.random(len(fv)) < frac
            fv, lv = fv[keep], lv[keep]
        if len(fv):
            xs.append(fv.astype(np.float32))
            ys.append(lv)
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    print(f"{split}: X {X.shape} {X.nbytes/1e9:.2f} GB, y {y.shape}")
    return X, y
