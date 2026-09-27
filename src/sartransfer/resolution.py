"""Experiment B, 2026-09-27: is the X-band vs C-band difference a pixel-size effect?

In CaFFe the radar band always comes with its own pixel size (X-band only at
7 m, L-band only at 17 m, C-band at 12 and 20 m), so the per-band results
cannot say which one matters. Test, on the validation glacier SI only (the
test glaciers stay untouched):

1. `degrade`: the 32 X-band (TerraSAR-X, 7 m) SI images are shrunk to 20 m
   pixels: radar image by area average, zone labels by nearest neighbour (keeps
   the official grey values), the true front re-thinned to a 1 px line, the
   front bounding box scaled. The pixel size in the file name becomes 20, so
   the official scoring route measures in 20 m pixels, as for a real 20 m image.
2. `model_maps`: the SAVED models, unchanged (nothing retrained): U-Net 4-class,
   C-RADIO and satellite-DINOv3 linear probes (row 1) and decoders (row 3,
   seeds 0/1/2), on the original 7 m images and on the shrunk 20 m copies.
   Control for the heads: the 7 m rerun must reproduce the saved SI results
   exactly (front + per-image IoU), or the run stops. The U-Net is different
   (amended before any number, after review): its saved SI file came from the
   in-memory model right after training (channels-last, cuDNN auto-tuning) and
   has no per-image IoU, so a fresh load cannot match it bit for bit. Its X@7,
   X@20 AND C@20 are therefore all rerun on one path, and the comparison with
   the saved file is reported as information only.
3. `resolution_table`: pooled MDE (official pooling) of X@7 m, X@20 m and the
   SI C-band images that really are 20 m (ERS, Envisat, RADARSAT at 20 m):
   from the saved SI results for the heads, from the rerun for the U-Net.

Pre-registered rule (docs/results.md, 2026-09-27, before any number):
r = (MDE X@20 - MDE X@7) / (MDE C@20 - MDE X@7). Pixel size explains the X-vs-C
difference for a model if r >= 0.5 (X at 20 m moves at least halfway toward
C-band at 20 m); otherwise it does not. If |MDE C@20 - MDE X@7| < 100 m there
is no difference to explain for that model. Caveats stated with every result:
at equal pixel size the rest still mixes band with satellite, year and scene;
a shrunk X-band image is not identical to a real 20 m product (speckle,
processing); SI only, 32 images; exploratory, changes no earlier choice.

Follow-up, pre-registered 2026-09-27 after experiment B: the same test WITHIN
C-band (`source=("C", 12.0, "RSAT")`): the 21 SI RADARSAT 12 m images shrunk to
20 m, so any change is pixel size alone (same band and satellite). Primary
read-out for every source: pixel size hurts a model if its pooled MDE is at
least 100 m worse on the shrunk copies than on the originals. `band_table`:
pooled MDE per glacier, band, satellite and pixel size from the saved results
(SI and test), descriptive only.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.inventory import parse_name
from .features import zone_path_for
from .predmaps import _pooled
from .runs import run_name

NORM, TILE = "caffe", 512
ENC = {"cradio": ("cradio-v4-h", 32), "dinosat": ("dinov3-l-sat", 21)}


def _head_key(enc: str, row: int, seed: int = 0) -> str:
    return f"{enc}_row{row}" + (f"_s{seed}" if row == 3 else "")


MODELS = {"unet": {"kind": "unet", "weights": "unet__unet-scratch-48k__caffe__tile256__na4.pt",
                   "saved": "fronts_images__unet-scratch-48k__caffe__tile256__na4.csv"}}
for _e, (_name, _layer) in ENC.items():
    for _row, _seeds in ((1, (0,)), (3, (0, 1, 2))):
        for _s in _seeds:
            _tag = f"{run_name(_name, NORM, TILE)}__L{_layer}__row{_row}" + (f"__s{_s}" if _s else "")
            MODELS[_head_key(_e, _row, _s)] = {"kind": "head", "enc": _e, "row": _row, "seed": _s,
                                               "weights": f"heads__{_tag}.pt",
                                               "saved": f"heads_val_images__{_tag}.csv"}


def _new_stem(stem: str, target_m: float) -> str:
    parts = stem.split("_")
    parts[3] = f"{target_m:g}"
    return "_".join(parts)


def degrade(df: pd.DataFrame, data_root: str | Path, out_root: str | Path, target_m: float = 20.0,
            glacier: str = "SI", satellite: str = "TSX") -> pd.DataFrame:
    """Write shrunk copies of the chosen images in the CaFFe layout under out_root.

    Returns one row per image: original stem, new stem, sizes, scale factor."""
    from PIL import Image
    from skimage.morphology import skeletonize

    data_root, out_root = Path(data_root), Path(out_root)
    rows = []
    for p in sorted(Path(x) for x in df["path"]):
        m = parse_name(p.name)
        if not m.get("parsed") or m["glacier"] != glacier or m["satellite"] != satellite:
            continue
        split_dir, stem = p.parent.name, p.stem
        new = _new_stem(stem, target_m)
        arr = np.asarray(Image.open(p))
        H, W = arr.shape[:2]
        f = float(m["resolution_m"]) / target_m
        size = (max(1, round(W * f)), max(1, round(H * f)))
        for sub in ("sar_images", "zones", "fronts"):
            (out_root / sub / split_dir).mkdir(parents=True, exist_ok=True)
        (out_root / "bounding_boxes").mkdir(parents=True, exist_ok=True)
        # radar image: area average, back to the original integer type
        small = np.asarray(Image.fromarray(arr.astype(np.float32)).resize(size, Image.BOX))   # float -> mode F
        Image.fromarray(np.rint(small).clip(np.iinfo(arr.dtype).min, np.iinfo(arr.dtype).max)
                        .astype(arr.dtype)).save(out_root / "sar_images" / split_dir / f"{new}.png")
        # zones: nearest neighbour keeps only the official grey values
        zones = Image.open(zone_path_for(p, data_root)).resize(size, Image.NEAREST)
        zones.save(out_root / "zones" / split_dir / f"{new}_zones.png")
        # true front: any front pixel in the block counts, then thinned back to a 1 px line
        front = np.asarray(Image.open(data_root / "fronts" / split_dir / f"{stem}_front.png")) > 0
        fs = np.asarray(Image.fromarray(front.astype(np.uint8) * 255).resize(size, Image.BOX)) > 0
        Image.fromarray(skeletonize(fs).astype(np.uint8) * 255).save(
            out_root / "fronts" / split_dir / f"{new}_front.png")
        # bounding box: same corners, scaled
        lines = [ln for ln in (data_root / "bounding_boxes" / f"{stem}_front_extent_coord.txt")
                 .read_text().splitlines() if ln.strip()]
        sx, sy = size[0] / W, size[1] / H
        pts = [tuple(float(v) for v in ln.split(",")) for ln in lines[1:5]]
        (out_root / "bounding_boxes" / f"{new}_front_extent_coord.txt").write_text(
            lines[0] + "\n" + "\n".join(f"{x * sx:.3f},{y * sy:.3f}" for x, y in pts) + "\n")
        rows.append({"stem": stem, "stem_20": new, "w": W, "h": H, "w_20": size[0], "h_20": size[1],
                     "factor": f, "front_px": int(front.sum()), "front_px_20": int(skeletonize(fs).sum())})
    out = pd.DataFrame(rows)
    print(f"shrunk {len(out)} {glacier} {satellite} images to {target_m:g} m pixels -> {out_root}")
    return out


def model_maps(df: pd.DataFrame, data_root: str | Path, res_dir: str | Path, token: str | None,
               keys: tuple[str, ...] = tuple(MODELS)) -> dict[str, dict]:
    """{model key: {stem: 4-class map}} for the SI images in df, from the saved models."""
    import torch

    from .baseline import load_split_images, predict
    from .data.tiling import TileSpec
    from .features import extract_in_memory
    from .heads import _head_maps, row1_maps
    from .models.decoder import GridDecoder, load_head_state
    from .models.encoders import ENCODERS, FrozenEncoder
    from .models.unet import UNet
    from .runs import ensure_packages

    data_root, res_dir = Path(data_root), Path(res_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, dict] = {}
    if "unet" in keys:
        ims = load_split_images(df, data_root, "val", norm=NORM, with_na=True)
        model = UNet(in_ch=1, n_classes=4, base=32).to(device)
        model.load_state_dict(torch.load(res_dir / MODELS["unet"]["weights"], map_location=device, weights_only=True))
        model.eval()
        out["unet"] = {im["stem"]: predict(model, im["x"], device) for im in ims}
        del ims, model
    for e, (name, layer) in ENC.items():
        ks = [k for k in keys if MODELS[k]["kind"] == "head" and MODELS[k]["enc"] == e]
        if not ks:
            continue
        ensure_packages(ENCODERS[name].kind)
        enc = FrozenEncoder(ENCODERS[name], token=token)
        try:
            data = extract_in_memory(df, data_root, enc, TileSpec(size=TILE, overlap=0.0), NORM,
                                     splits=("val",), with_na=True, eval_split="val",
                                     layer=layer, keep_tiles=True)
        finally:
            del enc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        D = next(iter(data["eval"].values()))["features"].shape[-1]
        for k in ks:
            spec = MODELS[k]
            state = torch.load(res_dir / spec["weights"], map_location="cpu", weights_only=True)
            if spec["row"] == 1:
                head = torch.nn.Linear(D, 4)
                head.load_state_dict(state)
                out[k] = row1_maps(head.to(device), data)
            else:
                head = load_head_state(GridDecoder(D), state).to(device)
                out[k] = _head_maps(head, data, device, anyup=None, batch=8)
            del head
        del data
        gc.collect()
    return out


def score(maps: dict[str, dict], data_root: str | Path, workers: int | None = None) -> dict[str, pd.DataFrame]:
    """Official zone -> front -> distance route per model; per-image records."""
    from .runs import fronts_from_zone_maps

    return {k: fronts_from_zone_maps(m, data_root, k, groupings=("satellite",), workers=workers)[0]
            for k, m in maps.items()}


def control_check(rec7: dict[str, pd.DataFrame], res_dir: str | Path) -> pd.DataFrame:
    """The 7 m rerun vs the saved SI per-image results on the same images.

    Strict (must be identical) for the heads; information only for the U-Net
    (see the module docstring)."""
    rows = []
    for k, rec in rec7.items():
        saved = pd.read_csv(Path(res_dir) / MODELS[k]["saved"])
        saved = saved[saved.stem.isin(rec.stem)]
        m = rec.merge(saved, on="stem", suffixes=("", "_saved"))
        same = ((m.has_front == m.has_front_saved) & (m.front_px == m.front_px_saved)
                & (((m.mean_m - m.mean_m_saved).abs() < 0.01) | (m.mean_m.isna() & m.mean_m_saved.isna())))
        zones_checked = "img_mIoU_saved" in m      # older SI files may predate the per-image IoU columns
        if zones_checked:
            same &= ((m.img_mIoU - m.img_mIoU_saved).abs() < 1e-9) | (m.img_mIoU.isna() & m.img_mIoU_saved.isna())
        rows.append({"model": k, "strict": MODELS[k]["kind"] == "head", "images": len(rec), "matched": len(m),
                     "identical": int(same.sum()), "zone_iou_checked": zones_checked,
                     "pooled_diff_m": _pooled(rec) - _pooled(saved)})
    out = pd.DataFrame(rows)
    strict = out[out.strict]
    ok = (strict.identical == strict.images).all()
    print("control, heads (rerun at native pixel size == saved SI results): " + ("IDENTICAL for every head" if ok else
          "DIFFERENT -- stop and check\n" + strict.to_string(index=False)))
    for r in out[~out.strict].itertuples():
        print(f"control, {r.model} (information only): identical on {r.identical} of {r.images} images, "
              f"pooled MDE {r.pooled_diff_m:+.1f} m vs the saved file (in-memory model after training)")
    return out


def resolution_table(rec7: dict[str, pd.DataFrame], rec20: dict[str, pd.DataFrame],
                     res_dir: str | Path, c20_rerun: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Pooled MDE of the source images at their own pixel size (`native`), shrunk to 20 m, and real
    C-band@20 on SI per model, with the pre-registered verdicts (`rec7` = native records).

    C@20 comes from the saved SI results, or from `c20_rerun[model]` when given (the U-Net)."""
    rows = []
    for k in rec7:
        if c20_rerun and k in c20_rerun:
            c20, c20_src = c20_rerun[k], "rerun"
        else:
            saved = pd.read_csv(Path(res_dir) / MODELS[k]["saved"])
            c20, c20_src = saved[(saved.band.astype(str) == "C") & (saved.resolution_m.astype(float) == 20.0)], "saved"
        x7, x20 = rec7[k], rec20[k]
        a, b, c = _pooled(x7), _pooled(x20), _pooled(c20)
        pair = x7[["stem", "mean_m"]].assign(stem=lambda d: d.stem.map(lambda s: _new_stem(s, 20.0))) \
            .merge(x20[["stem", "mean_m"]], on="stem", suffixes=("_7", "_20"))
        gap = c - a
        r = (b - a) / gap if abs(gap) >= 100 else float("nan")
        verdict = ("no difference to C@20 to explain (< 100 m)" if abs(gap) < 100 else
                   "pixel size explains it (r >= 0.5)" if r >= 0.5 else "pixel size does not explain it (r < 0.5)")
        change = b - a
        effect = ("pixel size hurts (>= 100 m worse at 20 m)" if change >= 100 else
                  "better at 20 m (>= 100 m)" if change <= -100 else "no clear change (< 100 m)")
        rows.append({"model": k, "native": a, "shrunk_20m": b, "C_20m": c, "change_m": change,
                     "pixel_size_effect": effect, "r": r, "verdict": verdict,
                     "native_minus_C": a - c, "shrunk_minus_C": b - c,
                     "median_change_per_image_m": float((pair.mean_m_20 - pair.mean_m_7).median()),
                     "no_front_native": int((~x7.has_front.astype(bool)).sum()),
                     "no_front_20m": int((~x20.has_front.astype(bool)).sum()),
                     "C_20m_images": len(c20), "C_20m_from": c20_src, "source_images": len(x7)})
    return pd.DataFrame(rows)


def run_resolution_test(df: pd.DataFrame, data_root: str | Path, res_dir: str | Path, work_root: str | Path,
                        token: str | None, keys: tuple[str, ...] = tuple(MODELS),
                        source: tuple[str, float, str] = ("X", 7.0, "TSX")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps 1-3 for the SI images of `source` = (band, pixel size in m, satellite).

    Returns (control check, result table). Files go to work_root/<tag>/restest_<tag>_*.csv
    (tag e.g. "X7", "C12"); nothing is written to res_dir."""
    from .data import inventory as inv

    t0 = time.perf_counter()
    band, res_m, sat = source
    tag = f"{band}{res_m:g}"
    work_root = Path(work_root) / tag
    def si(p, band, res):
        m = parse_name(Path(p).name)
        return bool(m.get("parsed")) and m["glacier"] == "SI" and m["band"] == band and float(m["resolution_m"]) == res
    sel = df[df["path"].map(lambda p: si(p, band, res_m) and f"_{sat}_" in Path(p).name)]
    c20 = df[df["path"].map(lambda p: si(p, "C", 20.0))]
    if not len(sel):
        raise ValueError(f"no SI images for {source}")
    info = degrade(sel, data_root, work_root, satellite=sat)
    df20 = inv.scan(work_root, subdirs=("sar_images",))
    print(f"\n--- {res_m:g} m (original, control): {len(sel)} {band}-band {sat} images ---")
    heads = tuple(k for k in keys if MODELS[k]["kind"] == "head")
    maps7 = model_maps(sel, data_root, res_dir, token, heads) if heads else {}
    c20_rerun = {}
    if "unet" in keys:                      # U-Net: X@7 and C@20 from one fresh-load path
        u = model_maps(pd.concat([sel, c20]), data_root, res_dir, token, ("unet",))["unet"]
        xs = {Path(p).stem for p in sel["path"]}
        maps7["unet"] = {s: m for s, m in u.items() if s in xs}
        c20_rerun = score({"unet": {s: m for s, m in u.items() if s not in xs}}, data_root)
        print(f"  U-Net C@20 rerun on {len(c20_rerun['unet'])} SI C-band 20 m images")
    rec7 = score(maps7, data_root)
    control = control_check(rec7, res_dir)
    strict = control[control.strict]
    if (strict.identical != strict.images).any():
        control.to_csv(work_root / f"restest_{tag}_control.csv", index=False)
        raise RuntimeError("a head's rerun does not reproduce the saved SI results -- stopped before the 20 m run")
    print("\n--- 20 m (shrunk copies) ---")
    rec20 = score(model_maps(df20, work_root, res_dir, token, keys), work_root)
    table = resolution_table(rec7, rec20, res_dir, c20_rerun)
    for d, name in ((info, "images"), (pd.concat(rec7.values()), "records_native"),
                    (pd.concat(rec20.values()), "records_20m"), (control, "control"), (table, "table")):
        d.to_csv(work_root / f"restest_{tag}_{name}.csv", index=False)
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min")
    return control, table


# ------------------------------------------------------------------ all bands, from the saved results

TEST_FILES = {"unet": "TEST_fronts_images__unet-scratch-48k__caffe__tile256__na4.csv"}


def _saved_file(key: str, split: str) -> str:
    if split == "SI":
        return MODELS[key]["saved"]
    return TEST_FILES.get(key) or MODELS[key]["saved"].replace("heads_val_images__", "heads_test_images__")


def band_table(res_dir: str | Path, keys: tuple[str, ...] = tuple(MODELS)) -> pd.DataFrame:
    """Pooled MDE per (split, glacier, band, satellite, pixel size) and per model, from the saved
    per-image results (SI and test). Also pooled over satellites per band and pixel size
    (satellite = "all"). Descriptive: groups differ in satellite, years and scenes too."""
    rows = []
    for k in keys:
        for split in ("SI", "test"):
            d = pd.read_csv(Path(res_dir) / _saved_file(k, split))
            d["resolution_m"] = d["resolution_m"].astype(float)
            for by in (["glacier", "band", "satellite", "resolution_m"], ["glacier", "band", "resolution_m"]):
                for key, g in d.groupby(by):
                    rec = dict(zip(by, key))
                    rows.append({"split": split, "glacier": rec["glacier"], "band": rec["band"],
                                 "satellite": rec.get("satellite", "all"), "pixel_m": rec["resolution_m"],
                                 "model": k, "images": len(g), "MDE": _pooled(g),
                                 "no_front": int((~g.has_front.astype(bool)).sum())})
    return pd.DataFrame(rows)


def band_summary(bt: pd.DataFrame) -> pd.DataFrame:
    """One line per group, one column per model family (decoders: mean of the 3 seeds' pooled MDEs)."""
    fam = {k: k.rsplit("_s", 1)[0] if k.endswith(("_s0", "_s1", "_s2")) else k for k in bt.model.unique()}
    t = bt.assign(family=bt.model.map(fam))
    idx = ["split", "glacier", "band", "satellite", "pixel_m"]
    wide = t.pivot_table(index=idx + ["images"], columns="family", values="MDE", aggfunc="mean")
    return wide.reset_index().sort_values(["split", "glacier", "band", "pixel_m", "satellite"], ignore_index=True)
