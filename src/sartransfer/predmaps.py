"""Two looks at the saved TEST results; nothing is retrained and no result file is touched.

Part 1, `col_s1_breakdown` (CPU, reads CSVs): how much of each run's pooled test
MDE comes from the 18 Sentinel-1 images of Columbia, and the MDE without them.
A description of the test set after the fact; it changes no choice.

Part 2, `draw_predictions` (GPU): for a few saved heads, encode the test images
again exactly as the one-time test run did, load the saved head, and write its
4-class zone map as a PNG with the official grey values (0 NA, 64 stone,
127 glacier, 254 ocean) plus the front the official route draws from it
(thickened for display, value 2 so it gets its own colour next to the true
front's 1). Each image's error is scored again from these maps and compared
with the saved per-image CSV, so the pictures are checked to be the
predictions that were scored: the front error AND the per-image zone IoUs
must match (same IoUs is strong evidence of the same map, not a bit-for-bit
proof). Everything goes to a new output folder.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .runs import run_name

# The runs drawn in part 2: the reference, the two best decoders (seed 0) and a probe.
PRED_RUNS = {
    "unet": {"kind": "unet", "weights": "unet__unet-scratch-48k__caffe__tile256__na4.pt",
             "saved": "TEST_fronts_images__unet-scratch-48k__caffe__tile256__na4.csv"},
    "cradio_row1": {"kind": "head", "encoder": "cradio-v4-h", "layer": 32, "row": 1},
    "cradio_row3": {"kind": "head", "encoder": "cradio-v4-h", "layer": 32, "row": 3},
    "dinosat_row3": {"kind": "head", "encoder": "dinov3-l-sat", "layer": 21, "row": 3},
}
NORM, TILE = "caffe", 512                    # as in the reported test runs
PRED_FRONT_VALUE = 2
IOU_COLS = ["img_IoU_na", "img_IoU_stone", "img_IoU_glacier", "img_IoU_ocean", "img_mIoU"]


def head_tag(spec: dict) -> str:
    """Run tag of a saved head (seed 0), as heads.run_heads named its files."""
    return f"{run_name(spec['encoder'], NORM, TILE)}__L{spec['layer']}__row{spec['row']}"


def saved_images_csv(spec: dict) -> str:
    return spec["saved"] if spec["kind"] == "unet" else f"heads_test_images__{head_tag(spec)}.csv"


# ------------------------------------------------------------------ part 1

def _is_col_s1(df: pd.DataFrame) -> pd.Series:
    return (df["glacier"].astype(str) == "COL") & (df["satellite"].astype(str) == "S1")


def _pooled(df: pd.DataFrame) -> float:
    ok = df[df["has_front"].astype(bool)]
    return float(ok["sum_m"].sum() / ok["n_dist"].sum()) if len(ok) and ok["n_dist"].sum() else float("nan")


def col_s1_breakdown(res_dir: str | Path) -> pd.DataFrame:
    """One row per test run: pooled MDE with and without Columbia's Sentinel-1 images.

    Pooled exactly like the official metric (all point distances of all images
    with a front, then the mean). `share_error` = COL-S1 part of the summed
    distances; `share_points` = COL-S1 part of the distance points. `MDE_saved`
    is the "all" row of the saved MDE table, as a check of the pooling.
    """
    res_dir = Path(res_dir)
    files = sorted(res_dir.glob("TEST_fronts_images__*.csv")) + sorted(res_dir.glob("heads_test_images__*.csv"))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        if not {"glacier", "satellite", "has_front", "sum_m", "n_dist"} <= set(df.columns):
            print(f"  skipped {f.name}: columns missing")
            continue
        run = f.stem.split("images__", 1)[1]
        prefix = "arm1" if f.name.startswith("TEST_") else "heads"
        saved = res_dir / f.name.replace("_images__", "_mde__")
        mde_saved = (pd.read_csv(saved).query("grouping == 'all'")["MDE_m"].iloc[0]
                     if saved.exists() else float("nan"))
        cs = _is_col_s1(df)
        ok = df["has_front"].astype(bool)
        tot_err, tot_pts = df.loc[ok, "sum_m"].sum(), df.loc[ok, "n_dist"].sum()
        rows.append({"set": prefix, "run": run, "images": len(df), "col_s1_images": int(cs.sum()),
                     "MDE_all": _pooled(df), "MDE_saved": mde_saved,
                     "MDE_without_col_s1": _pooled(df[~cs]), "MDE_col_s1_only": _pooled(df[cs]),
                     "share_error": float(df.loc[ok & cs, "sum_m"].sum() / tot_err) if tot_err else float("nan"),
                     "share_points": float(df.loc[ok & cs, "n_dist"].sum() / tot_pts) if tot_pts else float("nan"),
                     "no_front": int((~ok).sum()), "no_front_col_s1": int((~ok & cs).sum())})
    out = pd.DataFrame(rows)
    if len(out):
        bad = (out["MDE_all"] - out["MDE_saved"]).abs() > 0.5
        print(f"pooling check: {int((~bad).sum())} of {len(out)} runs match their saved MDE within 0.5 m"
              + ("" if not bad.any() else f"; MISMATCH: {list(out.loc[bad, 'run'])}"))
    return out


# ------------------------------------------------------------------ part 2

def _draw_one(args: tuple) -> dict:
    """Front + PNGs + per-image error for one predicted map (module level for worker processes)."""
    from PIL import Image
    from scipy.ndimage import binary_dilation

    from .fronts import (GREY_OF_CLASS4, front_distances, front_from_zones, mask_with_bbox,
                         postprocess_zones, read_bbox)
    from .runs import _front_inputs, _meta, _per_image_iou

    stem, pred, data_root, out_dir = args
    z4, true_front, bbox = _front_inputs(Path(data_root), stem)
    res = _meta(stem)["resolution_m"]
    zone = GREY_OF_CLASS4[pred]
    front = mask_with_bbox(front_from_zones(postprocess_zones(zone), res), read_bbox(bbox))
    d = front_distances(front, true_front > 0)
    out_dir = Path(out_dir)
    Image.fromarray(zone).save(out_dir / f"{stem}_zones.png")
    width = max(2, round(max(front.shape) / 500))            # same thickening as viz._front_display
    thick = binary_dilation(front, iterations=width) if front.any() else front
    Image.fromarray((thick * PRED_FRONT_VALUE).astype(np.uint8)).save(out_dir / f"{stem}_front.png")
    m = np.bincount(z4.ravel().astype(np.int64) * 4 + pred.ravel().astype(np.int64), minlength=16).reshape(4, 4)
    return {"stem": stem, "has_front": d is not None, "front_px": int(front.sum()),
            "mean_m": float(d.mean() * res) if d is not None else float("nan"),
            **{f"img_{k}": v for k, v in _per_image_iou(m).items()}}


def _draw(maps: dict, key: str, data_root: Path, out_root: Path, res_dir: Path, spec: dict,
          workers: int | None = None) -> pd.DataFrame:
    """Write the PNGs of one run and compare its per-image errors with the saved CSV."""
    from .runs import _score_workers

    out_dir = out_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(stem, pred, str(data_root), str(out_dir)) for stem, pred in maps.items()]
    n = _score_workers(len(jobs), workers)
    if n > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n) as pool:
            recs = pool.map(_draw_one, jobs, chunksize=1)
    else:
        recs = [_draw_one(j) for j in jobs]
    now = pd.DataFrame(recs)
    saved = pd.read_csv(res_dir / saved_images_csv(spec))[["stem", "has_front", "front_px", "mean_m"] + IOU_COLS]
    cmp = now.merge(saved, on="stem", how="outer", suffixes=("", "_saved"), indicator=True)
    cmp.insert(0, "key", key)
    both = cmp["_merge"] == "both"
    diff = (cmp["mean_m"] - cmp["mean_m_saved"]).abs()
    same_front = (cmp["has_front"] == cmp["has_front_saved"]) & (cmp["front_px"] == cmp["front_px_saved"])
    front_ok = both & same_front & ((diff < 0.01) | (cmp["mean_m"].isna() & cmp["mean_m_saved"].isna()))
    zones_ok = both & np.logical_and.reduce(
        [((cmp[c] - cmp[f"{c}_saved"]).abs() < 1e-9) | (cmp[c].isna() & cmp[f"{c}_saved"].isna()) for c in IOU_COLS])
    cmp["front_match"], cmp["zones_match"] = front_ok, zones_ok
    print(f"  {key}: {len(now)} maps drawn | front identical on {int(front_ok.sum())}, zone map identical "
          f"(per-image IoU) on {int(zones_ok.sum())} of {int(both.sum())} images | largest front difference "
          f"{np.nanmax(diff.values) if diff.notna().any() else 0:.1f} m"
          + ("" if both.all() else f" | {int((~both).sum())} images not in both"))
    return cmp.drop(columns="_merge")


def draw_predictions(df: pd.DataFrame, data_root: str | Path, res_dir: str | Path, out_root: str | Path,
                     token: str | None, keys: tuple[str, ...] = tuple(PRED_RUNS),
                     workers: int | None = None) -> pd.DataFrame:
    """Part 2 for the chosen runs. Returns (and writes) the per-image comparison with the saved CSVs."""
    import torch

    from .baseline import load_split_images, predict
    from .data.tiling import TileSpec
    from .features import extract_in_memory
    from .heads import _head_maps, row1_maps
    from .models.decoder import GridDecoder, load_head_state
    from .models.encoders import ENCODERS, FrozenEncoder
    from .models.unet import UNet
    from .runs import ensure_packages

    data_root, res_dir, out_root = Path(data_root), Path(res_dir), Path(out_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = []

    # U-Net: saved weights, same sliding-window prediction as runs.run_final_unet
    for key in [k for k in keys if PRED_RUNS[k]["kind"] == "unet"]:
        t0, spec = time.perf_counter(), PRED_RUNS[key]
        print(f"\n===== {key} =====")
        te4 = load_split_images(df, data_root, "test", norm=NORM, with_na=True)
        model = UNet(in_ch=1, n_classes=4, base=32).to(device)
        model.load_state_dict(torch.load(res_dir / spec["weights"], map_location=device, weights_only=True))
        model.eval()
        maps = {im["stem"]: predict(model, im["x"], device) for im in te4}
        del te4, model
        torch.cuda.empty_cache()
        checks.append(_draw(maps, key, data_root, out_root, res_dir, spec, workers))
        print(f"  {(time.perf_counter() - t0) / 60:.1f} min")

    # saved heads: one extraction per encoder, the same call as heads.run_heads uses for the test split
    by_enc: dict[tuple, list[str]] = {}
    for key in [k for k in keys if PRED_RUNS[k]["kind"] == "head"]:
        s = PRED_RUNS[key]
        by_enc.setdefault((s["encoder"], s["layer"]), []).append(key)
    for (encoder, layer), ks in by_enc.items():
        t0 = time.perf_counter()
        print(f"\n===== {encoder} layer {layer}: {', '.join(ks)} =====")
        spec_e = ENCODERS[encoder]
        ensure_packages(spec_e.kind)
        enc = FrozenEncoder(spec_e, token=token)
        try:
            data = extract_in_memory(df, data_root, enc, TileSpec(size=TILE, overlap=0.0), NORM,
                                     splits=("test",), with_na=True, eval_split="test",
                                     layer=layer, keep_tiles=True)
        finally:
            del enc
            gc.collect()
            torch.cuda.empty_cache()
        try:
            import psutil
            print(f"  RAM used {psutil.virtual_memory().used / 1e9:.0f} of {psutil.virtual_memory().total / 1e9:.0f} GB")
        except Exception:
            pass
        D = next(iter(data["eval"].values()))["features"].shape[-1]
        for key in ks:
            spec = PRED_RUNS[key]
            wpath = res_dir / f"heads__{head_tag(spec)}.pt"
            if spec["row"] == 1:
                head = torch.nn.Linear(D, 4).to(device)
                head.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
                maps = row1_maps(head, data)
            elif spec["row"] == 3:
                head = load_head_state(GridDecoder(D), torch.load(wpath, map_location="cpu", weights_only=True)).to(device)
                maps = _head_maps(head, data, device, anyup=None, batch=8)
            else:
                raise ValueError(f"row {spec['row']} is not drawn here (rows 1 and 3 need no AnyUp)")
            checks.append(_draw(maps, key, data_root, out_root, res_dir, spec, workers))
            del head, maps
            torch.cuda.empty_cache()
        del data
        gc.collect()
        print(f"  {(time.perf_counter() - t0) / 60:.1f} min")

    out = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    out_root.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_root / "check_vs_saved.csv", index=False)
    return out
