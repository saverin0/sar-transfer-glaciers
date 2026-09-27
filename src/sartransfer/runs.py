"""Run the whole encoder comparison in one loop: extract -> probe -> save -> clean up.

One run = one frozen encoder + one input normalisation. For each run:

1. install the encoder's extra packages if missing,
2. extract frozen patch features for train + val to local disk (`features.py`),
3. train the linear zone probe (unweighted and class-balanced, `probe.py`),
4. score it on the validation glacier SI per band / sensor / resolution,
5. save `probe_zones__<run>.csv` + `.pt` to the results folder on Drive,
6. delete the run's features (31-38 GB each) unless asked to keep them.

A run whose results CSV already exists is skipped, so an interrupted loop can
simply be started again.
"""

from __future__ import annotations

import gc
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# (import name, pip name) per encoder kind. C-RADIO's remote code is
# import-checked by transformers, so open_clip must exist even though its
# adaptor is never used here.
EXTRA_PACKAGES = {
    "radio": [("timm", "timm"), ("einops", "einops"), ("open_clip", "open_clip_torch")],
    "terramind": [("terratorch", "terratorch")],
}

GROUPINGS = ("band", "satellite", "resolution_m", "sensor_res")


def run_name(encoder: str, norm: str, tile: int = 512) -> str:
    return f"{encoder}__{norm}__tile{tile}"


def ensure_packages(kind: str) -> None:
    missing = [pip for mod, pip in EXTRA_PACKAGES.get(kind, [])
               if importlib.util.find_spec(mod) is None]
    if missing:
        print("installing", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
        importlib.invalidate_caches()


def probe_run(feat_dir: Path, run: str, max_train: int = 2_000_000,
              seed: int = 0) -> tuple[pd.DataFrame, dict, dict]:
    """Train both probes on train patches, score on SI. Returns (results, probes, baseline)."""
    import torch
    from .features import load_split, rebuild_manifest
    from .probe import confusion, majority_baseline, report, train_linear_probe

    feat_dir = Path(feat_dir)
    man = pd.read_csv(feat_dir / "manifest.csv")
    if "band" not in man.columns:
        man = rebuild_manifest(feat_dir)
    man["sensor_res"] = (man["satellite"] + "@"
                         + man["resolution_m"].astype(int).astype(str) + "m")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = load_split(feat_dir, "train", man, max_patches=max_train, seed=seed)
    counts = np.bincount(y, minlength=3)
    probes = {tag: train_linear_probe(X, y, balanced=(tag == "balanced"),
                                      epochs=30, lr=1e-3, device=device, verbose=False)
              for tag in ("unweighted", "balanced")}
    del X, y
    gc.collect()

    base = majority_baseline(man, feat_dir, "val", counts)
    tables = []
    for tag, probe in probes.items():
        for grouping in GROUPINGS:
            r = report(confusion(probe, feat_dir, man, "val", device=device, group_by=grouping))
            r.insert(0, "grouping", grouping)
            r.insert(0, "probe", tag)
            tables.append(r)
    res = pd.concat(tables, ignore_index=True)
    res.insert(0, "run", run)
    return res, probes, base


def run_encoder(encoder: str, norm: str, df: pd.DataFrame, data_root: Path,
                feat_root: Path, res_dir: Path, token: str | None,
                tile: int = 512, splits: tuple[str, ...] = ("train", "val"),
                keep_features: bool = False, force: bool = False) -> pd.DataFrame | None:
    """Extract + probe + save one run. Returns its results, or None if skipped."""
    import torch

    from .data.tiling import TileSpec
    from .features import extract_dataset
    from .models.encoders import ENCODERS, FrozenEncoder

    run = run_name(encoder, norm, tile)
    res_dir = Path(res_dir)
    res_csv = res_dir / f"probe_zones__{run}.csv"
    if res_csv.exists() and not force:
        print(f"skip {run}: results already on Drive")
        return None

    spec = ENCODERS[encoder]
    out = Path(feat_root) / run
    t0 = time.perf_counter()
    print(f"\n===== {run} =====")

    if not (out / "meta.json").exists():
        ensure_packages(spec.kind)
        enc = FrozenEncoder(spec, token=token)
        try:
            extract_dataset(df, data_root, out, enc, TileSpec(size=tile, overlap=0.0),
                            norm, splits=splits)
            print(f"extracted: {enc} in {(time.perf_counter() - t0) / 60:.1f} min")
        finally:
            del enc
            gc.collect()
            torch.cuda.empty_cache()

    res, probes, base = probe_run(out, run)
    res_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(res_csv, index=False)
    torch.save({k: v.state_dict() for k, v in probes.items()}, res_dir / f"probe_zones__{run}.pt")

    overall = res[(res.probe == "balanced") & (res.grouping == "band") & (res.group == "all")].iloc[0]
    print(f"mIoU {overall.mIoU:.4f} (balanced, SI) vs majority baseline {base['mIoU']:.4f} "
          f"| total {(time.perf_counter() - t0) / 60:.1f} min | saved {res_csv.name}")

    if not keep_features:
        shutil.rmtree(out, ignore_errors=True)
    return res


def summarise(res_dir: Path, probe: str = "balanced") -> dict[str, pd.DataFrame]:
    """mIoU tables across every results CSV in `res_dir`: runs x groups."""
    files = sorted(Path(res_dir).glob("probe_zones__*.csv"))
    if not files:
        raise FileNotFoundError(f"no probe_zones__*.csv in {res_dir}")
    allres = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allres = allres[allres["probe"] == probe]
    out = {}
    for grouping in GROUPINGS:
        t = allres[allres["grouping"] == grouping].pivot_table(
            index="run", columns="group", values="mIoU", aggfunc="first")
        cols = ["all"] + sorted(c for c in t.columns if c != "all")
        out[grouping] = t[cols].sort_values("all", ascending=False)
    per_class = allres[(allres["grouping"] == "band") & (allres["group"] == "all")]
    out["per_class"] = (per_class.set_index("run")[["IoU_stone", "IoU_glacier", "IoU_ocean", "mIoU"]]
                        .sort_values("mIoU", ascending=False))
    return out


# ============================================================ notebook 06: fronts (MDE)

def _front_inputs(data_root: Path, stem: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """True 4-class zone ids, true front mask and bbox path for one SI/train image."""
    from PIL import Image
    from .data.targets import zones_to_class

    from .features import TEST_GLACIERS
    split_dir = "test" if stem.split("_")[0] in TEST_GLACIERS else "train"   # SI lives in train
    data_root = Path(data_root)
    zone_raw = np.asarray(Image.open(data_root / "zones" / split_dir / f"{stem}_zones.png"))
    z4, _ = zones_to_class(zone_raw, with_na=True)
    front = np.asarray(Image.open(data_root / "fronts" / split_dir / f"{stem}_front.png")) > 0
    bbox = data_root / "bounding_boxes" / f"{stem}_front_extent_coord.txt"
    return z4, front, bbox


def _meta(stem: str) -> dict:
    from .data.inventory import parse_name
    m = parse_name(stem + ".png")
    return {"stem": stem, "glacier": m["glacier"], "satellite": m["satellite"], "band": m["band"],
            "resolution_m": m["resolution_m"], "sensor_res": f"{m['satellite']}@{int(m['resolution_m'])}m"}


FINAL_GROUPINGS = ("glacier",) + GROUPINGS


def _per_image_iou(m: np.ndarray) -> dict:
    """Per-image IoU per class, NaN where a class is absent from both truth and
    prediction -- the paper's per-image definition (sklearn jaccard per class,
    NaN filtered when averaging), plus the per-image macro mean over 4 classes."""
    tp = np.diag(m).astype(float)
    denom = m.sum(0) + m.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    names = ("na", "stone", "glacier", "ocean")
    out = {f"IoU_{n}": float(v) for n, v in zip(names, iou)}
    out["mIoU"] = float(np.nanmean(iou))
    return out


def _front_one(args: tuple) -> tuple[dict, np.ndarray, dict]:
    """One image of the official route (module level so worker processes can run it)."""
    from .fronts import zone_front_record

    stem, pred, data_root = args
    z4, true_front, bbox = _front_inputs(Path(data_root), stem)
    meta = _meta(stem)
    rec = zone_front_record(pred, true_front, bbox, meta["resolution_m"], meta)
    m = np.bincount(z4.ravel().astype(np.int64) * 4 + pred.ravel().astype(np.int64),
                    minlength=16).reshape(4, 4)
    rec.update({f"img_{k}": v for k, v in _per_image_iou(m).items()})
    return rec, m, meta


def _score_workers(n_images: int, workers: int | None) -> int:
    """Processes for front scoring: explicit, else $SARTRANSFER_SCORE_WORKERS, else all
    cores on Linux/Colab. 1 on Windows, where spawned workers would re-run a
    script without a __main__ guard."""
    import os

    if workers is None:
        env = os.environ.get("SARTRANSFER_SCORE_WORKERS")
        workers = int(env) if env else (1 if sys.platform == "win32" else (os.cpu_count() or 1))
    return max(1, min(int(workers), n_images, 32))


def fronts_from_zone_maps(zone_maps: dict, data_root: Path, run: str,
                          groupings=GROUPINGS, workers: int | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Official zones -> front -> MDE route for {stem: 4-class id map}.

    Returns (per-image records, pooled MDE table, pixel-level 4-class zone IoU
    pooled over pixels). Per-image records also carry the paper-style
    per-image IoUs (`img_IoU_*`, `img_mIoU`). Images are scored in parallel
    worker processes (speed-up 2026-09-25); results are collected in the input
    order, so every table is identical to the serial loop.
    """
    from .fronts import mde_table
    from .probe import report

    jobs = [(stem, pred, str(data_root)) for stem, pred in zone_maps.items()]
    n_workers = _score_workers(len(jobs), workers)
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_workers) as pool:   # spawn: no fork after CUDA/threads
            results = pool.map(_front_one, jobs, chunksize=1)
    else:
        results = [_front_one(j) for j in jobs]

    recs, mats = [], {g: {"all": np.zeros((4, 4), np.int64)} for g in groupings}
    for rec, m, meta in results:
        recs.append(rec)
        for g in groupings:
            mats[g]["all"] += m
            key = str(meta[g])
            mats[g].setdefault(key, np.zeros((4, 4), np.int64))
            mats[g][key] += m
    zones = []
    for g in groupings:
        r = report(mats[g]); r.insert(0, "grouping", g); zones.append(r)
    zones = pd.concat(zones, ignore_index=True)
    zones.insert(0, "run", run)
    records = pd.DataFrame(recs)
    records.insert(0, "run", run)
    return records, mde_table(recs, run=run, groupings=groupings), zones


def paper_zone_table(records: pd.DataFrame, groupings=FINAL_GROUPINGS) -> pd.DataFrame:
    """Paper-style zone metric: per-image IoU averaged over images (NaN skipped)."""
    cols = [c for c in records.columns if c.startswith("img_")]
    rows = [{"grouping": "all", "group": "all", "images": len(records),
             **{c[4:]: records[c].mean(skipna=True) for c in cols}}]
    for g in groupings:
        for key, sub in records.groupby(g):
            rows.append({"grouping": g, "group": str(key), "images": len(sub),
                         **{c[4:]: sub[c].mean(skipna=True) for c in cols}})
    out = pd.DataFrame(rows)
    out.insert(0, "run", records["run"].iloc[0])
    return out


def run_fronts(encoder: str, norm: str, df: pd.DataFrame, data_root: Path,
               feat_root: Path, res_dir: Path, token: str | None, tile: int = 512,
               keep_features: bool = False, force: bool = False) -> pd.DataFrame | None:
    """Notebook 06 for one frozen encoder: 4-class features -> 4-class probe -> SI fronts -> MDE.

    Fast path (2026-09-24): features stay in RAM (2 M training patches + all SI
    patches), CPU preparation overlaps GPU encoding, probe data sits on the GPU.
    Nothing is written to disk except the results. `feat_root` / `keep_features`
    are kept for call compatibility and unused.
    """
    import torch

    from .data.tiling import TileSpec
    from .features import extract_in_memory
    from .fronts import probe_zone_map
    from .models.encoders import ENCODERS, FrozenEncoder
    from .probe import train_linear_probe

    run = f"{run_name(encoder, norm, tile)}__na4"
    res_dir = Path(res_dir)
    out_csv = res_dir / f"fronts_mde__{run}.csv"
    if out_csv.exists() and not force:
        print(f"skip {run}: front results already on Drive")
        return None
    spec = ENCODERS[encoder]
    t0 = time.perf_counter()
    print(f"\n===== {run} =====")

    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    try:
        data = extract_in_memory(df, data_root, enc, TileSpec(size=tile, overlap=0.0), norm,
                                 splits=("train", "val"), with_na=True)
    finally:
        del enc
        gc.collect()
        torch.cuda.empty_cache()
    t_ext = time.perf_counter() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = train_linear_probe(data["train_X"], data["train_y"], balanced=True, n_classes=4,
                               epochs=30, lr=1e-3, device=device, verbose=False)
    W = probe.weight.detach().float().cpu().numpy()
    b = probe.bias.detach().float().cpu().numpy()
    zone_maps = {stem: probe_zone_map(v["features"], v["positions"], v["shape"], W, b)
                 for stem, v in data["eval"].items()}
    del data
    gc.collect()
    torch.cuda.empty_cache()

    records, mde, zones = fronts_from_zone_maps(zone_maps, data_root, run)
    res_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(res_dir / f"fronts_images__{run}.csv", index=False)
    zones.to_csv(res_dir / f"zones4_pixel__{run}.csv", index=False)
    mde.to_csv(out_csv, index=False)
    torch.save(probe.state_dict(), res_dir / f"probe4__{run}.pt")
    a = mde[mde.grouping == "all"].iloc[0]
    print(f"MDE {a.MDE_m:.0f} m over {a.images} SI images ({a.no_front} without a predicted front) "
          f"| extraction {t_ext / 60:.1f} min, total {(time.perf_counter() - t0) / 60:.1f} min")
    return mde


def unet_fronts(model, val_imgs: list[dict], data_root: Path, run: str, device: str = "cuda"):
    """Notebook 06 for the 4-class U-Net: dense prediction -> official front route -> MDE."""
    from .baseline import predict

    zone_maps = {im["stem"]: predict(model, im["x"], device) for im in val_imgs}
    return fronts_from_zone_maps(zone_maps, data_root, run)


# ================================================= notebook 07: final test evaluation

def _save_final(res_dir: Path, run: str, records, mde, zones, paper, zones3=None):
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(res_dir / f"TEST_fronts_images__{run}.csv", index=False)
    mde.to_csv(res_dir / f"TEST_fronts_mde__{run}.csv", index=False)
    zones.to_csv(res_dir / f"TEST_zones4_pixel__{run}.csv", index=False)
    paper.to_csv(res_dir / f"TEST_zones4_paper__{run}.csv", index=False)
    if zones3 is not None:
        zones3.to_csv(res_dir / f"TEST_probe_zones3__{run}.csv", index=False)
    a = mde[mde.grouping == "all"].iloc[0]
    per_gl = mde[mde.grouping == "glacier"].set_index("group")["MDE_m"].round(0).to_dict()
    print(f"TEST: MDE {a.MDE_m:.0f} m ({a.no_front} of {a.images} images without a front) "
          f"| per glacier {per_gl} | paper-style mIoU {paper.iloc[0]['mIoU']:.3f}")


def run_final_encoder(encoder: str, norm: str, df: pd.DataFrame, data_root: Path,
                      res_dir: Path, token: str | None, tile: int = 512,
                      force: bool = False) -> bool:
    """Notebook 07 for one frozen encoder, on the TEST glaciers (COL, Mapple).

    From one in-memory extraction of train + test: (a) the 3-class probe, scored
    at patch level exactly as in notebooks 03-05; (b) the 4-class probe -> dense zone
    maps -> official fronts / MDE, pixel IoU pooled AND paper-style per image.
    """
    import torch

    from .data.tiling import TileSpec
    from .features import extract_in_memory
    from .fronts import probe_zone_map
    from .models.encoders import ENCODERS, FrozenEncoder
    from .probe import confusion_from_arrays, report, train_linear_probe

    run = run_name(encoder, norm, tile)
    res_dir = Path(res_dir)
    if (res_dir / f"TEST_fronts_mde__{run}.csv").exists() and not force:
        print(f"skip {run}: test results already on Drive")
        return False
    spec = ENCODERS[encoder]
    t0 = time.perf_counter()
    print(f"\n===== TEST {run} =====")
    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    try:
        data = extract_in_memory(df, data_root, enc, TileSpec(size=tile, overlap=0.0), norm,
                                 splits=("train", "test"), with_na=True, eval_split="test")
    finally:
        del enc
        gc.collect()
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (a) 3-class probe, patch level, same as notebooks 03-05
    from .data.targets import IGNORE
    keep = data["train_y3"] != IGNORE
    p3 = train_linear_probe(data["train_X"][keep], data["train_y3"][keep], balanced=True,
                            n_classes=3, epochs=30, lr=1e-3, device=device, verbose=False)
    man = data["manifest"]
    man = man[man["split"] == "test"].copy()
    man["sensor_res"] = man["satellite"] + "@" + man["resolution_m"].astype(int).astype(str) + "m"
    mats = confusion_from_arrays(p3, data["eval"], man, device, FINAL_GROUPINGS,
                                 feat_key="features", lab_key="labels3", ok_key="valid3")
    zones3 = []
    for g in FINAL_GROUPINGS:
        r = report(mats[g]); r.insert(0, "grouping", g); zones3.append(r)
    zones3 = pd.concat(zones3, ignore_index=True)
    zones3.insert(0, "run", run)

    # (b) 4-class probe -> fronts
    p4 = train_linear_probe(data["train_X"], data["train_y"], balanced=True, n_classes=4,
                            epochs=30, lr=1e-3, device=device, verbose=False)
    W, b = p4.weight.detach().float().cpu().numpy(), p4.bias.detach().float().cpu().numpy()
    zone_maps = {stem: probe_zone_map(v["features"], v["positions"], v["shape"], W, b)
                 for stem, v in data["eval"].items()}
    del data
    gc.collect()
    torch.cuda.empty_cache()
    records, mde, zones = fronts_from_zone_maps(zone_maps, data_root, run, groupings=FINAL_GROUPINGS)
    paper = paper_zone_table(records)
    _save_final(res_dir, run, records, mde, zones, paper, zones3)
    torch.save({"probe3": p3.state_dict(), "probe4": p4.state_dict()}, res_dir / f"TEST_probes__{run}.pt")
    print(f"  3-class patch mIoU on test: {zones3[(zones3.grouping == 'glacier') & (zones3.group == 'all')].mIoU.iloc[0]:.4f}"
          f" | {(time.perf_counter() - t0) / 60:.1f} min")
    return True


def run_final_unet(weights_path: Path, n_classes: int, test_imgs: list[dict], data_root: Path,
                   res_dir: Path, run: str, device: str = "cuda", base: int = 32,
                   force: bool = False) -> bool:
    """Notebook 07 for a saved U-Net: no retraining, just dense prediction on the test images."""
    import torch

    from .baseline import evaluate, predict, to_table
    from .models.unet import UNet

    res_dir = Path(res_dir)
    marker = res_dir / (f"TEST_fronts_mde__{run}.csv" if n_classes == 4 else f"TEST_probe_zones3__{run}.csv")
    if marker.exists() and not force:
        print(f"skip {run}: test results already on Drive")
        return False
    print(f"\n===== TEST {run} ({n_classes} classes, saved weights) =====")
    model = UNet(in_ch=1, n_classes=n_classes, base=base).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    if n_classes == 3:
        preds = {im["stem"]: predict(model, im["x"], device) for im in test_imgs}   # one pass, used twice
        conf = evaluate(model, test_imgs, device, preds=preds)
        # add per-glacier grouping on top of evaluate()'s groupings
        from .probe import report
        from .data.targets import N_CLASSES
        from .baseline import _confusion, _patch_level
        mats = {"all": np.zeros((3, 3), np.int64)}
        for im in test_imgs:
            t_lab, p_lab = _patch_level(im["y"], preds[im["stem"]])
            m = _confusion(t_lab, p_lab)
            mats["all"] += m
            mats.setdefault(im["glacier"], np.zeros((3, 3), np.int64))
            mats[im["glacier"]] += m
        r = report(mats); r.insert(0, "grouping", "glacier")
        patch = pd.concat([r, to_table(conf["patch"], run)], ignore_index=True)
        patch["run"] = run
        patch.to_csv(res_dir / f"TEST_probe_zones3__{run}.csv", index=False)
        pixel = to_table(conf["pixel"], run)
        pixel.to_csv(res_dir / f"TEST_unet_pixel3__{run}.csv", index=False)
        print(f"  3-class patch mIoU on test: {r[r.group == 'all'].mIoU.iloc[0]:.4f}")
        return True
    zone_maps = {im["stem"]: predict(model, im["x"], device) for im in test_imgs}
    records, mde, zones = fronts_from_zone_maps(zone_maps, data_root, run, groupings=FINAL_GROUPINGS)
    paper = paper_zone_table(records)
    _save_final(res_dir, run, records, mde, zones, paper)
    return True


# ====================================== second arm, step 1: which layer to tap?

def layer_sweep(encoder: str, norm: str, df: pd.DataFrame, data_root: Path, res_dir: Path,
                token: str | None, layers: list[int] | None = None, tile: int = 512,
                train_frac: float = 0.10, seed: int = 0, force: bool = False,
                probe_kwargs: dict | None = None) -> pd.DataFrame | None:
    """3-class linear probe on SI for several encoder layers, one pass per split.

    Pass 1 (train glaciers): collect a `train_frac` subsample of valid patches per
    layer, train one balanced probe per layer. Pass 2 (SI): score each probe,
    streaming image by image (nothing kept). Criterion for the layer choice is
    zone patch mIoU on SI -- a cheap proxy, declared as such in docs/results.md.
    """
    import torch

    from .data.targets import IGNORE
    from .data.tiling import TileSpec
    from .features import _prefetched, _prepare, _select
    from .models.encoders import ENCODERS, FrozenEncoder
    from .probe import iou_from_confusion, train_linear_probe

    run = run_name(encoder, norm, tile)
    res_dir = Path(res_dir)
    out_csv = res_dir / f"layersweep__{run}.csv"
    if out_csv.exists() and not force:
        print(f"skip {run}: layer sweep already on Drive")
        return None
    spec = ENCODERS[encoder]
    t0 = time.perf_counter()
    print(f"\n===== layer sweep {run} =====")
    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    n = enc.n_layers
    if layers is None:
        layers = sorted({max(1, round(n * f)) for f in (0.25, 0.5, 0.75, 0.875, 1.0)})
    print(f"{n} blocks; tapping layers {layers}")
    tspec = TileSpec(size=tile, overlap=0.0)
    prep = lambda p_: _prepare(p_, data_root, tspec, norm, enc.patch)  # noqa: E731
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pass 1: train subsample per layer
    xs = {k: [] for k in layers}
    ys = []
    for it in _prefetched(_select(df, ("train",), None), prep):
        ok = it["valid"]
        keep = rng.random(int(ok.sum())) < train_frac
        feats = enc.encode_layers(it["x"], layers)
        for k in layers:
            xs[k].append(feats[k][ok][keep])
        ys.append(it["labels"][ok][keep])
    y = np.concatenate(ys)
    probes = {}
    for k in layers:
        X = np.concatenate(xs[k]); xs[k] = None
        kw = {"epochs": 30, "lr": 1e-3, **(probe_kwargs or {})}     # real runs: defaults
        probes[k] = train_linear_probe(X, y, balanced=True, n_classes=3, device=device,
                                       verbose=False, **kw)
        del X
    print(f"trained {len(layers)} probes on {len(y):,} patches, {(time.perf_counter()-t0)/60:.1f} min")
    del xs, ys, y
    gc.collect()

    # pass 2: score on SI, streaming
    groups = ("band", "resolution_m", "sensor_res")
    mats = {k: {g: {"all": np.zeros((3, 3), np.int64)} for g in groups} for k in layers}
    with torch.inference_mode():
        for it in _prefetched(_select(df, ("val",), None), prep):
            ok = it["valid"]
            if not ok.any():
                continue
            feats = enc.encode_layers(it["x"], layers)
            true = it["labels"][ok].astype(np.int64)
            meta = {"band": it["band"], "resolution_m": it["resolution_m"],
                    "sensor_res": f"{it['satellite']}@{int(it['resolution_m'])}m"}
            for k in layers:
                xb = torch.from_numpy(feats[k][ok].astype(np.float32)).to(device)
                pred = probes[k](xb).argmax(1).cpu().numpy()
                m = np.bincount(true * 3 + pred, minlength=9).reshape(3, 3)
                for g in groups:
                    mats[k][g]["all"] += m
                    key = str(meta[g])
                    mats[k][g].setdefault(key, np.zeros((3, 3), np.int64))
                    mats[k][g][key] += m
    del enc
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for k in layers:
        for g in groups:
            for key, m in mats[k][g].items():
                rows.append({"run": run, "layer": k, "n_layers": n, "grouping": g, "group": key,
                             **iou_from_confusion(m)})
    res = pd.DataFrame(rows)
    res_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    overall = res[(res.grouping == "band") & (res.group == "all")].set_index("layer")["mIoU"]
    print("SI patch mIoU per layer:", overall.round(4).to_dict(),
          f"| best layer {int(overall.idxmax())} | {(time.perf_counter()-t0)/60:.1f} min")
    return res
