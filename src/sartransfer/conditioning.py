"""Approach A (exploratory, after the one test run): tell the row-3 decoder the
radar band and pixel size of each image.

Question: in the first arm, band differences on SI looked like resolution
differences; the data cannot settle this. Does a decoder that KNOWS an image's
band and ground resolution draw better fronts?

Design (matched pair, everything else identical):
    plain  row 3's GridDecoder, retrained here
    film   the same GridDecoder plus FiLM: a small MLP maps the metadata
           vector to a per-channel scale and shift after each of the five
           decoder stages. Its last layer starts at zero, so at step 0 the
           film head computes exactly what the plain head computes.
Both variants of a seed start from identical decoder weights, see identical
tiles in identical order, with identical flips.

Metadata vector (4 values, from the file name only): one-hot band X / C / L
and resolution_m / 20. Band and resolution, not the satellite name: the test
glaciers include satellites that never occur in training, but their bands and
resolutions do. The glacier name is never used.

Kept apart from the second arm on purpose: new file, own result names
(`cond_*` on Drive), nothing in heads.py / features.py / decoder.py changed.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.targets import IGNORE

BANDS = ("X", "C", "L")
META_DIM = 4
VARIANTS = ("plain", "film")


def meta_vector(band: str | None, resolution_m: float | None) -> np.ndarray:
    """One-hot band X/C/L + resolution in units of 20 m."""
    if band not in BANDS or resolution_m is None:
        raise ValueError(f"cannot build metadata from band={band!r}, resolution_m={resolution_m!r}")
    v = np.zeros(META_DIM, np.float32)
    v[BANDS.index(band)] = 1.0
    v[3] = float(resolution_m) / 20.0
    return v


def _film_decoder_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .models.decoder import GridDecoder

    class FiLMGridDecoder(GridDecoder):
        """GridDecoder + FiLM after each of its five stages (identity at init)."""

        def __init__(self, feat_dim: int, n_classes: int = 4, width: int = 96, hidden: int = 32):
            super().__init__(feat_dim, n_classes, width)          # same RNG use as GridDecoder
            self.stage_ch = [m[0].out_channels for m in self.up]  # conv of each stage
            self.film = nn.Sequential(nn.Linear(META_DIM, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 2 * sum(self.stage_ch)))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

        def forward(self, feats: torch.Tensor, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
            parts = torch.split(self.film(meta), [2 * c for c in self.stage_ch], dim=1)

            def mod(x, p):
                g, b = p.chunk(2, dim=1)
                return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

            maps = self.stem(img)
            x = mod(self.up[0](torch.cat([self.reduce(feats), maps[4]], 1)), parts[0])
            for i, m in zip(range(1, 5), (maps[3], maps[2], maps[1], maps[0])):
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = mod(self.up[i](torch.cat([x, m], 1)), parts[i])
            return self.head(x)

    return FiLMGridDecoder


def _build(variant: str, D: int, seed: int):
    import torch

    from .models.decoder import GridDecoder

    torch.manual_seed(seed)                                   # seed fixes the initial weights
    return GridDecoder(D) if variant == "plain" else _film_decoder_class()(D)


def _call(head, variant, f, img, meta):
    return head(f, img) if variant == "plain" else head(f, img, meta)


# ------------------------------------------------------------------ data

def _extract(df, data_root, enc, spec, norm, layer, split, cap, seed=0) -> dict:
    """Frozen features at `layer` + per-image metadata, no disk.

    Training tiles (split="val" only): uniform reservoir sample (Algorithm R)
    of at most `cap` tiles, each carrying its image's metadata vector.
    """
    from .features import _prefetched, _prepare, _select

    splits = ("train", split) if split == "val" else (split,)
    paths = _select(df, splits, None)
    rng = np.random.default_rng(seed)
    tiles, ev, n_seen, t0 = [], {}, 0, time.perf_counter()
    prep = lambda p: _prepare(p, data_root, spec, norm, enc.patch, with_na=True)  # noqa: E731
    gen, k, t_wait, t_enc = _prefetched(paths, prep), 0, 0.0, 0.0
    while True:
        tw = time.perf_counter()
        it = next(gen, None)                 # time spent here = waiting for CPU preparation
        t_wait += time.perf_counter() - tw
        if it is None:
            break
        k += 1
        if it["split"] not in splits:
            continue
        te = time.perf_counter()
        feats = enc.encode(it["x"], layer=layer)                    # GPU forward + copy back
        t_enc += time.perf_counter() - te
        m = meta_vector(it["band"], it["resolution_m"])
        if it["split"] == "train":
            for t in range(len(feats)):
                n_seen += 1
                item = (np.copy(feats[t]), np.copy(it["raw"][t]), np.copy(it["zone_tiles"][t]),
                        np.copy(it["pad_valid"][t]), m)
                if len(tiles) < cap:
                    tiles.append(item)
                else:
                    j = int(rng.integers(0, n_seen))
                    if j < cap:
                        tiles[j] = item
        else:
            ev[it["stem"]] = {"features": feats, "raw": it["raw"], "pad_valid": it["pad_valid"],
                              "positions": it["positions"], "shape": it["shape"], "meta": m}
        if k % 100 == 0:
            print(f"  {k}/{len(paths)}  {(time.perf_counter() - t0) / 60:.1f} min")
    total = time.perf_counter() - t0
    print(f"features ready in {total / 60:.1f} min | {len(tiles)} of {n_seen} "
          f"training tiles kept | {len(ev)} {split} images")
    print(f"extraction time: waiting for CPU preparation {t_wait / 60:.1f} min, encoder {t_enc / 60:.1f} min, "
          f"keeping/copying {(total - t_wait - t_enc) / 60:.1f} min")
    return {"tiles": tiles, "eval": ev}


# ------------------------------------------------------------------ train / score

def _train(head, variant, tiles, device, steps=4000, batch=8, lr=1e-3, seed=0, log_every=1000):
    """Same recipe as heads._train_head (AdamW + OneCycle, balanced CE, flips)."""
    import torch
    import torch.nn as nn

    from .heads import TrainSet, _class_weights, _pixel_labels

    rng = np.random.default_rng(seed)
    labels = [_pixel_labels(zt, pv) for _, _, zt, pv, _ in tiles]
    w = torch.tensor(_class_weights(labels), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    head = head.to(device)
    ts = TrainSet([t[0] for t in tiles], [t[1] for t in tiles], labels, device,
                  extra=[t[4] for t in tiles])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0, running, n = time.perf_counter(), torch.zeros((), device=device), 0
    for it in range(1, steps + 1):
        head.train()
        idx = rng.choice(len(tiles), batch, replace=len(tiles) < batch)
        f, raw, y, meta = ts.batch(idx)
        if rng.random() < 0.5:
            f, raw, y = f.flip(-1), raw.flip(-1), y.flip(-1)
        img = raw.float() / 255.0
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = loss_fn(_call(head, variant, f, img, meta).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.detach()                      # read back only at log lines (no per-step sync)
        n += 1
        if it % log_every == 0 or it == steps:
            print(f"    step {it:>5}  loss {float(running) / n:.4f}  {(time.perf_counter() - t0) / 60:.1f} min")
            running, n = torch.zeros((), device=device), 0
    return head


def _maps(head, variant, ev, device, batch=8) -> dict:
    import torch

    from .heads import _stitch

    head.eval()
    out = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for stem, v in ev.items():
            meta = torch.from_numpy(v["meta"])[None].to(device)
            logits = []
            for i in range(0, len(v["features"]), batch):
                f = torch.from_numpy(v["features"][i:i + batch]).to(device).float().permute(0, 3, 1, 2)
                raw = torch.from_numpy(v["raw"][i:i + batch])[:, None].to(device)
                o = _call(head, variant, f, raw.float() / 255.0, meta.expand(len(f), -1))
                logits.append(o.float().cpu().numpy())
            out[stem] = _stitch(np.concatenate(logits), v["positions"], v["shape"])
    return out


# ------------------------------------------------------------------ driver

def run_conditioning(encoder: str, norm: str, layer: int, df: pd.DataFrame, data_root: Path,
                     res_dir: Path, token: str | None, seeds=(0, 1, 2), variants=VARIANTS,
                     split: str = "val", tile: int = 512, max_train_tiles: int = 4000,
                     steps: int = 4000, batch: int = 8, force: bool = False) -> pd.DataFrame:
    """Train (split="val") or load (other splits) plain and FiLM decoders, score on `split`.

    Writes `cond_<split>_{images,zones4,mde}__<run>__<variant>__s<seed>.csv`, the
    summary `cond_<split>_summary__<run>.csv` and, when training, the heads
    `cond__<run>__<variant>__s<seed>.pt`. Scored (variant, seed) pairs are skipped.
    """
    import torch

    from .data.tiling import TileSpec
    from .models.encoders import ENCODERS, FrozenEncoder
    from .runs import FINAL_GROUPINGS, GROUPINGS, ensure_packages, fronts_from_zone_maps, run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"{run_name(encoder, norm, tile)}__L{layer}"
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    tag = lambda v, s: f"{run}__{v}__s{s}"  # noqa: E731
    jobs = [(v, s) for v in variants for s in seeds
            if force or not (res_dir / f"cond_{split}_mde__{tag(v, s)}.csv").exists()]
    if not jobs:
        print(f"skip {run}: every variant/seed already scored on {split}")
        return pd.DataFrame()
    print(f"\n===== conditioning {run} on {split}: {jobs} =====")

    spec = ENCODERS[encoder]
    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    try:
        data = _extract(df, data_root, enc, TileSpec(size=tile, overlap=0.0), norm, layer, split,
                        max_train_tiles)
    finally:
        del enc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    D = next(iter(data["eval"].values()))["features"].shape[-1]
    groupings = GROUPINGS if split == "val" else FINAL_GROUPINGS

    summary = []
    for v, s in jobs:
        t1 = time.perf_counter()
        wpath = res_dir / f"cond__{tag(v, s)}.pt"
        head = _build(v, D, s)
        if split == "val":
            head = _train(head, v, data["tiles"], device, steps=steps, batch=batch, seed=s)
            torch.save(head.state_dict(), wpath)
        else:
            head.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True), strict=True)
            head = head.to(device)
        maps = _maps(head, v, data["eval"], device, batch=batch)
        records, mde, zones = fronts_from_zone_maps(maps, data_root, tag(v, s), groupings=groupings)
        records.to_csv(res_dir / f"cond_{split}_images__{tag(v, s)}.csv", index=False)
        zones.to_csv(res_dir / f"cond_{split}_zones4__{tag(v, s)}.csv", index=False)
        mde.to_csv(res_dir / f"cond_{split}_mde__{tag(v, s)}.csv", index=False)
        a = mde[mde.grouping == "all"].iloc[0]
        z = zones[(zones.grouping == groupings[0]) & (zones.group == "all")].iloc[0]
        n_par = sum(p.numel() for p in head.parameters())
        summary.append({"run": run, "variant": v, "seed": s, "params": n_par, "MDE_m": a.MDE_m,
                        "no_front": a.no_front, "images": a.images, "mIoU4_pixel": z.mIoU,
                        "minutes": (time.perf_counter() - t1) / 60})
        print(f"  {v:<5} s{s}  {n_par / 1e3:6.1f} k params | MDE {a.MDE_m:6.0f} m ({a.no_front} no front) "
              f"| 4-class pixel mIoU {z.mIoU:.3f} | {(time.perf_counter() - t1) / 60:.1f} min")
        del head, maps
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    del data
    gc.collect()
    out = pd.DataFrame(summary)
    spath = res_dir / f"cond_{split}_summary__{run}.csv"
    merged = out
    if spath.exists():
        old = pd.read_csv(spath)
        done = set(zip(out.variant, out.seed))
        merged = pd.concat([old[[(a, b) not in done for a, b in zip(old.variant, old.seed)]], out],
                           ignore_index=True)
    merged.sort_values(["variant", "seed"]).to_csv(spath, index=False)
    return out


# ------------------------------------------------------------------ tables

def conditioning_tables(res_dir: Path, split: str = "val") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per run and variant over seeds, with the pre-registered verdict; MDE per sensor@resolution)."""
    import glob

    res_dir = Path(res_dir)
    summ = [pd.read_csv(f) for f in sorted(glob.glob(str(res_dir / f"cond_{split}_summary__*.csv")))]
    if not summ:
        raise FileNotFoundError(f"no cond_{split}_summary__*.csv in {res_dir}")
    s = pd.concat(summ, ignore_index=True)
    agg = (s.groupby(["run", "variant"])
             .agg(seeds=("seed", "count"), MDE_mean=("MDE_m", "mean"), MDE_sd=("MDE_m", "std"),
                  no_front_total=("no_front", "sum"), mIoU4_mean=("mIoU4_pixel", "mean"),
                  params=("params", "first"))
             .reset_index())
    verdict = {}
    for run, g in agg.groupby("run"):
        if set(g.variant) != set(VARIANTS):
            continue
        p, f = g.set_index("variant").loc["plain"], g.set_index("variant").loc["film"]
        if min(p.seeds, f.seeds) < 2:                 # one seed: no spread to compare with
            verdict[run] = "not enough seeds for the pre-registered rule (need >= 2 per variant)"
            continue
        gap = p.MDE_mean - f.MDE_mean
        thr = max(p.MDE_sd, f.MDE_sd)
        if gap > thr and f.no_front_total <= p.no_front_total:
            verdict[run] = f"FiLM HELPS: {gap:.0f} m lower, more than the larger seed spread ({thr:.0f} m)"
        elif -gap > thr:
            verdict[run] = f"FiLM HURTS: {-gap:.0f} m higher, more than the larger seed spread ({thr:.0f} m)"
        else:
            verdict[run] = f"no clear effect: difference {gap:+.0f} m is within the seed spread ({thr:.0f} m)"
    agg["verdict"] = agg.run.map(verdict)

    frames = []
    for f in sorted(glob.glob(str(res_dir / f"cond_{split}_mde__*.csv"))):
        t = Path(f).stem.split(f"cond_{split}_mde__", 1)[1]
        run, variant, seed = t.rsplit("__", 2)
        d = pd.read_csv(f)
        d = d[d.grouping == "sensor_res"].assign(run=run, variant=variant)
        frames.append(d[["run", "variant", "group", "MDE_m"]])
    per_sensor = (pd.concat(frames).pivot_table(index=["run", "variant"], columns="group", values="MDE_m",
                                                aggfunc="mean").round(0)
                  if frames else pd.DataFrame())
    return agg, per_sensor
