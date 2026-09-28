"""Second arm: six rows of heads on the same frozen features -> zone maps -> fronts.

    row 1  linear probe on the 16 px grid, logits upsampled bilinearly
    row 2  AnyUp -> per-pixel features -> linear probe (trained on sampled
           pixels; evaluated exactly by upsampling the 4 logits, see
           anyup_loader.upsample_with_value)
    row 3  GridDecoder: small learned decoder with an image skip
    row 4  PixelHead: learned 1x1 reduction -> AnyUp attention -> light head
    row 5  GridDecoder on four layers concatenated (layer=(l1, l2, l3, l4))
    row 6  AnyUpDecoder: AnyUp to 1/4 resolution, then row 3's last stages

Every head is trained on the training glaciers only, for a fixed number of
steps, with no checkpoint picked on the evaluation split; scored through the
same official zone -> front -> MDE route as notebooks 06-07. Heads are saved to
Drive so the one-time test run loads them instead of retraining.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data.targets import IGNORE, N_CLASSES4, zones_to_class
from .runs import FINAL_GROUPINGS, GROUPINGS, fronts_from_zone_maps, paper_zone_table

if TYPE_CHECKING:                            # annotations only; torch stays a lazy import
    import torch

# Names as in the saved CSVs, kept: ROWS[4] "anyup-decoder" is class PixelHead, ROWS[6] "anyup128-decoder" is class AnyUpDecoder.
ROWS = {1: "linear-grid", 2: "anyup-linear", 3: "grid-decoder", 4: "anyup-decoder",
        5: "grid-decoder-4L", 6: "anyup128-decoder"}


# ------------------------------------------------------------------ helpers

def _stitch(tile_logits: np.ndarray, positions: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(N, C, T, T) logits at tile top-lefts -> (H, W) argmax map, cropped to `shape`."""
    C, T = tile_logits.shape[1], tile_logits.shape[-1]
    H = max(int(positions[:, 0].max()) + T, shape[0])
    W = max(int(positions[:, 1].max()) + T, shape[1])
    canvas = np.zeros((C, H, W), np.float32)
    for (t, l), lg in zip(positions, tile_logits):
        canvas[:, t:t + T, l:l + T] = lg
    return canvas.argmax(0)[:shape[0], :shape[1]].astype(np.uint8)


def _pixel_labels(zone_tile: np.ndarray, pad_valid: np.ndarray) -> np.ndarray:
    """4-class ids per pixel, IGNORE on padding."""
    cls, ok = zones_to_class(zone_tile, with_na=True)
    return np.where(ok & pad_valid, cls, IGNORE).astype(np.uint8)


def _class_weights(labels) -> np.ndarray:
    """Balanced class weights from per-tile label counts (no stacked copy of all labels)."""
    counts = np.zeros(N_CLASSES4, np.int64)
    for lab in labels:
        counts += np.bincount(lab[lab != IGNORE], minlength=N_CLASSES4)
    counts = counts.astype(np.float64)
    return counts.sum() / (N_CLASSES4 * np.maximum(counts, 1))


def _to_guide(raw_uint8: "torch.Tensor") -> "torch.Tensor":
    from .models.anyup_loader import guide_image
    return guide_image(raw_uint8)


class TrainSet:
    """Training tiles as tensors, built once and reused (run_heads: once for all its heads).

    Speed-up 2026-09-25: the per-step numpy stack + fp32 conversion + host->GPU
    copy of 8 tiles took ~40-50 ms of a ~70 ms step (measured on a desktop CPU),
    while the decoder itself needs ~2-5 ms on an A100. Here the tiles live on the
    GPU when they fit (fp16 features, uint8 image and labels), else in host
    memory with only the fp16 batch copied per step. `batch(idx)` returns the
    same values as the old per-step path (fp16 -> fp32 is exact).
    """

    def __init__(self, feats: list, raws: list, labels: list, device: str,
                 extra: list | None = None, reserve_gb: float = 20.0):
        import torch

        n, (g1, g2, d), t = len(feats), feats[0].shape, raws[0].shape[-1]
        need = n * (g1 * g2 * d * 2 + 2 * t * t)
        fits = device == "cuda" and torch.cuda.mem_get_info()[0] > need + reserve_gb * 1e9
        self.device, self.store = device, (device if device != "cuda" or fits else "cpu")
        self.feats = torch.empty((n, g1, g2, d), dtype=torch.float16, device=self.store)
        self.raw = torch.empty((n, t, t), dtype=torch.uint8, device=self.store)
        self.lab = torch.empty((n, t, t), dtype=torch.uint8, device=self.store)
        for i in range(n):
            self.feats[i] = torch.from_numpy(np.ascontiguousarray(feats[i], dtype=np.float16))
            self.raw[i] = torch.from_numpy(np.ascontiguousarray(raws[i]))
            self.lab[i] = torch.from_numpy(np.ascontiguousarray(labels[i]))
        self.extra = (None if extra is None else
                      torch.from_numpy(np.stack(extra).astype(np.float32)).to(device))
        print(f"    training tiles held on {self.store} ({need / 1e9:.1f} GB)")

    def batch(self, idx):
        import torch

        ix = torch.as_tensor(np.asarray(idx), device=self.store)
        f = self.feats.index_select(0, ix).to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
        raw = self.raw.index_select(0, ix).to(self.device, non_blocking=True)[:, None]
        y = self.lab.index_select(0, ix).to(self.device, non_blocking=True).long()
        ex = None if self.extra is None else self.extra.index_select(0, ix.to(self.extra.device))
        return f, raw, y, ex


# ------------------------------------------------------------------ row 1

def row1_train(data: dict, device: str):
    from .probe import train_linear_probe
    return train_linear_probe(data["train_X"], data["train_y"], balanced=True, n_classes=4,
                              epochs=30, lr=1e-3, device=device, verbose=False)


def row1_maps(probe, data: dict) -> dict:
    from .fronts import probe_zone_map
    W = probe.weight.detach().float().cpu().numpy()
    b = probe.bias.detach().float().cpu().numpy()
    return {stem: probe_zone_map(v["features"], v["positions"], v["shape"], W, b)
            for stem, v in data["eval"].items()}


# ------------------------------------------------------------------ row 2

def row2_train(data: dict, anyup, device: str, n_tiles: int = 600, pixels_per_tile: int = 400,
               seed: int = 0):
    """Linear probe on AnyUp-upsampled features, trained on sampled pixels."""
    import torch
    from .probe import train_linear_probe

    rng = np.random.default_rng(seed)
    tiles = data["train_tiles"]
    idx = rng.choice(len(tiles), min(n_tiles, len(tiles)), replace=False)
    xs, ys = [], []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in idx:
            f, raw, zt, pv = tiles[i]
            lab = _pixel_labels(zt, pv)
            ok = np.argwhere(lab != IGNORE)
            if len(ok) == 0:
                continue
            pick = ok[rng.choice(len(ok), min(pixels_per_tile, len(ok)), replace=False)]
            ft = torch.from_numpy(f.astype(np.float32)).permute(2, 0, 1)[None].to(device)
            guide = _to_guide(torch.from_numpy(raw)[None, None].to(device))
            up = anyup(guide, ft)[0]                                # (D, T, T)
            xs.append(up[:, pick[:, 0], pick[:, 1]].float().T.cpu().numpy().astype(np.float16))
            ys.append(lab[pick[:, 0], pick[:, 1]])
    X, y = np.concatenate(xs), np.concatenate(ys)
    print(f"row 2: probe on {len(y):,} upsampled pixels from {len(idx)} tiles")
    return train_linear_probe(X, y, balanced=True, n_classes=4, epochs=30, lr=1e-3,
                              device=device, verbose=False)


def row2_maps(probe, data: dict, anyup, device: str) -> dict:
    import torch
    from .models.anyup_loader import upsample_with_value

    W = probe.weight.detach().float().to(device)
    b = probe.bias.detach().float().to(device)
    out = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for stem, v in data["eval"].items():
            logits = []
            for t in range(len(v["features"])):
                ft = torch.from_numpy(v["features"][t].astype(np.float32)).permute(2, 0, 1)[None].to(device)
                lr = torch.einsum("cd,bdhw->bchw", W, ft) + b.view(1, -1, 1, 1)   # 4 logits on the grid
                guide = _to_guide(torch.from_numpy(v["raw"][t])[None, None].to(device))
                logits.append(upsample_with_value(anyup, guide, ft, lr)[0].float().cpu().numpy())
            out[stem] = _stitch(np.stack(logits), v["positions"], v["shape"])
    return out


# ------------------------------------------------------------------ rows 3-6

def _training_set(tiles: list, device: str) -> tuple:
    """(class weights, TrainSet) of the training tiles.

    Depends only on the tiles and the device, not on the row or seed, so
    run_heads builds it once and passes it to every head it trains.
    """
    labels = [_pixel_labels(zt, pv) for _, _, zt, pv in tiles]
    return _class_weights(labels), TrainSet([t[0] for t in tiles], [t[1] for t in tiles], labels, device)


def _train_head(head, data: dict, device: str, anyup=None, steps: int = 4000, batch: int = 8,
                lr: float = 1e-3, seed: int = 0, log_every: int = 1000, prepared: tuple | None = None):
    """`prepared`: `_training_set(data["train_tiles"], device)`, shared across heads; None builds it here."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tiles = data["train_tiles"]
    cw, ts = _training_set(tiles, device) if prepared is None else prepared
    w = torch.tensor(cw, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0, running, n_run = time.perf_counter(), torch.zeros((), device=device), 0
    for it in range(1, steps + 1):
        head.train()
        idx = rng.choice(len(tiles), batch, replace=len(tiles) < batch)
        f, raw, y, _ = ts.batch(idx)
        flip = rng.random() < 0.5
        if flip:
            f, raw, y = f.flip(-1), raw.flip(-1), y.flip(-1)
        img = raw.float() / 255.0
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = head(f, img) if anyup is None else head(f, img, _to_guide(raw), anyup)
            loss = loss_fn(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        running += loss.detach(); n_run += 1          # read back only at log lines (no per-step sync)
        if it % log_every == 0 or it == steps:
            print(f"    step {it:>5}  loss {float(running) / n_run:.4f}  {(time.perf_counter() - t0) / 60:.1f} min")
            running, n_run = torch.zeros((), device=device), 0
    return head


def _head_maps(head, data: dict, device: str, anyup=None, batch: int = 8) -> dict:
    import torch

    head.eval()
    out = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for stem, v in data["eval"].items():
            logits = []
            for i in range(0, len(v["features"]), batch):
                f = torch.from_numpy(v["features"][i:i + batch]).to(device).float().permute(0, 3, 1, 2)
                raw = torch.from_numpy(v["raw"][i:i + batch])[:, None].to(device)
                img = raw.float() / 255.0
                o = head(f, img) if anyup is None else head(f, img, _to_guide(raw), anyup)
                logits.append(o.float().cpu().numpy())
            out[stem] = _stitch(np.concatenate(logits), v["positions"], v["shape"])
    return out


# ------------------------------------------------------------------ driver

def _tag(run: str, row: int, seed: int = 0) -> str:
    return f"{run}__row{row}" + (f"__s{seed}" if seed else "")


def run_heads(encoder: str, norm: str, layer: int | tuple[int, ...], df: pd.DataFrame,
              data_root: Path, res_dir: Path, token: str | None, anyup, rows=(1, 2, 3, 4),
              split: str = "val", tile: int = 512, max_train_tiles: int = 4000, steps: int = 4000,
              force: bool = False, head_batch: dict | None = None,
              train_frac: float = 0.21, seeds: tuple[int, ...] = (0,),
              legacy_tile_sampler: bool = False) -> pd.DataFrame:
    """Train (split="val") or load (split="test") the heads, score on `split`.

    Writes `heads_<split>_mde__<run>__row<k>[__s<seed>].csv` (+ images, zones4,
    paper-style) and, when training, `heads__<run>__row<k>[__s<seed>].pt`.
    `seeds` (rows 3-6 only) re-trains the same head from other initialisations
    and tile orders; seed 0 keeps the original file names. `head_batch` entries
    override the defaults per row. `legacy_tile_sampler=True` keeps the tilted
    training-tile sample of the runs reported up to 2026-09-25 (see
    features._reservoir_slot). Loading (test) extracts the evaluation split only.
    """
    import torch

    from .data.tiling import TileSpec
    from .features import extract_in_memory
    from .models.decoder import AnyUpDecoder, GridDecoder, PixelHead, load_head_state
    from .models.encoders import ENCODERS, FrozenEncoder
    from .runs import ensure_packages, run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_tag = "+".join(map(str, layer)) if isinstance(layer, (tuple, list)) else str(layer)
    run = f"{run_name(encoder, norm, tile)}__L{layer_tag}"
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    head_batch = {3: 8, 4: 4, 5: 8, 6: 8} | (head_batch or {})
    training = split == "val"
    jobs = [(r, s) for r in rows for s in seeds
            if force or not (res_dir / f"heads_{split}_mde__{_tag(run, r, s)}.csv").exists()]
    if not jobs:
        print(f"skip {run}: all requested rows/seeds already scored on {split}")
        return pd.DataFrame()
    todo = sorted({r for r, _ in jobs})
    if 5 in todo and not isinstance(layer, (tuple, list)):
        raise ValueError("row 5 (grid-decoder-4L) needs layer=(l1, l2, l3, l4)")
    if any(r in (1, 2) and s for r, s in jobs):
        raise ValueError("seeds apply to the learned heads (rows 3-6) only")
    print(f"\n===== heads {run} on {split}: rows {todo}, seeds {sorted({s for _, s in jobs})} =====")

    spec = ENCODERS[encoder]
    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    t0 = time.perf_counter()
    try:
        # loading needs no training images: only the split being scored is extracted
        data = extract_in_memory(df, data_root, enc, TileSpec(size=tile, overlap=0.0), norm,
                                 splits=("train", split) if training else (split,), with_na=True,
                                 eval_split=split, layer=layer, keep_tiles=True,
                                 max_train_tiles=max_train_tiles, train_frac=train_frac,
                                 legacy_tile_sampler=legacy_tile_sampler)
    finally:
        del enc
        gc.collect()
        torch.cuda.empty_cache()
    if training:
        D = data["train_X"].shape[1]
    elif data["eval"]:
        D = next(iter(data["eval"].values()))["features"].shape[-1]
    else:
        raise ValueError(f"no {split} images in df")
    try:
        import psutil
        ram = f" | RAM used {psutil.virtual_memory().used / 1e9:.0f} of {psutil.virtual_memory().total / 1e9:.0f} GB"
    except Exception:
        ram = ""
    print(f"features ready in {(time.perf_counter() - t0) / 60:.1f} min | {len(data['train_tiles'])} train tiles kept{ram}")
    groupings = FINAL_GROUPINGS if split == "test" else GROUPINGS

    summary, prepared = [], None             # training set: built at the first learned head, then shared
    for r, seed in jobs:
        t1 = time.perf_counter()
        tag = _tag(run, r, seed)
        wpath = res_dir / f"heads__{tag}.pt"
        if r == 1:
            if training:
                head = row1_train(data, device); torch.save(head.state_dict(), wpath)
            else:
                head = torch.nn.Linear(D, 4).to(device); head.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
            maps = row1_maps(head, data)
        elif r == 2:
            if training:
                head = row2_train(data, anyup, device); torch.save(head.state_dict(), wpath)
            else:
                head = torch.nn.Linear(D, 4).to(device); head.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
            maps = row2_maps(head, data, anyup, device)
        else:
            cls = {3: GridDecoder, 5: GridDecoder, 4: PixelHead, 6: AnyUpDecoder}[r]
            au = None if r in (3, 5) else anyup
            if training:
                if prepared is None:
                    prepared = _training_set(data["train_tiles"], device)
                torch.manual_seed(seed)              # the seed also sets the initial weights
                head = _train_head(cls(D), data, device, anyup=au, steps=steps, batch=head_batch[r], seed=seed,
                                   prepared=prepared)
                torch.save(head.state_dict(), wpath)
            else:                                    # also loads heads saved with all five stem levels
                head = load_head_state(cls(D), torch.load(wpath, map_location="cpu", weights_only=True)).to(device)
            maps = _head_maps(head, data, device, anyup=au, batch=head_batch[r])
        records, mde, zones = fronts_from_zone_maps(maps, data_root, tag, groupings=groupings)
        paper = paper_zone_table(records, groupings=groupings)
        records.to_csv(res_dir / f"heads_{split}_images__{tag}.csv", index=False)
        zones.to_csv(res_dir / f"heads_{split}_zones4__{tag}.csv", index=False)
        paper.to_csv(res_dir / f"heads_{split}_paper__{tag}.csv", index=False)
        mde.to_csv(res_dir / f"heads_{split}_mde__{tag}.csv", index=False)
        a = mde[mde.grouping == "all"].iloc[0]
        z = zones[(zones.grouping == groupings[0]) & (zones.group == "all")].iloc[0]
        n_par = sum(p.numel() for p in head.parameters())
        summary.append({"run": run, "row": r, "seed": seed, "head": ROWS[r], "params": n_par, "MDE_m": a.MDE_m,
                        "no_front": a.no_front, "images": a.images, "mIoU4_pixel": z.mIoU,
                        "minutes": (time.perf_counter() - t1) / 60})
        print(f"  row {r} {ROWS[r]:<14}{f' s{seed}' if seed else '   '} {n_par/1e3:7.1f} k params | MDE {a.MDE_m:6.0f} m "
              f"({a.no_front} no front) | 4-class pixel mIoU {z.mIoU:.3f} | {(time.perf_counter() - t1) / 60:.1f} min")
        del head, maps
        gc.collect()
        torch.cuda.empty_cache()
    del data, prepared
    gc.collect()
    torch.cuda.empty_cache()
    out = pd.DataFrame(summary)
    spath = res_dir / f"heads_{split}_summary__{run}.csv"
    merged = out
    if spath.exists():                       # keep rows/seeds scored in an earlier session
        old = pd.read_csv(spath)
        if "seed" not in old:
            old["seed"] = 0
        done = set(zip(out.row, out.seed))
        keep = [(r, s) not in done for r, s in zip(old.row, old.seed)]
        merged = pd.concat([old[keep], out], ignore_index=True).sort_values(["row", "seed"])
    merged.to_csv(spath, index=False)
    return out                               # this call's rows only


def summary_table(res_dir: Path, split: str = "val") -> pd.DataFrame:
    """One line per (run, head) rebuilt from the per-row CSVs on disk.

    Complete even when a run's rows were scored in different sessions (the
    summary CSV only ever held one session's rows before the merge above);
    `params` / `minutes` are joined from the summaries where present.
    """
    import glob

    res_dir = Path(res_dir)
    # first grouping, as the run code. Read as @g0 in the query below, which ruff cannot see.
    g0 = (FINAL_GROUPINGS if split == "test" else GROUPINGS)[0]   # noqa: F841
    rows = []
    for f in sorted(glob.glob(str(res_dir / f"heads_{split}_mde__*__row*.csv"))):
        tag = Path(f).stem.split(f"heads_{split}_mde__", 1)[1]
        run, rest = tag.rsplit("__row", 1)
        r, _, s = rest.partition("__s")
        a = pd.read_csv(f).query("grouping == 'all'").iloc[0]
        z = pd.read_csv(res_dir / f"heads_{split}_zones4__{tag}.csv")
        z = z.query("grouping == @g0 and group == 'all'").iloc[0]
        rows.append({"run": run, "row": int(r), "seed": int(s or 0), "head": ROWS[int(r)], "MDE_m": a.MDE_m,
                     "no_front": a.no_front, "images": a.images, "mIoU4_pixel": z.mIoU})
    out = pd.DataFrame(rows)
    summ = [pd.read_csv(f) for f in sorted(glob.glob(str(res_dir / f"heads_{split}_summary__*.csv")))]
    if summ and len(out):
        sm = pd.concat(summ, ignore_index=True)
        if "seed" not in sm:
            sm["seed"] = 0
        sm["seed"] = sm["seed"].fillna(0).astype(int)
        out = out.merge(sm[["run", "row", "seed", "params", "minutes"]], on=["run", "row", "seed"], how="left")
    return out.sort_values(["run", "row", "seed"]).reset_index(drop=True)
