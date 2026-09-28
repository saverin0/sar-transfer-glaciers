"""Train the from-scratch U-Net reference and score it like the frozen-encoder probes.

Fairness rules, so the comparison with the linear probes holds:

- same training glaciers (Crane, DBE, JAC, Jorum), same validation glacier (SI),
  test glaciers never loaded
- same classes (stone, glacier, ocean); NA zone and padding ignored
- class-balanced loss, like the "balanced" probe
- fixed training length; the validation curve is logged for information only,
  the final model is scored (the probes got no checkpoint selection on SI either)
- scored twice: patch level (majority over 16x16 px, the probes' resolution, same
  CSV schema so 04_compare lists it next to them) and pixel level (the U-Net's
  native resolution; not available for the probes)
- input normalisation: the official CaFFe fixed statistics ("caffe"), as in the
  CaFFe baseline
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data.inventory import parse_name
from .data.targets import IGNORE, N_CLASSES, class_names, patch_labels, zones_to_class
from .data.tiling import normalise_image
from .features import split_of, zone_path_for
from .probe import report

Image.MAX_IMAGE_PIXELS = None
GROUPINGS = ("band", "satellite", "resolution_m", "sensor_res")


# ----------------------------------------------------------------------- data

def load_split_images(df: pd.DataFrame, data_root: Path, split: str,
                      norm: str = "caffe", limit: int | None = None,
                      with_na: bool = False) -> list[dict]:
    """Normalised images + class maps for one split, in RAM.

    Default classes 0/1/2 (stone, glacier, ocean), 255 = ignore (NA).
    `with_na=True`: 0/1/2/3 (na, stone, glacier, ocean), as the official model.
    """
    rows = []
    sel = df[df["path"].map(lambda p: split_of(Path(p).stem.split("_")[0])) == split]
    for p in list(sel["path"])[:limit]:
        p = Path(p)
        img = np.asarray(Image.open(p))
        cls, valid = zones_to_class(np.asarray(Image.open(zone_path_for(p, data_root))), with_na=with_na)
        meta = parse_name(p.name)
        rows.append({
            "stem": p.stem, "x": normalise_image(img, norm).astype(np.float32),
            "y": np.where(valid, cls, IGNORE).astype(np.uint8),
            "glacier": meta["glacier"], "satellite": meta["satellite"], "band": meta["band"],
            "resolution_m": meta["resolution_m"],
            "sensor_res": f"{meta['satellite']}@{int(meta['resolution_m'])}m",
        })
    return rows


class CropSampler:
    """Random crops from the training images, weighted by image area.

    `plan(n)` draws positions and flips with the numpy RNG and checks validity on
    the CPU copy of the labels. `batch(n)` builds the crops on the CPU; with
    `device="cuda"` the images live on the GPU (speed-up 2026-09-24) and crops
    are cut there from the SAME plan, so batches are identical either way.
    """

    def __init__(self, images: list[dict], crop: int = 256, min_valid: float = 0.1, seed: int = 0,
                 device: str | None = None):
        self.images, self.crop, self.min_valid = images, crop, min_valid
        area = np.array([im["x"].size for im in images], dtype=np.float64)
        self.p = area / area.sum()
        self.rng = np.random.default_rng(seed)
        self.device = device
        if device is not None:
            import torch
            self._x = [torch.from_numpy(im["x"]).to(device) for im in images]
            self._y = [torch.from_numpy(im["y"]).to(device) for im in images]

    def plan(self, n: int) -> list[tuple[int, int, int, bool, bool]]:
        out, c = [], self.crop
        while len(out) < n:
            k = self.rng.choice(len(self.images), p=self.p)
            im = self.images[k]
            h, w = im["x"].shape
            t, l = self.rng.integers(0, h - c + 1), self.rng.integers(0, w - c + 1)
            if (im["y"][t:t + c, l:l + c] != IGNORE).mean() < self.min_valid:
                continue
            fw, fh = self.rng.random() < 0.5, self.rng.random() < 0.5
            out.append((int(k), int(t), int(l), bool(fw), bool(fh)))
        return out

    def batch(self, n: int):
        c, plan = self.crop, self.plan(n)
        if self.device is None:
            xs, ys = [], []
            for k, t, l, fw, fh in plan:
                x = self.images[k]["x"][t:t + c, l:l + c]
                y = self.images[k]["y"][t:t + c, l:l + c]
                if fw:
                    x, y = x[:, ::-1], y[:, ::-1]
                if fh:
                    x, y = x[::-1], y[::-1]
                xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
            return np.stack(xs)[:, None], np.stack(ys).astype(np.int64)
        import torch
        xs, ys = [], []
        for k, t, l, fw, fh in plan:
            x, y = self._x[k][t:t + c, l:l + c], self._y[k][t:t + c, l:l + c]
            dims = [d for d, f in ((1, fw), (0, fh)) if f]
            if dims:
                x, y = torch.flip(x, dims), torch.flip(y, dims)
            xs.append(x); ys.append(y)
        return torch.stack(xs)[:, None], torch.stack(ys).long()


def class_weights(images: list[dict], n: int = N_CLASSES) -> np.ndarray:
    counts = np.zeros(n, dtype=np.float64)
    for im in images:
        counts += np.bincount(im["y"][im["y"] != IGNORE].ravel(), minlength=n)
    return counts.sum() / (n * np.maximum(counts, 1))


# ------------------------------------------------------------------- inference

def _windows(H: int, W: int, tile: int, stride: int) -> list[tuple[int, int]]:
    tops = list(range(0, max(H - tile, 0) + 1, stride))
    lefts = list(range(0, max(W - tile, 0) + 1, stride))
    if tops[-1] != H - tile:
        tops.append(H - tile)
    if lefts[-1] != W - tile:
        lefts.append(W - tile)
    return [(t, l) for t in tops for l in lefts]


def predict(model, x: np.ndarray, device: str, tile: int = 512, stride: int = 384,
            batch: int = 16) -> np.ndarray:
    """Sliding-window class map for one whole image (averaged logits).

    Speed-up 2026-09-24: windows run `batch` at a time and logits are summed on
    the device -- same windows and same averaging as the one-by-one version.
    """
    import torch

    h, w = x.shape
    H = max(tile, int(np.ceil(h / 16)) * 16)
    W = max(tile, int(np.ceil(w / 16)) * 16)
    xp = torch.zeros((H, W), dtype=torch.float32, device=device)
    xp[:h, :w] = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    wins = _windows(H, W, tile, stride)
    logits, hits = None, torch.zeros((H, W), dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            inp = torch.stack([xp[t:t + tile, l:l + tile] for t, l in chunk])[:, None]
            out = model(inp).float()
            if logits is None:
                logits = torch.zeros((out.shape[1], H, W), dtype=torch.float32, device=device)
            for (t, l), o in zip(chunk, out):
                logits[:, t:t + tile, l:l + tile] += o
                hits[t:t + tile, l:l + tile] += 1
    return (logits / hits.clamp_min(1)).argmax(0)[:h, :w].to(torch.uint8).cpu().numpy()


def _confusion(true: np.ndarray, pred: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    k = (true != IGNORE)
    return np.bincount(true[k].astype(np.int64) * n + pred[k].astype(np.int64),
                       minlength=n * n).reshape(n, n)


def _patch_level(y: np.ndarray, pred: np.ndarray, patch: int = 16,
                 n: int = N_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """Majority label per 16 px patch for truth and prediction, over valid pixels."""
    h, w = y.shape
    H, W = int(np.ceil(h / patch)) * patch, int(np.ceil(w / patch)) * patch
    yp = np.full((H, W), IGNORE, np.uint8); yp[:h, :w] = y
    pp = np.zeros((H, W), np.uint8); pp[:h, :w] = pred
    valid = yp != IGNORE
    t_lab, t_keep = patch_labels(np.where(valid, yp, 0), valid, patch, n_classes=n)
    p_lab, _ = patch_labels(pp, valid, patch, n_classes=n)
    t_lab = np.where(t_keep, t_lab, IGNORE)
    return t_lab, p_lab


def evaluate(model, images: list[dict], device: str,
             preds: dict | None = None) -> dict[str, dict[str, dict]]:
    """Confusion matrices per grouping, at 'patch' and 'pixel' level.

    `preds` {stem: class map from `predict`} reuses maps predicted already
    (run_final_unet needs them twice); None predicts each image here.
    """
    n = model.head.out_channels
    out = {lvl: {g: {"all": np.zeros((n, n), np.int64)} for g in GROUPINGS}
           for lvl in ("patch", "pixel")}
    for im in images:
        pred = predict(model, im["x"], device) if preds is None else preds[im["stem"]]
        t_lab, p_lab = _patch_level(im["y"], pred, n=n)
        mats = {"pixel": _confusion(im["y"], pred, n), "patch": _confusion(t_lab, p_lab, n)}
        for lvl, m in mats.items():
            for g in GROUPINGS:
                key = str(im[g])
                out[lvl][g]["all"] += m if g == GROUPINGS[0] else 0
                out[lvl][g].setdefault(key, np.zeros_like(m))
                out[lvl][g][key] += m
    for lvl in out:                                   # "all" is the same for every grouping
        for g in GROUPINGS[1:]:
            out[lvl][g]["all"] = out[lvl][GROUPINGS[0]]["all"].copy()
    return out


def to_table(conf: dict[str, dict], run: str, head: str = "balanced") -> pd.DataFrame:
    """Same schema as the probe CSVs, so 04_compare can read it."""
    tables = []
    for g, mats in conf.items():
        r = report(mats)
        r.insert(0, "grouping", g); r.insert(0, "probe", head)
        tables.append(r)
    res = pd.concat(tables, ignore_index=True)
    res.insert(0, "run", run)
    return res


# -------------------------------------------------------------------- training

def train_unet(train_imgs: list[dict], val_imgs: list[dict], iters: int = 12000,
               batch: int = 16, lr: float = 1e-3, crop: int = 256, base: int = 32,
               log_every: int = 1000, seed: int = 0, device: str | None = None,
               n_classes: int = N_CLASSES, fast: bool = True):
    """Fixed-length training; returns (model, log DataFrame).

    Speed-ups (2026-09-24): training images on the GPU, crops cut there from the
    same random plan; loss summed on the GPU (no CPU sync every step). With
    `fast=True` also cuDNN autotuning + channels-last memory layout, which can
    change results in the last floating-point digits (GPU training is not
    bit-reproducible anyway); `fast=False` switches those two off.
    """
    import torch
    import torch.nn as nn

    from .models.unet import UNet
    from .probe import iou_from_confusion

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = UNet(in_ch=1, n_classes=n_classes, base=base).to(device)
    use_fast = fast and device == "cuda"
    if use_fast:
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    w = torch.tensor(class_weights(train_imgs, n_classes), dtype=torch.float32, device=device)
    print("class weights:", dict(zip(class_names(n_classes), w.cpu().numpy().round(3))))
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    sampler = CropSampler(train_imgs, crop=crop, seed=seed,
                          device=device if device == "cuda" else None)

    log, t0 = [], time.perf_counter()
    running, n_run = torch.zeros((), device=device), 0
    for it in range(1, iters + 1):
        model.train()
        xb, yb = sampler.batch(batch)
        if isinstance(xb, np.ndarray):
            xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
        if use_fast:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = loss_fn(model(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        running += loss.detach(); n_run += 1
        if it % log_every == 0 or it == iters:
            val = evaluate(model, val_imgs, device)["pixel"]["band"]["all"] if val_imgs else None
            miou = iou_from_confusion(val)["mIoU"] if val is not None else float("nan")
            mean_loss = running.item() / n_run          # the last window can be shorter
            log.append({"iter": it, "loss": mean_loss, "val_pixel_mIoU": miou,
                        "minutes": (time.perf_counter() - t0) / 60})
            print(f"  iter {it:>6}  loss {mean_loss:.4f}  SI pixel mIoU {miou:.4f}  "
                  f"{(time.perf_counter() - t0) / 60:.1f} min")
            running.zero_(); n_run = 0
    return model, pd.DataFrame(log)
