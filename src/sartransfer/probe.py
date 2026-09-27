"""Linear probe on frozen patch features -- the feature-ceiling measurement.

A linear probe is deliberate, not a shortcut. The question is *how much zone
structure the frozen features already encode*, so the head must add as little
capacity as possible: 1024 -> 3 is about 3 k parameters. Anything deeper would
measure the head instead of the encoder.

Zone patch classes (see `data.targets`): 0 stone, 1 glacier, 2 ocean.
Patches marked invalid -- the dataset's no-information zone and tile padding --
are excluded from both training and metrics, as the CaFFe paper requires.

Measured class balance on the training split, 2026-09-22:
glacier 64.2 %, ocean 23.4 %, stone 12.5 %. Both unweighted and balanced
variants are reported, because an unweighted probe flatters the dominant class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data.targets import N_CLASSES, class_names


def train_linear_probe(X: np.ndarray, y: np.ndarray, *, balanced: bool = False,
                       n_classes: int = N_CLASSES,
                       epochs: int = 30, lr: float = 1e-3, batch: int = 65536,
                       device: str = "cuda", seed: int = 0, verbose: bool = True) -> nn.Linear:
    """Fit a single linear layer on patch vectors.

    `balanced` weights the loss by inverse class frequency.
    """
    torch.manual_seed(seed)
    n, dim = X.shape
    probe = nn.Linear(dim, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if balanced:
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        if verbose:
            print("class weights:", dict(zip(class_names(n_classes), w.round(3))))
    else:
        weight = None
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.int64))
    # Speed-up (2026-09-24): put the whole training set on the GPU once instead of
    # gathering every batch on the CPU. ~2 M x 1280 patches fit easily in 80 GB.
    # The shuffle order is still drawn on the CPU, so batches are identical to before.
    on_gpu = str(device).startswith("cuda")
    if on_gpu:
        try:
            Xt, yt = Xt.to(device), yt.to(device)
        except torch.cuda.OutOfMemoryError:
            on_gpu = False
            torch.cuda.empty_cache()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if on_gpu:
                idx = idx.to(device)
            xb = Xt[idx].to(device, non_blocking=True).float()
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3}  loss {total/n:.4f}")
    return probe


def confusion(probe: nn.Linear, feat_dir: Path, manifest: pd.DataFrame, split: str,
              device: str = "cuda", group_by: str | None = None) -> dict[str, np.ndarray]:
    """Accumulate confusion matrices over a split, streaming file by file.

    `group_by` names a manifest column (e.g. "band" or "glacier"); the result
    then holds one matrix per group plus "all".
    """
    feat_dir = Path(feat_dir)
    rows = manifest[manifest["split"] == split]
    n = probe.out_features
    mats: dict[str, np.ndarray] = {"all": np.zeros((n, n), dtype=np.int64)}

    probe.eval()
    with torch.inference_mode():
        for r in rows.itertuples():
            with np.load(feat_dir / f"{r.stem}.npz") as z:
                f, lab, ok = z["features"], z["labels"], z["valid"]
            if not ok.any():
                continue
            xb = torch.from_numpy(f[ok].astype(np.float32)).to(device)
            pred = probe(xb).argmax(1).cpu().numpy()
            true = lab[ok].astype(np.int64)
            m = np.bincount(true * n + pred, minlength=n * n).reshape(n, n)
            mats["all"] += m
            if group_by:
                key = str(getattr(r, group_by))
                mats.setdefault(key, np.zeros_like(m))
                mats[key] += m
    return mats


def iou_from_confusion(m: np.ndarray) -> dict[str, float]:
    """Per-class IoU and their mean. Rows are truth, columns are prediction."""
    tp = np.diag(m).astype(np.float64)
    fp = m.sum(axis=0) - tp
    fn = m.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    out = {f"IoU_{name}": float(v) for name, v in zip(class_names(m.shape[0]), iou)}
    out["mIoU"] = float(np.nanmean(iou))
    out["accuracy"] = float(tp.sum() / max(m.sum(), 1))
    out["n_patches"] = int(m.sum())
    return out


def report(mats: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per group, sorted with 'all' first."""
    rows = [{"group": k, **iou_from_confusion(v)} for k, v in mats.items()]
    df = pd.DataFrame(rows)
    df["_o"] = (df["group"] != "all").astype(int)
    return df.sort_values(["_o", "group"]).drop(columns="_o").reset_index(drop=True)


def majority_baseline(manifest: pd.DataFrame, feat_dir: Path, split: str,
                      train_counts: np.ndarray) -> dict[str, float]:
    """What you get by predicting the most common training class everywhere.

    The floor any real result has to clear.
    """
    feat_dir = Path(feat_dir)
    cls = int(train_counts.argmax())
    n = len(train_counts)
    m = np.zeros((n, n), dtype=np.int64)
    for stem in manifest.loc[manifest["split"] == split, "stem"]:
        with np.load(feat_dir / f"{stem}.npz") as z:
            lab, ok = z["labels"], z["valid"]
        counts = np.bincount(lab[ok], minlength=n)
        m[:, cls] += counts
    out = iou_from_confusion(m)
    out["predicts"] = class_names(n)[cls]
    return out


def confusion_from_arrays(probe: nn.Linear, items: dict, manifest: pd.DataFrame, device: str,
                          groupings=("band",), feat_key: str = "features",
                          lab_key: str = "labels", ok_key: str = "valid") -> dict[str, dict[str, np.ndarray]]:
    """Like `confusion`, but over in-memory {stem: arrays} and several groupings at once."""
    n = probe.out_features
    meta = manifest.set_index("stem")
    mats = {g: {"all": np.zeros((n, n), dtype=np.int64)} for g in groupings}
    probe.eval()
    with torch.inference_mode():
        for stem, v in items.items():
            ok = v[ok_key]
            if not ok.any():
                continue
            xb = torch.from_numpy(v[feat_key][ok].astype(np.float32)).to(device)
            pred = probe(xb).argmax(1).cpu().numpy()
            true = v[lab_key][ok].astype(np.int64)
            m = np.bincount(true * n + pred, minlength=n * n).reshape(n, n)
            for g in groupings:
                mats[g]["all"] += m
                key = str(meta.loc[stem, g])
                mats[g].setdefault(key, np.zeros_like(m))
                mats[g][key] += m
    return mats
