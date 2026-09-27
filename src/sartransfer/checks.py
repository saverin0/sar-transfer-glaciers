"""One-off measurements run from the notebooks. No training; SI (validation) only."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed 2026-09-25 BEFORE the numbers exist (code review): re-run the AnyUp rows
# at full precision on SI only if the as-run bf16 AnyUp
# changes the class of more than this share of valid pixels in row 2, 4 or 6 ...
MAX_CHANGED_PCT = 1.0
# ... or its feature error exceeds this fraction of AnyUp's own effect
# (fp32 AnyUp output vs plain bilinear upsampling of the same features).
MAX_ERR_OVER_EFFECT = 0.10


def _si_pick(df: pd.DataFrame, n_images: int) -> list[Path]:
    """Up to `n_images` SI images, spread evenly over sensor@resolution groups and dates."""
    from .data.inventory import parse_name
    from .features import split_of

    groups: dict[str, list[str]] = {}
    for p in sorted(map(str, df["path"])):
        m = parse_name(Path(p).name)
        glacier = m.get("glacier") or Path(p).stem.split("_")[0]
        if split_of(glacier) == "val":
            groups.setdefault(f"{m.get('satellite')}@{m.get('resolution_m')}", []).append(p)
    if not groups:
        raise ValueError("no SI (validation) images in df")
    k = -(-n_images // len(groups))
    picked = []
    for key in sorted(groups):
        v = groups[key]
        idx = np.unique(np.linspace(0, len(v) - 1, min(k, len(v))).round().astype(int))
        picked += [Path(v[j]) for j in idx]
    return picked[:n_images]


def anyup_precision_check(encoder: str, norm: str, layer: int, df: pd.DataFrame, data_root: Path,
                          res_dir: Path, token: str | None, anyup, n_images: int = 12,
                          tile: int = 512, device: str = "cuda") -> pd.DataFrame:
    """How much does running AnyUp under bf16 autocast change what the heads produce?

    Code review 2026-09-25: AnyUp builds its pixel coordinates in the dtype of its
    image encoder -- bf16 under the autocast every head ran with -- and computes
    the RoPE angles in bf16. This loads the SAVED heads of rows 2, 4 and 6 (no
    training) and runs one tile per SI image three ways:

      fp32                   autocast off
      bf16 (as run)          autocast bf16, exactly as every reported run
      bf16 + fp32 positions  autocast bf16, coordinates and RoPE angles in fp32

    Against fp32, on the tile's valid pixels: relative error of the upsampled
    full features, and the share of pixels whose class changes in rows 2, 4, 6.
    Scale reference: how far fp32 AnyUp is from plain bilinear upsampling.
    Measures inference only; whether bf16 also changed what the heads LEARNED
    is not measured. Writes `check_anyup_precision__<run>.csv`.
    """
    import torch
    import torch.nn.functional as F

    from .data.tiling import TileSpec
    from .features import _prepare
    from .heads import _to_guide
    from .models.anyup_loader import fp32_positions, upsample_with_value
    from .models.decoder import AnyUpDecoder, PixelHead, load_head_state
    from .models.encoders import ENCODERS, FrozenEncoder
    from .runs import ensure_packages, run_name

    run = f"{run_name(encoder, norm, tile)}__L{layer}"
    res_dir = Path(res_dir)
    paths = _si_pick(df, n_images)
    print(f"\n===== AnyUp precision check {run}: {len(paths)} SI images, one tile each =====")
    spec = ENCODERS[encoder]
    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    t0 = time.perf_counter()
    tiles = []
    try:
        for p in paths:
            it = _prepare(p, data_root, TileSpec(size=tile, overlap=0.0), norm, enc.patch, with_na=True)
            t = int(np.argmax(it["pad_valid"].reshape(len(it["pad_valid"]), -1).sum(1)))  # least padding
            f = enc.encode(it["x"][t:t + 1], layer=layer)                                    # (1, g, g, D)
            tiles.append((it["stem"], f[0], it["raw"][t], it["pad_valid"][t],
                          f"{it['satellite']}@{it['resolution_m']:g}m"))
    finally:
        del enc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    D = tiles[0][1].shape[-1]

    def saved(r: int, head):
        sd = torch.load(res_dir / f"heads__{run}__row{r}.pt", map_location="cpu", weights_only=True)
        return load_head_state(head, sd).to(device).eval()     # strict; drops only unused stem levels

    lin = saved(2, torch.nn.Linear(D, 4))
    h4, h6 = saved(4, PixelHead(D)), saved(6, AnyUpDecoder(D))
    W, b = lin.weight.detach().float(), lin.bias.detach().float()
    variants = {"fp32": (False, False), "bf16 (as run)": (True, False),
                "bf16 + fp32 positions": (True, True)}

    rows = []
    with torch.inference_mode():
        for stem, f, raw, pv, sensor in tiles:
            ft = torch.from_numpy(f.astype(np.float32)).permute(2, 0, 1)[None].to(device)
            rw = torch.from_numpy(raw)[None, None].to(device)
            img, guide = rw.float() / 255.0, _to_guide(rw)
            ok = torch.from_numpy(pv).to(device)
            outs = {}
            for name, (amp, pos) in variants.items():
                with torch.autocast(device, dtype=torch.bfloat16, enabled=amp), fp32_positions(pos):
                    full = upsample_with_value(anyup, guide, ft, ft)[0].float()           # (D, H, W)
                    lr = torch.einsum("cd,bdhw->bchw", W, ft) + b.view(1, -1, 1, 1)       # row 2 logits
                    c2 = upsample_with_value(anyup, guide, ft, lr)[0].float().argmax(0)
                    c4 = h4(ft, img, guide, anyup)[0].float().argmax(0)
                    c6 = h6(ft, img, guide, anyup)[0].float().argmax(0)
                outs[name] = (full[:, ok], c2[ok], c4[ok], c6[ok])                        # valid pixels
                del full
            ref = outs["fp32"]
            bil = F.interpolate(ft, size=tuple(raw.shape), mode="bilinear", align_corners=False)[0][:, ok]
            effect = float((ref[0] - bil).norm() / ref[0].norm())
            del bil
            for name, (fv, c2, c4, c6) in outs.items():
                rows.append({"run": run, "stem": stem, "sensor": sensor, "variant": name,
                             "feat_rel_err": float((fv - ref[0]).norm() / ref[0].norm()),
                             "anyup_effect": effect,
                             "row2_changed_pct": 100 * float((c2 != ref[1]).float().mean()),
                             "row4_changed_pct": 100 * float((c4 != ref[2]).float().mean()),
                             "row6_changed_pct": 100 * float((c6 != ref[3]).float().mean())})
            del outs
            if device == "cuda":
                torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(res_dir / f"check_anyup_precision__{run}.csv", index=False)
    cols = ["feat_rel_err", "anyup_effect", "row2_changed_pct", "row4_changed_pct", "row6_changed_pct"]
    summ = out.groupby("variant", sort=False)[cols].mean()
    print(summ.round(4).to_string())
    a = summ.loc["bf16 (as run)"]
    ratio = a.feat_rel_err / max(a.anyup_effect, 1e-12)
    worst = max(a.row2_changed_pct, a.row4_changed_pct, a.row6_changed_pct)
    matters = worst > MAX_CHANGED_PCT or ratio > MAX_ERR_OVER_EFFECT
    print(f"as run: feature error = {100 * ratio:.1f} % of AnyUp's own effect "
          f"(rule: > {100 * MAX_ERR_OVER_EFFECT:.0f} %); up to {worst:.2f} % of valid pixels change class "
          f"(rule: > {MAX_CHANGED_PCT:.0f} %) -> "
          + ("PRECISION MATTERS: re-run the AnyUp rows at full precision on SI" if matters
             else "precision does not drive the AnyUp result at inference")
          + f" | {(time.perf_counter() - t0) / 60:.1f} min")
    return out


def copy_back_check(encoder: str, norm: str, df: pd.DataFrame, data_root: Path, token: str | None,
                    layer: int | None = None, n_images: int = 10, tile: int = 512,
                    device: str = "cuda") -> dict:
    """Faster feature copy-back (2026-09-27): same values? how much time saved?

    On real tiles of `n_images` training images, per batch: the encoder forward
    (timed with a GPU sync), then the old copy-back (fp32 to the CPU, numpy
    float16 conversion) and the new one (float16 cast on the GPU, half the
    bytes copied), in alternating order. Compares both bit for bit; nothing is
    kept or written.
    """
    import torch

    from .data.tiling import TileSpec
    from .features import _prepare, _select
    from .models.encoders import ENCODERS, FrozenEncoder, to_host_fp16, to_host_fp16_old
    from .runs import ensure_packages

    allp = _select(df, ("train",), None)
    idx = np.unique(np.linspace(0, len(allp) - 1, min(n_images, len(allp))).round().astype(int))
    spec = ENCODERS[encoder]
    ensure_packages(spec.kind)
    enc = FrozenEncoder(spec, token=token)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    t = {"forward": 0.0, "old": 0.0, "new": 0.0}
    n_tiles, n_diff, k = 0, 0, 0
    with torch.inference_mode():
        for i in idx:
            it = _prepare(allp[i], data_root, TileSpec(size=tile, overlap=0.0), norm, enc.patch, with_na=True)
            x = torch.from_numpy(it["x"])
            for j in range(0, len(x), spec.batch):
                b = x[j:j + spec.batch].to(enc.device)
                sync(); t0 = time.perf_counter()
                tok = enc._tokens(b) if layer is None else enc._layer_tokens(b, [layer])[layer]
                g = enc._grid(tok, tile)
                sync(); t["forward"] += time.perf_counter() - t0
                order = (("old", to_host_fp16_old), ("new", to_host_fp16))
                res = {}
                for name, fn in (order if k % 2 == 0 else order[::-1]):
                    sync(); t0 = time.perf_counter()
                    res[name] = fn(g)
                    t[name] += time.perf_counter() - t0
                a, c = res["old"], res["new"]
                nan_a, nan_c = np.isnan(a), np.isnan(c)
                if not (np.array_equal(nan_a, nan_c)
                        and np.array_equal(a.view(np.uint16)[~nan_a], c.view(np.uint16)[~nan_c])):
                    n_diff += 1
                n_tiles += len(b)
                k += 1
    del enc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    ms = {p: 1000 * v / max(n_tiles, 1) for p, v in t.items()}
    old_total, new_total = ms["forward"] + ms["old"], ms["forward"] + ms["new"]
    out = {"encoder": f"{encoder} L{layer}", "tiles": n_tiles, "batches_different": n_diff,
           "forward_ms_per_tile": ms["forward"], "old_copy_ms_per_tile": ms["old"],
           "new_copy_ms_per_tile": ms["new"], "encoder_pass_saving": 1 - new_total / old_total}
    print(f"{out['encoder']}: {n_tiles} tiles | forward {ms['forward']:.1f} ms/tile | copy-back old "
          f"{ms['old']:.1f} -> new {ms['new']:.1f} ms/tile | encoder pass {out['encoder_pass_saving']:.0%} faster | "
          + ("IDENTICAL on every batch" if n_diff == 0 else f"DIFFERENT on {n_diff} batches -- stop and check"))
    return out
