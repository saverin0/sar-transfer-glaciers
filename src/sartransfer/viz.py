"""FiftyOne view of the saved per-image results (no GPU, nothing is scored again).

One sample per scored SAR image (the validation glacier SI and the two test
glaciers). Each sample carries:

- `zones`: the true zones, the original CaFFe zone PNG shown as it is
  (64 stone, 127 glacier, 254 ocean; 0 = NA is left without colour);
- `front`: the true front, a thickened copy only so a 1 px line stays visible
  when the image is shown small. Scoring always used the original line;
- the name fields (glacier, date, satellite, band, resolution);
- one small record per saved run, read from the `*images__*.csv` files the
  runs wrote: `mde_m` (this image's mean front distance, empty when no front
  was found), `has_front`, `front_px`, `mIoU` (per-image 4-class zone IoU).

FiftyOne is imported inside the functions, so the rest of the package does not
need it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data.inventory import parse_name
from .features import TEST_GLACIERS

ZONE_TARGETS = {64: "stone", 127: "glacier", 254: "ocean"}     # data/targets.py, official values
FRONT_TARGETS = {1: "front"}


def run_key(csv_path: Path) -> str:
    """Field name for one results file: `heads_test_images__<tag>.csv` -> `heads_test__<tag>`."""
    prefix, tag = csv_path.stem.split("images__", 1)
    return re.sub(r"[^0-9A-Za-z_]", "_", f"{prefix}_{tag}")


def read_results(res_dir: str | Path) -> dict[str, pd.DataFrame]:
    """{run key: per-image records} for every `*images__*.csv` on Drive."""
    out = {}
    for p in sorted(Path(res_dir).glob("*images__*.csv")):
        df = pd.read_csv(p)
        if "stem" not in df.columns or "has_front" not in df.columns:
            print(f"  skipped {p.name}: not a per-image front file")
            continue
        out[run_key(p)] = df
    return out


def _front_display(data_root: Path, split_dir: str, stem: str, out_dir: Path) -> Path:
    """Thickened 0/1 copy of the true front (display only)."""
    from PIL import Image
    from scipy.ndimage import binary_dilation

    front = np.asarray(Image.open(data_root / "fronts" / split_dir / f"{stem}_front.png")) > 0
    width = max(2, round(max(front.shape) / 500))
    thick = binary_dilation(front, iterations=width)
    path = out_dir / f"{stem}_front.png"
    Image.fromarray(thick.astype(np.uint8)).save(path)
    return path


def build_dataset(data_root: str | Path, res_dir: str | Path, out_dir: str | Path,
                  name: str = "sar-transfer-results"):
    """Build (or rebuild) the FiftyOne dataset from the images on the runtime and the CSVs on Drive."""
    import fiftyone as fo

    data_root, out_dir = Path(data_root), Path(out_dir)
    (out_dir / "fronts").mkdir(parents=True, exist_ok=True)
    results = read_results(res_dir)
    stems = sorted({s for df in results.values() for s in df["stem"].astype(str)})
    print(f"{len(results)} result files, {len(stems)} scored images")

    # per stem: {run key: record}
    per_image: dict[str, dict] = {s: {} for s in stems}
    for key, df in results.items():
        for r in df.to_dict("records"):
            rec = {"has_front": bool(r["has_front"]), "front_px": int(r["front_px"]),
                   "mde_m": None if pd.isna(r.get("mean_m")) else float(r["mean_m"]),
                   "mIoU": None if pd.isna(r.get("img_mIoU", np.nan)) else float(r["img_mIoU"])}
            per_image[str(r["stem"])][key] = rec

    samples = []
    for stem in stems:
        m = parse_name(stem + ".png")
        split_dir = "test" if m["glacier"] in TEST_GLACIERS else "train"      # SI lives in train
        img = data_root / "sar_images" / split_dir / f"{stem}.png"
        zones = data_root / "zones" / split_dir / f"{stem}_zones.png"
        if not img.exists() or not zones.exists():
            raise FileNotFoundError(f"missing image or zones for {stem}")
        s = fo.Sample(filepath=str(img), tags=["test" if split_dir == "test" else "SI"])
        s["zones"] = fo.Segmentation(mask_path=str(zones))
        s["front"] = fo.Segmentation(mask_path=str(_front_display(data_root, split_dir, stem, out_dir / "fronts")))
        for k in ("glacier", "date", "satellite", "band", "resolution_m", "quality", "orbit"):
            s[k] = m[k]
        for key, rec in per_image[stem].items():
            s[key] = fo.DynamicEmbeddedDocument(**rec)
        samples.append(s)

    ds = fo.Dataset(name, overwrite=True)
    ds.add_samples(samples)
    ds.add_dynamic_sample_fields()
    ds.mask_targets = {"zones": ZONE_TARGETS, "front": FRONT_TARGETS}
    ds.save()
    print(f"dataset '{name}': {len(ds)} samples, {len(results)} runs")
    return ds
