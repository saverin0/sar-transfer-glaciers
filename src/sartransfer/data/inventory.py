"""Inventory of the CaFFe dataset -- discovery only, no modelling.

Written to *measure* the dataset rather than assume it. The only things taken
from primary sources are:

  * folder layout: bounding_boxes / fronts / sar_images / zones, the last three
    split into train and test     (PANGAEA landing page, doi:10.1594/PANGAEA.940950)
  * file name scheme:
    Glacier_Date_Satellite_SpatialResolutionInMeter_QualityFactor_Orbit(_Modality).png
                                  (PANGAEA landing page)

Everything else -- pixel values, dtypes, image sizes, counts, sensor token
spellings -- is read off disk here, not hard-coded.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Verified 2026-09-22: HTTP 200, application/zip, no login required. CC-BY-4.0.
CAFFE_ZIP_URL = "https://download.pangaea.de/dataset/940950/files/data_raw.zip"
CAFFE_ZIP_NAME = "data_raw.zip"

# Band per sensor, from the CaFFe paper's sensor list. Matching is on a lowercased
# substring of the Satellite token because the exact token spelling in the file
# names has NOT been verified -- scan() reports anything it cannot map.
BAND_BY_SENSOR_TOKEN = {
    "tsx": "X", "terrasar": "X", "tdx": "X", "tandem": "X",
    "s1": "C", "sentinel": "C", "envisat": "C", "asar": "C",
    "ers": "C", "rsat": "C", "radarsat": "C",
    "palsar": "L", "alos": "L",
}

Image.MAX_IMAGE_PIXELS = None  # CaFFe scenes are large; disable the decompression-bomb guard


# --------------------------------------------------------------------- get the data

def fetch_caffe_zip(dest_dir: str | Path) -> Path:
    """Download the CaFFe zip into dest_dir unless a complete copy is already there.

    ~2.9 GB. wget -c writes `<zip>.part` and resumes it after an interruption;
    the file gets the zip's name only after wget exits without error. A file
    under the zip's name that does not open as a zip (an interrupted download
    of the older code) is moved back to `.part` and resumed, never returned.
    Returns the local zip path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / CAFFE_ZIP_NAME
    part = zip_path.with_name(zip_path.name + ".part")
    if zip_path.exists():
        if zipfile.is_zipfile(zip_path):
            print(f"already present: {zip_path}  ({zip_path.stat().st_size / 1e9:.2f} GB)")
            return zip_path
        print(f"{zip_path} is not a complete zip -- resuming it as {part.name}")
        zip_path.replace(part)
    print(f"downloading {CAFFE_ZIP_URL} -> {zip_path}")
    subprocess.run(["wget", "-c", "-O", str(part), CAFFE_ZIP_URL], check=True)
    part.replace(zip_path)
    return zip_path


DONE_MARKER = ".extraction_complete"


def unzip_caffe(zip_path: str | Path, out_dir: str | Path, retries: int = 3) -> Path:
    """Extract the zip into out_dir, safely.

    - The zip is first copied from Drive to local disk in one sequential read and
      extracted from there: reading a 2.9 GB zip in random-access chunks over the
      Drive mount is slow and dies if the mount drops ("Transport endpoint is not
      connected", seen 2026-09-24).
    - A marker file is written only after a complete extraction. A folder without
      it is a half-finished extraction: it is deleted and redone, never used.
      (Before this, a partial folder was silently reused as if complete.)
    """
    import shutil
    import time

    zip_path, out_dir = Path(zip_path), Path(out_dir)
    if (out_dir / DONE_MARKER).exists():
        print(f"{out_dir} already extracted -- skipping")
        return out_dir
    if out_dir.exists():
        print(f"{out_dir} has no completion marker -- removing the partial extraction")
        shutil.rmtree(out_dir)

    local_zip = Path("/content/_caffe_zip") / zip_path.name
    if not str(zip_path).startswith("/content/drive"):
        local_zip = zip_path                          # already on local disk
    elif not (local_zip.exists() and local_zip.stat().st_size == zip_path.stat().st_size):
        local_zip.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, retries + 1):
            try:
                print(f"copying zip to local disk (attempt {attempt}) ...")
                shutil.copyfile(zip_path, local_zip)
                break
            except OSError as e:
                print(f"  copy failed: {e}")
                if attempt == retries:
                    raise RuntimeError("Drive dropped while copying the zip. Remount Drive: "
                                       "drive.mount('/content/drive', force_remount=True), "
                                       "then run this cell again.") from e
                time.sleep(10)

    tmp = out_dir.with_name(out_dir.name + "_partial")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(tmp)
    tmp.rename(out_dir)
    (out_dir / DONE_MARKER).write_text("ok\n")
    print(f"extracted to {out_dir}")
    return out_dir


def find_data_root(search_root: str | Path) -> Path:
    """Locate the directory that actually holds the sar_images folder.

    The zip may or may not add a wrapper directory; this finds it either way
    instead of assuming.
    """
    search_root = Path(search_root)
    candidates = [search_root / "sar_images", *sorted(search_root.rglob("sar_images"))]
    for cand in candidates:
        if cand.is_dir():
            return cand.parent
    raise FileNotFoundError(f"no 'sar_images' folder found under {search_root}")


# --------------------------------------------------------------------- structure

def tree(root: str | Path, max_depth: int = 3, sample_files: int = 2) -> None:
    """Print the folder tree with per-folder file counts and a few example names."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        pad = "  " * depth
        name = root.name if depth == 0 else rel.parts[-1]
        print(f"{pad}{name}/  [{len(filenames)} files]")
        for fn in sorted(filenames)[:sample_files]:
            print(f"{pad}    {fn}")
        if len(filenames) > sample_files:
            print(f"{pad}    ... ({len(filenames) - sample_files} more)")


def dir_size_bytes(root: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


# --------------------------------------------------------------------- file names

def parse_name(filename: str) -> dict:
    """Parse the documented CaFFe name scheme.

    Glacier_Date_Satellite_SpatialResolutionInMeter_QualityFactor_Orbit(_Modality).png

    Returns parsed=False plus the raw stem when the name does not fit, so
    unexpected names surface in the inventory instead of crashing or being
    silently coerced.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    out: dict = {"file": filename, "stem": stem, "n_parts": len(parts), "parsed": False}
    if len(parts) not in (6, 7):
        return out
    glacier, date, satellite, res, quality, orbit = parts[:6]
    out.update(
        parsed=True,
        glacier=glacier,
        date=date,
        satellite=satellite,
        resolution_m=_to_float(res),
        quality=_to_int(quality),
        orbit=orbit,
        modality=parts[6] if len(parts) == 7 else None,
        band=band_of(satellite),
    )
    return out


def band_of(satellite_token: str) -> str | None:
    t = satellite_token.lower()
    for key, band in BAND_BY_SENSOR_TOKEN.items():
        if key in t:
            return band
    return None


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def scan(data_root: str | Path,
         subdirs: tuple[str, ...] = ("sar_images", "zones", "fronts")) -> pd.DataFrame:
    """Walk the dataset, one row per file, with the parsed name fields.

    Columns: kind (sar_images/zones/fronts), split (train/test), path, bytes,
    plus everything parse_name returns.
    """
    data_root = Path(data_root)
    rows = []
    for kind in subdirs:
        base = data_root / kind
        if not base.is_dir():
            print(f"missing: {base}")
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(base)
            split = rel.parts[0] if len(rel.parts) > 1 else None
            rows.append({"kind": kind, "split": split, "path": str(p),
                         "bytes": p.stat().st_size, **parse_name(p.name)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    bad = df[~df["parsed"]]
    if len(bad):
        print(f"WARNING: {len(bad)} file names do not fit the documented scheme, e.g.:")
        print(bad["file"].head(5).to_string(index=False))
    unmapped = sorted(set(df.loc[df["parsed"] & df["band"].isna(), "satellite"].dropna()))
    if unmapped:
        print(f"WARNING: satellite tokens with no band mapping: {unmapped}")
    return df


def counts(df: pd.DataFrame, kind: str = "sar_images") -> dict[str, pd.DataFrame]:
    """Counts per split, glacier, satellite, band, resolution and quality factor."""
    d = df[df["kind"] == kind]
    return {
        "per_split": d.groupby("split").size().to_frame("n"),
        "per_glacier_split": pd.crosstab(d["glacier"], d["split"], margins=True),
        "per_satellite_split": pd.crosstab(d["satellite"], d["split"], margins=True),
        "per_band_split": pd.crosstab(d["band"].fillna("UNMAPPED"), d["split"], margins=True),
        "per_glacier_satellite": pd.crosstab(d["glacier"], d["satellite"], margins=True),
        "resolution_m": d["resolution_m"].value_counts().to_frame("n"),
        "quality": d["quality"].value_counts().sort_index().to_frame("n"),
    }


# --------------------------------------------------------------------- pixels

def image_stats(paths, limit: int | None = None) -> pd.DataFrame:
    """PIL mode, numpy dtype, shape and value range for each file."""
    paths = list(paths)
    if limit:
        paths = paths[:limit]
    rows = []
    for p in paths:
        with Image.open(p) as im:
            mode = im.mode
            a = np.asarray(im)
        rows.append({
            "path": str(p), "pil_mode": mode, "dtype": str(a.dtype),
            "height": a.shape[0], "width": a.shape[1], "ndim": a.ndim,
            "min": a.min(), "max": a.max(),
            "bits_per_pixel": a.dtype.itemsize * 8,
        })
    return pd.DataFrame(rows)


def unique_values(paths, limit: int | None = None, max_report: int = 32) -> pd.DataFrame:
    """Unique pixel values and their counts over a set of mask files.

    This is how the zone coding gets settled, instead of trusting the
    0 / 64 / 127 / 254 figure that reached us second-hand via GEO-Bench-2.
    """
    paths = list(paths)
    if limit:
        paths = paths[:limit]
    tally: Counter = Counter()
    for p in paths:
        with Image.open(p) as im:
            a = np.asarray(im)
        vals, cnts = np.unique(a, return_counts=True)
        tally.update(dict(zip(vals.tolist(), cnts.tolist())))
    total = sum(tally.values()) or 1
    df = pd.DataFrame({"value": list(tally.keys()), "pixels": list(tally.values())})
    df = df.sort_values("value").reset_index(drop=True)
    df["share_%"] = (df["pixels"] / total * 100).round(3)
    print(f"{len(paths)} files, {len(df)} distinct values")
    return df.head(max_report)


def dimension_consistency(stats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Per glacier: how many distinct (height, width) shapes occur."""
    merged = stats.merge(df[["path", "glacier", "satellite", "split"]], on="path", how="left")
    merged["shape"] = list(zip(merged["height"], merged["width"]))
    return merged.groupby("glacier").agg(
        n_images=("path", "size"),
        n_distinct_shapes=("shape", "nunique"),
        min_h=("height", "min"), max_h=("height", "max"),
        min_w=("width", "min"), max_w=("width", "max"),
    )


def parse_bbox(path: str | Path) -> dict:
    """Parse a `*_front_extent_coord.txt` file.

    Format observed in all 681 files: an `x,y` header then four corner points.
    The values are PIXEL coordinates of the matching image -- the official
    `validate_or_test.mask_prediction_with_bounding_box` clips them against the
    mask's width and height, and zeroes every predicted front pixel outside the
    box before metrics are computed.
    """
    lines = [ln for ln in Path(path).read_text().strip().splitlines() if ln.strip()]
    pts = [tuple(float(v) for v in ln.split(",")) for ln in lines[1:5]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys),
            "box_w_px": max(xs) - min(xs), "box_h_px": max(ys) - min(ys)}


def scan_bboxes(data_root: str | Path) -> pd.DataFrame:
    """One row per bounding-box file, with the parsed name fields and box size.

    Adds `box_w_m` / `box_h_m` = box size in pixels x the image's own resolution.
    If a glacier's box is a fixed geographic region, those metre values are
    constant across that glacier's images even though the pixel values are not.
    """
    rows = []
    for p in sorted((Path(data_root) / "bounding_boxes").glob("*.txt")):
        name = p.name.replace("_front_extent_coord.txt", ".png")
        row = {"path": str(p), **parse_name(name), **parse_bbox(p)}
        if row.get("resolution_m"):
            row["box_w_m"] = row["box_w_px"] * row["resolution_m"]
            row["box_h_m"] = row["box_h_px"] * row["resolution_m"]
        rows.append(row)
    return pd.DataFrame(rows)
