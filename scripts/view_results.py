"""Open the FiftyOne viewer on your computer for the dataset notebook 10 saved to Drive.

Run it with the Python of a separate environment that has FiftyOne installed, in
the version notebook 10 printed (`pip install fiftyone==<that version>`):

    python scripts/view_results.py [--dest DIR] [--remote gdrive:] [--no-pull]

- the zip is pulled from Google Drive with rclone, only when it changed.
  `--remote` is your rclone remote for Google Drive (default `gdrive:`); rclone
  must already be set up for it (`rclone config`). `--no-pull` skips Drive and
  uses the zip already in `--dest`;
- `--dest` is the local folder for the zips and their unpacked copies (default
  `data/sar-transfer-fiftyone`, relative to the folder you run from; `/data/` is
  gitignored at the repository root);
- a new zip is unpacked and loaded again automatically;
- the predicted zones and fronts drawn by notebook 11 (a second zip) are pulled
  the same way and added as layers `pred_<run>_zones` / `pred_<run>_front`;
- zones and fronts open in fixed colours (stone grey, glacier light blue, ocean
  dark blue, true front red, predicted front yellow);
- usage tracking is switched off before FiftyOne is imported, and FiftyOne's
  database and home folder live inside the environment's folder (sys.prefix).
Nothing is uploaded anywhere; the viewer runs at http://localhost:5151.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# paths inside the rclone remote (--remote)
DRIVE_ZIP = "sar-transfer/fiftyone/sar-transfer-results.zip"
DRIVE_PRED_ZIP = "sar-transfer/predictions/sar-transfer-predictions.zip"
ZONE_TARGETS = {64: "stone", 127: "glacier", 254: "ocean"}      # official grey values
PRED_FRONT_TARGETS = {2: "predicted front"}                     # true front is 1
# fixed colours: stone grey, glacier light blue, ocean dark blue, true front red, predicted front yellow
ZONE_COLORS = {64: "#9e9e9e", 127: "#bfe6ff", 254: "#1f3b8c"}
FRONT_COLOR, PRED_FRONT_COLOR = "#ff2020", "#ffd400"


def apply_colors(fo, ds) -> None:
    """Colour masks by value with the fixed colours above, for every zones/front layer."""
    def entry(path, colors):
        return {"path": path, "maskTargetsColors": [{"intTarget": v, "color": c} for v, c in colors.items()]}
    fields = []
    for name in ds.get_field_schema():
        if name == "zones" or (name.startswith("pred_") and name.endswith("_zones")):
            fields.append(entry(name, ZONE_COLORS))
        elif name == "front":
            fields.append(entry(name, {1: FRONT_COLOR}))
        elif name.startswith("pred_") and name.endswith("_front"):
            fields.append(entry(name, {2: PRED_FRONT_COLOR}))
    ds.app_config.color_scheme = fo.ColorScheme(color_by="value", opacity=0.5, fields=fields)
    ds.save()


def unpack(zip_file: Path, folder: Path, dest: Path) -> None:
    """Unpack a zip into a fresh folder: the old copy is deleted first, so no files from an older zip stay."""
    folder, dest = folder.resolve(), dest.resolve()
    if dest not in folder.parents:                          # only a folder inside --dest, never --dest itself
        sys.exit(f"will not delete {folder}: it is not inside {dest}")
    if folder.is_dir():
        shutil.rmtree(folder)
    with zipfile.ZipFile(zip_file) as z:
        z.extractall(folder)


def add_predictions(fo, ds, pred_zip: Path, folder: Path, dest: Path) -> None:
    """Unpack the predictions zip and (re)attach one zones + one front layer per run."""
    print(f"new predictions: unpacking to {folder} ...")
    unpack(pred_zip, folder, dest)
    keys = sorted(p.name for p in folder.iterdir() if p.is_dir())
    for name in list(ds.get_field_schema()):                # every old run, also ones missing from this zip
        if name.startswith("pred_") and name.endswith(("_zones", "_front")):
            ds.delete_sample_field(name)
    n = 0
    for s in ds.iter_samples(autosave=True, progress=True):
        stem = Path(s.filepath).stem
        for key in keys:
            zones, front = folder / key / f"{stem}_zones.png", folder / key / f"{stem}_front.png"
            if zones.is_file() and front.is_file():
                s[f"pred_{key}_zones"] = fo.Segmentation(mask_path=str(zones))
                s[f"pred_{key}_front"] = fo.Segmentation(mask_path=str(front))
                n += 1
    targets = {k: v for k, v in ds.mask_targets.items() if not k.startswith("pred_")}
    for key in keys:
        targets[f"pred_{key}_zones"] = ZONE_TARGETS
        targets[f"pred_{key}_front"] = PRED_FRONT_TARGETS
    ds.mask_targets = targets
    ds.save()
    print(f"predictions added: {', '.join(keys)} ({n} image layers)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path("data") / "sar-transfer-fiftyone",
                    help="local folder for the zips and their unpacked copies "
                         "(default: data/sar-transfer-fiftyone, relative to where you run this)")
    ap.add_argument("--remote", default="gdrive:",
                    help="your rclone remote for Google Drive (default: gdrive:)")
    ap.add_argument("--rclone", default="rclone", help="rclone command (default: the one on PATH)")
    ap.add_argument("--no-pull", action="store_true", help="use the local zip, do not ask Drive")
    ap.add_argument("--name", default="sar-transfer-results")
    a = ap.parse_args()

    dest = a.dest.resolve()
    db = (Path(sys.prefix) / "fiftyone_db").resolve()
    remote = a.remote if a.remote.endswith(":") else a.remote + ":"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "sar-transfer-results.zip"
    pred_zip = dest / "sar-transfer-predictions.zip"

    if not a.no_pull:
        try:
            print(f"pulling {remote}{DRIVE_ZIP} (only if it changed) ...")
            subprocess.run([a.rclone, "copy", remote + DRIVE_ZIP, str(dest), "--progress"], check=True)
            print(f"pulling {remote}{DRIVE_PRED_ZIP} (only if it changed) ...")
            pulled = subprocess.run([a.rclone, "copy", remote + DRIVE_PRED_ZIP, str(dest),
                                     "--progress"])
        except FileNotFoundError:
            sys.exit(f"rclone not found: {a.rclone}\n"
                     "Install rclone or give its path with --rclone, or use --no-pull for zips already in --dest.")
        except subprocess.CalledProcessError:
            sys.exit(f"rclone could not pull {remote}{DRIVE_ZIP}\n"
                     "Check the remote name (--remote, default gdrive:, set up with `rclone config`) and that\n"
                     "notebook 10 has written the zip to Drive, or use --no-pull for zips already in --dest.")
        if pulled.returncode:
            print("  no predictions on Drive yet (notebook 11) -- showing the results only")
    if not zip_path.is_file():
        sys.exit(f"not found: {zip_path}")

    os.environ["FIFTYONE_DO_NOT_TRACK"] = "true"
    os.environ["FIFTYONE_DATABASE_DIR"] = str(db)
    # FiftyOne keeps small files (uid, welcome.json) in ~/.fiftyone; point "~" inside the
    # environment's folder for this process and the viewer server it starts. Set only now:
    # rclone above must still find its config.
    home = (Path(sys.prefix) / "fiftyone_home").resolve()
    home.mkdir(exist_ok=True)
    os.environ["USERPROFILE"] = os.environ["HOME"] = str(home)
    import fiftyone as fo                                   # only after the two settings above

    folder = dest / "unpacked"
    stamp = folder / ".from_zip"
    zip_id = f"{zip_path.stat().st_size} {zip_path.stat().st_mtime_ns}"
    new = not stamp.is_file() or stamp.read_text() != zip_id
    if new:
        print(f"new export: unpacking to {folder} ...")
        unpack(zip_path, folder, dest)
        stamp.write_text(zip_id)
        if fo.dataset_exists(a.name):
            fo.delete_dataset(a.name)
    if fo.dataset_exists(a.name):
        ds = fo.load_dataset(a.name)
    else:
        ds = fo.Dataset.from_dir(dataset_dir=str(folder), dataset_type=fo.types.FiftyOneDataset, name=a.name)
        ds.persistent = True
    if pred_zip.is_file():
        pred_id = f"{pred_zip.stat().st_size} {pred_zip.stat().st_mtime_ns}"
        if ds.info.get("pred_zip") != pred_id:
            add_predictions(fo, ds, pred_zip, dest / "predictions", dest)
            ds.info["pred_zip"] = pred_id
            ds.save()
    apply_colors(fo, ds)
    print(f"fiftyone {fo.__version__} | tracking off: {fo.config.do_not_track} | database: {db}")
    print(f"{a.name}: {len(ds)} images")
    session = fo.launch_app(ds)
    print("viewer:", session.url, "-- close with Ctrl+C")
    session.wait()


if __name__ == "__main__":
    main()
