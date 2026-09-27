"""Explained GIFs of the test predictions: one hard and one easy test image.

Runs on your computer, CPU only, from the files scripts/view_results.py already
pulled into its --dest folder: the radar image, the dataset's labels, and the
four redrawn predictions of notebook 11. Frames, each with a caption and a
colour key: radar image, the truth, U-Net, C-RADIO probe, C-RADIO decoder,
satellite-DINOv3 decoder, all fronts together. Per-image errors come from
notebook 11's check file (equal to the saved test results), not typed in.

    python scripts/make_gifs.py [--root DIR] [--out DIR]

--root is view_results.py's --dest folder (default data/sar-transfer-fiftyone),
--out is where the GIFs go (default data/sar-transfer-gifs); both relative to
the folder you run from.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

HARD = "COL_2020-03-08_S1_20_3_094"          # worst test image of the C-RADIO decoder; hard for every model
MODELS = [  # key, name in the caption, front colour
    ("unet", "U-Net trained from scratch (the reference)", (255, 212, 0)),
    ("cradio_row1", "C-RADIO + one linear layer (probe)", (255, 140, 0)),
    ("cradio_row3", "C-RADIO + small decoder", (34, 221, 68)),
    ("dinosat_row3", "Satellite DINOv3 + small decoder", (255, 51, 204)),
]
ZONES = [(64, "rock", (158, 158, 158)), (127, "glacier", (191, 230, 255)),
         (254, "ocean incl. floating ice", (31, 59, 140))]
TRUE_FRONT = (255, 32, 32)
GLACIER_NAME = {"COL": "Columbia Glacier", "Mapple": "Mapple Glacier"}
SAT_NAME = {"S1": "Sentinel-1", "TDX": "TanDEM-X", "TSX": "TerraSAR-X", "PALSAR": "ALOS PALSAR",
            "ENVISAT": "Envisat", "ERS": "ERS", "RSAT": "RADARSAT"}
MAX_W, MAX_H, HEAD, FOOT = 900, 640, 78, 86
CREDIT = "Data: CaFFe, Gourmelon et al. 2022, CC BY 4.0, doi:10.1594/PANGAEA.940950. Colours and lines added."
BG, FG, DIM = (17, 17, 17), (245, 245, 245), (170, 170, 170)


def palette() -> Image.Image:
    """One fixed 256-colour palette: greys, each zone colour blended over greys, and the
    exact line and text colours. An automatic palette drops the thin lines' colours
    (orange came out red, looking like the real front)."""
    cols = [tuple([v] * 3) for v in np.linspace(0, 255, 96).round().astype(int)]
    for _, _, c in ZONES:
        cols += [tuple((0.55 * v + 0.45 * np.array(c)).round().astype(int)) for v in np.linspace(0, 255, 48)]
    cols += [c for _, _, c in ZONES] + [m[2] for m in MODELS] + [TRUE_FRONT, BG, FG, DIM, (90, 90, 90)]
    cols = list(dict.fromkeys(cols))[:256]
    flat = [int(x) for c in cols for x in c] + [0] * (768 - 3 * len(cols))
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    return pal


def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).resize(size, Image.NEAREST))


def load_line(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Thin line masks survive downscaling: box filter, anything > 0 counts, then 1 px wider."""
    m = (np.asarray(Image.open(path)) > 0).astype(np.uint8) * 255
    small = np.asarray(Image.fromarray(m).resize(size, Image.BOX)) > 0
    return binary_dilation(small, iterations=1)


def overlay(sar: np.ndarray, zones: np.ndarray | None, lines: list[tuple[np.ndarray, tuple]]) -> Image.Image:
    rgb = np.repeat(sar[..., None], 3, axis=2).astype(np.float32)
    if zones is not None:
        for value, _, colour in ZONES:
            hit = zones == value
            rgb[hit] = 0.55 * rgb[hit] + 0.45 * np.array(colour, np.float32)
    for mask, colour in lines:
        rgb[mask] = colour
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def frame(img: Image.Image, title: str, subtitle: str, legend: list[tuple[tuple, str, str]]) -> Image.Image:
    """legend: (colour, label, kind) with kind 'box' or 'line'."""
    w = max(img.width, MAX_W)
    out = Image.new("RGB", (w, HEAD + img.height + FOOT), BG)
    out.paste(img, ((w - img.width) // 2, HEAD))
    d = ImageDraw.Draw(out)
    d.text((14, 10), title, font=font(24, bold=True), fill=FG)
    d.text((14, 44), subtitle, font=font(17), fill=DIM)
    x, y, f = 14, HEAD + img.height + 12, font(16)
    for colour, label, kind in legend:
        need = 30 + d.textlength(label, font=f) + 22
        if x + need > w - 10:
            x, y = 14, y + 24
        if kind == "box":
            d.rectangle([x, y + 2, x + 20, y + 18], fill=colour, outline=(90, 90, 90))
        else:
            d.rectangle([x, y + 8, x + 20, y + 12], fill=colour)
        d.text((x + 28, y), label, font=f, fill=FG)
        x += need
    d.text((14, out.height - 20), CREDIT, font=font(12), fill=DIM)
    return out


def error_text(row: pd.Series) -> str:
    if not bool(row["has_front"]):
        return "found NO front on this image"
    return f"front is {row['mean_m']:,.0f} m off on average (per-image error)"


def pick_easy(check: pd.DataFrame) -> str:
    """Mapple Sentinel-1 image where the worst of the four models is best (all four found a front)."""
    c = check[check.stem.str.startswith("Mapple_") & check.stem.str.contains("_S1_")]
    per = c.groupby("stem").agg(all_front=("has_front", "all"), worst=("mean_m", "max"), n=("key", "nunique"))
    per = per[per.all_front & (per.n == len(MODELS))]
    return str(per.worst.idxmin())


def make_gif(stem: str, root: Path, check: pd.DataFrame, out_dir: Path, note: str) -> Path:
    un, pred = root / "unpacked", root / "predictions"
    sar_img = Image.open(un / "data" / f"{stem}.png").convert("L")
    s = min(1.0, MAX_W / sar_img.width, MAX_H / sar_img.height)
    size = (max(1, round(sar_img.width * s)), max(1, round(sar_img.height * s)))
    sar = np.asarray(sar_img.resize(size, Image.BILINEAR))
    truth_z = load_mask(un / "fields" / "zones" / f"{stem}_zones.png", size)
    truth_f = load_line(un / "fields" / "front" / f"{stem}_front.png", size)
    g, date, sat, res = stem.split("_")[:4]
    where = f"{GLACIER_NAME.get(g, g)} · {SAT_NAME.get(sat, sat)} radar · {date} · {res} m pixels"
    zone_key = [(c, n, "box") for _, n, c in ZONES] + [(BG, "no colour = no data", "box")]

    frames = [frame(overlay(sar, None, []), where, note, []),
              frame(overlay(sar, truth_z, [(truth_f, TRUE_FRONT)]), "The truth (the dataset's labels)",
                    "Colours = what each area really is. Red line = the real calving front.",
                    zone_key + [(TRUE_FRONT, "real front", "line")])]
    all_lines, all_key = [(truth_f, TRUE_FRONT)], [(TRUE_FRONT, "real front", "line")]
    for key, name, colour in MODELS:
        row = check[(check.key == key) & (check.stem == stem)].iloc[0]
        pz = load_mask(pred / key / f"{stem}_zones.png", size)
        pf = load_line(pred / key / f"{stem}_front.png", size)
        frames.append(frame(overlay(sar, pz, [(truth_f, TRUE_FRONT), (pf, colour)]), name,
                            f"Colours = what the model thinks each area is. Its {error_text(row)}.",
                            zone_key + [(TRUE_FRONT, "real front", "line"), (colour, "model's front", "line")]))
        all_lines.append((pf, colour))
        short = name.split(" (")[0]
        err = "no front" if not bool(row["has_front"]) else f"{row['mean_m']:,.0f} m"
        all_key.append((colour, f"{short}: {err}", "line"))
    frames.append(frame(overlay(sar, None, all_lines[1:] + all_lines[:1]), "All fronts together",   # red on top
                        "Red = real front. The closer a coloured line is to red, the better.", all_key))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.gif"
    durations = [2500, 3500] + [3500] * len(MODELS) + [5000]
    pal = palette()
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("data") / "sar-transfer-fiftyone",
                    help="view_results.py's --dest folder (default: data/sar-transfer-fiftyone)")
    ap.add_argument("--out", type=Path, default=Path("data") / "sar-transfer-gifs",
                    help="folder for the GIFs (default: data/sar-transfer-gifs)")
    a = ap.parse_args()
    check = pd.read_csv(a.root / "predictions" / "check_vs_saved.csv")
    bad = check[~(check.front_match & check.zones_match)]
    if len(bad):
        raise SystemExit(f"{len(bad)} predictions do not match the saved results -- not drawing them")
    easy = pick_easy(check)
    for stem, note in ((HARD, "Test image, never seen in training. Worst image for the U-Net and the C-RADIO decoder."),
                       (easy, "Test image, never seen in training. Same satellite as the hard one, but an easy case.")):
        p = make_gif(stem, a.root, check, a.out, note)
        print(f"{p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
