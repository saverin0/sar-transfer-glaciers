"""Explained GIFs of the frozen-feature views of notebook 14.

One GIF per test image. Inputs:
- from Google Drive (pulled with rclone, only when changed):
  embeddings_gif_<stem>.npz and embeddings_numbers.csv, which notebook 14 writes
  to MyDrive/sar-transfer/embeddings;
- the radar image, the zones and the front of each image, from the folder
  scripts/view_results.py already unpacked (its --dest folder, then unpacked/).
Frames, each with a caption and a colour key: the radar image; the truth with the
two query spots; per encoder the feature colours (raw, then AnyUp 1/4); per
encoder what looks like the ocean spot next to the front (raw, then AnyUp 1/4);
the zoom at the front (raw, AnyUp 1/4, AnyUp full); the numbers. Every number is
read from the notebook's CSV, none is typed in.

    python scripts/make_embedding_gifs.py              (pull, then make both GIFs)
    python scripts/make_embedding_gifs.py --no-pull    (use the files already pulled)

--emb is where the pulled files go (default data/sar-transfer-embeddings), --data
is view_results.py's unpacked folder (default data/sar-transfer-fiftyone/unpacked),
--out is where the GIFs go (default data/sar-transfer-gifs); all relative to the
folder you run from.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gifs import (BG, DIM, FG, GLACIER_NAME, SAT_NAME, TRUE_FRONT, ZONES,  # noqa: E402
                       font, frame, load_line, load_mask, overlay)

STEMS = {"COL_2020-03-08_S1_20_3_094": "Test image, never seen in training (the hard one).",
         "Mapple_2017-09-08_S1_20_2_009": "Test image, never seen in training (the easy one)."}
NAMES = {"cradio-v4-h": "C-RADIO", "dinov3-l-sat": "Satellite DINOv3", "dinov3-l-photo": "Photo DINOv3"}
QUERY_OCEAN, QUERY_GLACIER = (255, 212, 0), (34, 221, 68)
SIM_RAMP = ((205, 226, 251), (134, 182, 239), (57, 135, 229), (28, 92, 171), (13, 54, 107))   # notebook's ramp
NO_DATA = (40, 40, 40)
PANEL_W, PANEL_H, LABEL_H = 900, 420, 24
TILE = 512                                   # the encoders see the image in 512 px tiles, as in the study
TILE_NOTE = "Straight lines in the colours = edges of the 512 px squares; the encoder sees one square at a time."
RESERVED = [BG, FG, DIM, (90, 90, 90), TRUE_FRONT, QUERY_OCEAN, QUERY_GLACIER, NO_DATA,
            *[c for _, _, c in ZONES], *SIM_RAMP]


# ------------------------------------------------------------------ small helpers

def fit(h: int, w: int, max_w: int = PANEL_W, max_h: int = PANEL_H) -> tuple[int, int]:
    """Panel size (width, height) for an h x w image, at most 1.5x enlarged."""
    s = min(max_w / w, max_h / h, 1.5)
    return max(1, round(w * s)), max(1, round(h * s))


def grid_image(grid: np.ndarray, cell: int, shape: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """Patch/block grid (gh, gw[, 3]) -> image pixels (cropped to the image) -> panel size, nearest."""
    big = np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1)[:shape[0], :shape[1]]
    return np.asarray(Image.fromarray(big).resize(size, Image.NEAREST))


def sim_colours(sim: np.ndarray) -> np.ndarray:
    """Similarity map -> the notebook's light-to-dark blue ramp, scaled over its 2nd-98th percentile."""
    s = sim.astype(np.float32)
    ok = np.isfinite(s)
    out = np.empty(s.shape + (3,), np.uint8)
    out[~ok] = NO_DATA
    if ok.any():
        lo, hi = np.percentile(s[ok], [2, 98])
        t = np.clip((s[ok] - lo) / max(hi - lo, 1e-6), 0, 1) * (len(SIM_RAMP) - 1)
        i = np.minimum(t.astype(int), len(SIM_RAMP) - 2)
        f = (t - i)[:, None]
        ramp = np.array(SIM_RAMP, np.float32)
        out[ok] = (ramp[i] * (1 - f) + ramp[i + 1] * f).round().astype(np.uint8)
    return out


def with_lines(rgb: np.ndarray, front: np.ndarray | None, boxes: list[tuple[np.ndarray, tuple]],
               scale: tuple[float, float]) -> Image.Image:
    """Draw the true front (mask at panel size) and query boxes (image px y0, x0, y1, x1) on a panel."""
    rgb = rgb.copy()
    if front is not None:
        rgb[front] = TRUE_FRONT
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for box, colour in boxes:
        y0, x0, y1, x1 = box
        d.rectangle([x0 * scale[1] - 2, y0 * scale[0] - 2, x1 * scale[1] + 1, y1 * scale[0] + 1],
                    outline=colour, width=3)
    return img


def stack(panels: list[tuple[Image.Image, str]]) -> Image.Image:
    """Panels one below the other, each with a one-line label strip above it."""
    w = max(PANEL_W, max(p.width for p, _ in panels))        # full width, so labels are never clipped
    out = Image.new("RGB", (w, sum(p.height + LABEL_H for p, _ in panels)), BG)
    d, y = ImageDraw.Draw(out), 0
    for p, label in panels:
        d.text((6, y + 3), label, font=font(15, bold=True), fill=FG)
        out.paste(p, ((w - p.width) // 2, y + LABEL_H))
        y += p.height + LABEL_H
    return out


def with_note(img: Image.Image, text: str | None) -> Image.Image:
    """One grey line of text under a stacked image (none when text is None)."""
    if not text:
        return img
    out = Image.new("RGB", (img.width, img.height + 26), BG)
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((6, img.height + 5), text, font=font(14), fill=DIM)
    return out


def to_gif_frame(img: Image.Image) -> Image.Image:
    """Per-frame palette: an adaptive part for the smooth feature colours plus the exact
    reserved colours (text, lines, zones, ramp), so thin lines never change colour."""
    n = 256 - len(RESERVED)
    q = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = (q.getpalette() or [])[:3 * n]
    flat += [0] * (3 * n - len(flat)) + [x for c in RESERVED for x in c]
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    return img.quantize(palette=pal, dither=Image.Dither.NONE)


# ------------------------------------------------------------------ numbers from the CSV

class Numbers:
    def __init__(self, csv: Path):
        self.t = pd.read_csv(csv)

    def _rows(self, stem, enc, features):
        t = self.t
        return t[(t.stem == stem) & (t.encoder == enc) & (t.features == features)]

    def auc(self, stem, enc, features) -> float:
        r = self._rows(stem, enc, features)
        r = r[r["auc_ocean_vs_glacier"].notna()]
        return float(r["auc_ocean_vs_glacier"].iloc[0])

    def shares(self, stem, enc, features, query="ocean") -> tuple[float, float, int]:
        """(ocean share, glacier share, k) of the fair k20 matches of the query."""
        r = self._rows(stem, enc, features)
        r = r[(r["query"] == query) & r["k20_share_ocean"].notna()]
        row = r.iloc[0]
        return float(row["k20_share_ocean"]), float(row["k20_share_glacier"]), int(row["k20"])

    def crop_aucs(self, stem, enc) -> tuple[float, float, float]:
        r = self._rows(stem, enc, "anyup_full_crop")
        r = r[r["crop_auc_raw"].notna()].iloc[0]
        return float(r["crop_auc_raw"]), float(r["crop_auc_anyup_quarter"]), float(r["crop_auc_anyup_full"])


# ------------------------------------------------------------------ one GIF

def make_gif(stem: str, note: str, emb: Path, data: Path, nums: Numbers, out_dir: Path) -> Path:
    z = np.load(emb / f"embeddings_gif_{stem}.npz", allow_pickle=False)
    h, w = (int(v) for v in z["image_shape"])
    patch, block = int(z["patch"]), int(z["block"])
    encs = [str(e) for e in z["encoders"]]
    layers = {e: int(l) for e, l in zip(encs, z["layers"])}
    size = fit(h, w)
    scale = (size[1] / h, size[0] / w)
    sar = np.asarray(Image.open(data / "data" / f"{stem}.png").convert("L").resize(size, Image.BILINEAR))
    zones = load_mask(data / "fields" / "zones" / f"{stem}_zones.png", size)
    front = load_line(data / "fields" / "front" / f"{stem}_front.png", size)
    qo, qg = z["query_ocean_box"], z["query_glacier_box"]
    queries = [(qo, QUERY_OCEAN), (qg, QUERY_GLACIER)]
    g, date, sat, res = stem.split("_")[:4]
    where = f"{GLACIER_NAME.get(g, g)} · {SAT_NAME.get(sat, sat)} radar · {date} · {res} m pixels"
    zone_key = [(c, n, "box") for _, n, c in ZONES] + [(BG, "no colour = no data", "box")]
    marks = [(TRUE_FRONT, "real front", "line"), (QUERY_OCEAN, "ocean spot next to the front", "line"),
             (QUERY_GLACIER, "glacier spot next to the front", "line")]
    tiles_note = TILE_NOTE if h > TILE or w > TILE else None   # only where the image spans several tiles
    frames, durations = [], []

    def add(img, title, subtitle, legend, ms=3500):
        frames.append(frame(img, title, subtitle, legend))
        durations.append(ms)

    truth = np.asarray(overlay(sar, zones, []))
    add(stack([(Image.fromarray(np.repeat(sar[..., None], 3, axis=2)), "the radar image"),
               (with_lines(truth, front, queries, scale),
                "the truth, with an ocean spot (yellow) and a glacier spot (green) next to the front")]),
        where, note + " Frozen features only, no training.", zone_key + marks, 5000)

    for e in encs:                                   # feature colours: raw, then AnyUp 1/4
        raw = grid_image(z[f"{e}/pca_raw"], patch, (h, w), size)
        quarter = grid_image(z[f"{e}/pca_quarter"], block, (h, w), size)
        a_raw, a_q = nums.auc(stem, e, "raw"), nums.auc(stem, e, "anyup_quarter")
        img = with_note(stack([(with_lines(raw, front, [], scale), f"raw features (16 px patches)   AUC {a_raw:.3f}"),
                               (with_lines(quarter, front, [], scale), f"+ AnyUp (4 px blocks)   AUC {a_q:.3f}")]),
                        tiles_note)
        add(img, f"{NAMES.get(e, e)} (layer {layers[e]}): how its features colour the image",
            "Same colour = similar features. AUC = how well its features keep ocean and glacier apart (1 = fully).",
            [(TRUE_FRONT, "real front", "line")])

    for e in encs:                                   # similarity to the ocean spot: raw, then AnyUp 1/4
        raw = grid_image(sim_colours(z[f"{e}/sim_ocean_raw"]), patch, (h, w), size)
        quarter = grid_image(sim_colours(z[f"{e}/sim_ocean_quarter"]), block, (h, w), size)
        o_r, g_r, k_r = nums.shares(stem, e, "raw")
        o_q, g_q, k_q = nums.shares(stem, e, "anyup_quarter")
        img = with_note(stack([(with_lines(raw, front, [(qo, QUERY_OCEAN)], scale),
                                f"raw: most similar {k_r} patches = ocean {o_r:.2f}, glacier {g_r:.2f}"),
                               (with_lines(quarter, front, [(qo, QUERY_OCEAN)], scale),
                                f"+ AnyUp, same area ({k_q} blocks) = ocean {o_q:.2f}, glacier {g_q:.2f}")]),
                        tiles_note)
        add(img, f"{NAMES.get(e, e)}: what looks like the ocean spot next to the front?",
            "Darker blue = more similar to the yellow box. High glacier share = glacier looks like this water to it.",
            [(QUERY_OCEAN, "ocean spot", "line"), (TRUE_FRONT, "real front", "line"),
             (SIM_RAMP[-1], "most similar", "box"), (SIM_RAMP[0], "least similar", "box")], 4500)

    y0, x0, ch, cw = (int(v) for v in z["crop_box"])     # zoom: rows = encoders, columns = raw / 1/4 / full
    cell = min(280, (PANEL_W - 20) // 3)
    # the viewer's front file is widened for display: thin it back to a 1 px line, then 3 px at zoom scale
    full = np.asarray(Image.open(data / "fields" / "front" / f"{stem}_front.png")) > 0
    fr = skeletonize(full)[y0:y0 + ch, x0:x0 + cw].astype(np.uint8) * 255
    fr = binary_dilation(np.asarray(Image.fromarray(fr).resize((cell, cell), Image.BOX)) > 0)
    sy, sx = cell / ch, cell / cw
    qy0, qx0, qy1, qx1 = (int(v) for v in qo)
    qbox = [(qx0 - x0) * sx - 2, (qy0 - y0) * sy - 2, (qx1 - x0) * sx + 1, (qy1 - y0) * sy + 1]
    grid = Image.new("RGB", (3 * cell + 20, len(encs) * (cell + LABEL_H)), BG)
    d = ImageDraw.Draw(grid)
    for i, e in enumerate(encs):
        crop_aucs = nums.crop_aucs(stem, e)
        for j, (key, name) in enumerate((("raw", "raw"), ("quarter", "AnyUp 1/4"), ("full", "AnyUp full"))):
            tile = np.asarray(Image.fromarray(z[f"{e}/pca_crop_{key}"]).resize((cell, cell), Image.NEAREST)).copy()
            tile[fr] = TRUE_FRONT
            tile = Image.fromarray(tile)
            ImageDraw.Draw(tile).rectangle(qbox, outline=QUERY_OCEAN, width=3)
            x, y = j * (cell + 10), i * (cell + LABEL_H)
            d.text((x + 4, y + 3), f"{NAMES.get(e, e)} {name}  AUC {crop_aucs[j]:.3f}", font=font(13, bold=True), fill=FG)
            grid.paste(tile, (x, y + LABEL_H))
    add(grid, f"Zoom: {ch} x {cw} px around the ocean spot at the front",
        "Sharper with AnyUp? Left: raw 16 px patches; middle: AnyUp 1/4; right: AnyUp at full resolution.",
        [(QUERY_OCEAN, "ocean spot", "line"), (TRUE_FRONT, "real front", "line")], 6000)

    table = Image.new("RGB", (PANEL_W, 60 + 36 * (len(encs) + 1) + 130), BG)   # the numbers + how to read
    d = ImageDraw.Draw(table)
    cols = ("encoder", "AUC raw", "AUC + AnyUp", "ocean spot: glacier share raw", "+ AnyUp")
    xs = (14, 250, 380, 540, 800)
    for x, c in zip(xs, cols):
        d.text((x, 14), c, font=font(15, bold=True), fill=FG)
    for i, e in enumerate(encs):
        y = 56 + 36 * i
        vals = (f"{NAMES.get(e, e)} (L{layers[e]})", f"{nums.auc(stem, e, 'raw'):.3f}",
                f"{nums.auc(stem, e, 'anyup_quarter'):.3f}", f"{nums.shares(stem, e, 'raw')[1]:.2f}",
                f"{nums.shares(stem, e, 'anyup_quarter')[1]:.2f}")
        for x, v in zip(xs, vals):
            d.text((x, y), v, font=font(16), fill=FG)
    y = 56 + 36 * len(encs) + 30
    for line in ("AUC: 1 = ocean and glacier features fully apart, 0.5 = completely mixed.",
                 "Glacier share: how many of the areas most similar to the ocean spot are glacier (0 = none).",
                 "AnyUp: the same frozen features, sharpened to 4 px blocks by the AnyUp upsampler."):
        d.text((14, y), line, font=font(15), fill=DIM)
        y += 28
    add(table, "The numbers for this image",
        "Descriptive, one image: AUC uses this image's own labels, so it is optimistic. No training anywhere.",
        [], 6000)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"embeddings_{stem}.gif"
    fw, fh = max(f.width for f in frames), max(f.height for f in frames)   # a GIF has ONE canvas size:
    padded = []                                                            # pad every frame to it, top-aligned
    for f in frames:
        canvas = Image.new("RGB", (fw, fh), BG)
        top = 0 if f.height > 0.6 * fh else (fh - f.height) // 2      # short frames (the numbers) centred
        canvas.paste(f, ((fw - f.width) // 2, top))
        padded.append(canvas)
    gif = [to_gif_frame(f) for f in padded]
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb", type=Path, default=Path("data") / "sar-transfer-embeddings",
                    help="local folder for the notebook's npz + CSV (default: data/sar-transfer-embeddings)")
    ap.add_argument("--data", type=Path, default=Path("data") / "sar-transfer-fiftyone" / "unpacked",
                    help="view_results.py's unpacked folder: data/<stem>.png, fields/zones, fields/front "
                         "(default: data/sar-transfer-fiftyone/unpacked)")
    ap.add_argument("--out", type=Path, default=Path("data") / "sar-transfer-gifs",
                    help="folder for the GIFs (default: data/sar-transfer-gifs)")
    ap.add_argument("--remote", default="gdrive:", help="your rclone remote for Google Drive")
    ap.add_argument("--rclone", default="rclone", help="rclone command (default: the one on PATH)")
    ap.add_argument("--no-pull", action="store_true", help="use the files already pulled, do not ask Drive")
    a = ap.parse_args()
    if not a.no_pull:
        remote = a.remote if a.remote.endswith(":") else a.remote + ":"
        print("pulling the notebook's GIF arrays and numbers from Drive (only if changed) ...")
        try:
            subprocess.run([a.rclone, "copy", f"{remote}sar-transfer/embeddings", str(a.emb),
                            "--include", "embeddings_gif_*.npz", "--include", "embeddings_numbers.csv",
                            "--progress"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise SystemExit(f"rclone pull failed ({err}). Check that rclone is installed (or pass --rclone), that "
                             f"the remote '{remote}' exists (--remote), and that the embeddings notebook has written "
                             "its files to Drive; or use --no-pull for files already pulled.")
    nums = Numbers(a.emb / "embeddings_numbers.csv")
    for stem, note in STEMS.items():
        p = make_gif(stem, note, a.emb, a.data, nums, a.out)
        print(f"{p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
