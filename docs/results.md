# sar-transfer-glaciers: full results

This file holds the long tables and caveats behind the [README](../README.md).
The README's definitions apply here: SI is the validation glacier, TEST the two
test glaciers (COL and Mapple), MDE is in metres with images without a
predicted front left out and counted, and ± is the sample SD (n − 1) over three
seeds. For published numbers, ± is as reported by the source paper (the CaFFe
baseline: 5 runs).

Notes on the numbers:

- Most numbers can be checked against the saved outputs of the notebooks in
  `notebooks/`. Some come only from result CSVs on Google Drive or from
  archived notebooks that are not in this repository, for example the first
  run's `per_tile` numbers, the Mapple per-sensor zone table, the TEST band
  patch counts, two-decimal SI pixel mIoUs and the Cosmos results. The first
  run's CSV was later overwritten on Drive by the `per_image` run. The CSVs are
  not in git (see [Source files](#source-files)).
- Seed means and SDs are computed from the unrounded per-seed values the
  notebook printed. They can differ by 1 in the last digit from a mean of the
  rounded values shown in the tables.
- "Rock" in the README is `stone` in the code and the per-class columns below.

## Contents

- [Data](#data)
- [Methods](#methods)
- [Arm 1: zones on SI](#arm-1-zones-on-si)
- [Band vs resolution](#band-vs-resolution)
- [Arm 1: fronts on SI](#arm-1-fronts-on-si)
- [Arm 1: TEST](#arm-1-test)
- [Published references](#published-references)
- [Arm 2: layer sweep](#arm-2-layer-sweep)
- [Arm 2: heads and parameter counts](#arm-2-heads-and-parameter-counts)
- [Arm 2: SI](#arm-2-si)
- [Arm 2: TEST](#arm-2-test)
- [Photo vs satellite DINOv3](#photo-vs-satellite-dinov3)
- [AnyUp precision check](#anyup-precision-check)
- [Approach A: band and resolution conditioning](#approach-a-band-and-resolution-conditioning)
- [Where the test error comes from](#where-the-test-error-comes-from)
- [Pixel size or radar band?](#pixel-size-or-radar-band)
- [Frozen features without training (notebook 14)](#frozen-features-without-training-notebook-14)
- [Caveats in full](#caveats-in-full)
- [Negative result: Cosmos-Reason2-8B](#negative-result-cosmos-reason2-8b)
- [Provenance: what the code fetches](#provenance-what-the-code-fetches)
- [Faster feature copy-back](#faster-feature-copy-back)
- [Run times](#run-times)
- [Source files](#source-files)

## Data

Measured by `01_inventory` on the CaFFe zip:

| item | value |
|---|---|
| files | 2,043 = 681 × (SAR image, zone mask, front mask); 681 bounding-box files |
| images | 8-bit, single channel, heights 405–3,561 px, widths 382–4,476 px |
| image shapes | JAC has one shape; every other glacier has 2–4 |
| zone values | 0 = NA, 64 = stone (rock), 127 = glacier, 254 = ocean with ice mélange |
| front values | 0 / 255 |
| file names | `Glacier_Date_Satellite_SpatialResolutionInMeter_QualityFactor_Orbit(_Modality).png`, quality 1 (best) to 6 (worst) |
| bounding boxes | an `x,y` header and 4 corners in pixel coordinates |

- The zone values were checked against the official code
  (`data_processing/glacier_zones_data.py`). Neither the paper nor PANGAEA
  states them.
- The official code removes predicted front pixels outside the bounding box
  before computing metrics. We do the same.

Images per ground resolution: 6 m 159, 7 m 236, 12 m 43, 17 m 48, 20 m 195.
Quality factor: 1: 334, 2: 141, 3: 111, 4: 54, 5: 36, 6: 5.

Images per glacier and sensor:

| glacier | split | ENVISAT | ERS | PALSAR | RADARSAT-1 | Sentinel-1 | TanDEM-X | TerraSAR-X | all |
|---|---|---|---|---|---|---|---|---|---|
| COL | TEST | 0 | 0 | 0 | 0 | 18 | 47 | 0 | 65 |
| Mapple | TEST | 10 | 2 | 8 | 0 | 15 | 0 | 22 | 57 |
| SI | validation | 27 | 28 | 7 | 27 | 0 | 0 | 32 | 121 |
| Crane | train | 9 | 5 | 7 | 0 | 0 | 0 | 48 | 69 |
| DBE | train | 26 | 16 | 20 | 27 | 0 | 0 | 44 | 133 |
| JAC | train | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 159 |
| Jorum | train | 10 | 3 | 6 | 0 | 15 | 0 | 43 | 77 |

The band of each sensor (X: TerraSAR-X, TanDEM-X; C: ENVISAT, ERS-1/2,
RADARSAT-1, Sentinel-1; L: ALOS PALSAR) is as listed in the CaFFe paper
(Gourmelon et al. 2022, Table 2). The images span 1995–2020 and 6–20 m ground
resolution.

Tiling: 512 px tiles, no overlap, 17,076 tiles in total (train 12,595, SI
2,025, TEST 2,456). SI has 1,582,515 labelled 16 px patches: ENVISAT 112,172,
ERS 111,473, PALSAR 42,608, RADARSAT-1 254,858, TerraSAR-X 1,061,404.

Why 512 px: the DINOv3 paper (arXiv 2508.10104, Table 3) evaluates dense tasks
at 512×512 for patch size 16. TerraMind was also run at 224 px with the same
result (0.351 vs 0.348).

## Methods

**Input.** The grey image is copied to 3 channels. Each model's own
preprocessing is bypassed (C-RADIO uses `make_preprocessor_external()`).
ImageNet statistics are not applied. Normalisations:

- `caffe`: the official CaFFe mean 0.3047 and SD 0.3219, on grey scaled to
  [0, 1], applied to the whole image. Used for the heads, the U-Net and every
  front and TEST result, except SARATR-X.
- `unit`: pixel / 255. What SARATR-X v1 saw in pretraining.
- `per_image`: z-score over the whole image, then tiling. Run for the zone
  probes only.
- `per_tile`: z-score per 512 px tile. Legacy, kept only to reproduce the first
  run.

TerraMind expects two channels (VV, VH) in dB; we copy the single CaFFe channel
into both.

**Linear probes** (zones, arm-1 fronts, arm-2 rows 1–2). One linear layer on the
frozen patch features. AdamW (learning rate 1e-3, weight decay 1e-4), cosine
schedule, 30 epochs, batch 65,536, cross-entropy weighted by inverse class
frequency, seed 0. Trained on a 21 % random sample of the valid training
patches (about 2 M of 9.5 M). The layer sweep used a 10 % sample. Row 2 trains
on 240,000 AnyUp-upsampled pixels (400 from each of 600 tiles). The front
probes predict 4 classes (NA included); their logits are upsampled bilinearly
to pixels.

**Rows 3–6.** 4,000 steps. AdamW (learning rate 1e-3, weight decay 1e-4),
OneCycle schedule with 5 % warm-up. Batch 8 (rows 3, 5, 6) or 4 (row 4).
Cross-entropy with balanced class weights (from the training tiles), padding
ignored. A random horizontal flip per batch (probability 0.5). bf16 autocast.
At most 4,000 training tiles; row 5 used 2,000. Seed 0; row 3 also seeds 1 and
2. No checkpoint was picked on SI. The heads were saved and then loaded, not
retrained, for TEST.

**U-Net.** Our own implementation: 4 levels, base width 32, BatchNorm, leaky
ReLU 0.1, no ASPP, 7.8 M parameters, single-channel input. 48,000 iterations,
batch 16, random 256 px crops drawn in proportion to image area (a crop with
under 10 % labelled pixels is redrawn), random horizontal and vertical flips.
AdamW (learning rate 1e-3, weight decay 1e-4), OneCycle with 5 % warm-up,
class-balanced cross-entropy, bf16 autocast, seed 0, fixed length (the SI
score was logged, not used). Prediction: 512 px sliding window, stride 384,
averaged logits. The 3-class U-Net (zones) and the 4-class U-Net (fronts) are
separate models, one run each.

**3-class patch mIoU.** Classes stone, glacier and ocean; NA and tile padding
are ignored. A 16×16 px patch's true label is the majority class of its
labelled pixels; patches without labelled pixels are skipped. The probes
predict one class per patch token. For the U-Net, the predicted label is the
majority of its pixel predictions over the same labelled pixels. One confusion
matrix is pooled over all patches of the split; the mIoU is the mean of the 3
class IoUs.

**4-class pixel mIoU.** Classes NA, stone, glacier and ocean. One confusion
matrix pooled over all pixels of all images of the split; mean of the 4 class
IoUs.

**Paper-style IoU.** Per image, the IoU of each of the 4 classes; a class absent
from both truth and prediction is skipped. The per-image mean over classes is
then averaged over images. The per-class columns are per-image IoUs averaged
over images.

**MDE.** Our own port of the official evaluation route for zone models:

1. fill gaps inside the ocean and keep only its largest connected part;
2. the front is every ocean pixel with a glacier neighbour;
3. drop front pieces of 750 m or less;
4. mask the front with the image's bounding box, keeping the official clipping;
5. for every predicted front pixel, the distance to the nearest true front
   pixel, and the reverse; pool all distances over all images, times the
   resolution.

Images without a predicted front are left out and counted. The port gave the
same result as the official functions on 25 of 25 random cases; it was tested
against the official code outside this repository. The CaFFe paper describes
MDE as averaged over images; the official code pools the distances, and we
follow the code.

## Arm 1: zones on SI

3-class patch mIoU on SI, balanced linear probe on the last layer. Majority-class
baseline: 0.176.

| model | per_image | caffe | other |
|---|---|---|---|
| C-RADIOv4-H | 0.605 | **0.609** | |
| U-Net from scratch, 48k iterations | | 0.590 (pixel level 0.585) | |
| DINOv3 photo | 0.578 | 0.584 | per_tile 0.578 |
| DINOv3 SAT | 0.572 | 0.571 | |
| U-Net from scratch, 12k iterations (under-trained) | | 0.536 | |
| SARATR-X v1 | | | unit 0.385 |
| TerraMind 1.0 base, S1 | 0.335 | 0.348 | caffe at 224 px 0.351 |

Per class (IoU):

| run | stone | glacier | ocean | mIoU |
|---|---|---|---|---|
| C-RADIOv4-H, caffe | 0.474 | 0.649 | 0.703 | 0.609 |
| U-Net 48k | 0.506 | 0.680 | 0.582 | 0.590 |
| DINOv3 photo, caffe | 0.466 | 0.641 | 0.645 | 0.584 |
| DINOv3 SAT, caffe | 0.446 | 0.633 | 0.635 | 0.571 |
| SARATR-X v1, unit | 0.336 | 0.455 | 0.365 | 0.385 |
| TerraMind, caffe | 0.353 | 0.380 | 0.311 | 0.348 |

Per band and per sensor at resolution (the runs used later):

| run | all | C | L | X | RSAT@12m | RSAT@20m | ENVISAT@20m | ERS@20m |
|---|---|---|---|---|---|---|---|---|
| C-RADIOv4-H, caffe | 0.609 | 0.532 | 0.523 | 0.653 | 0.570 | 0.450 | 0.526 | 0.493 |
| U-Net 48k | 0.590 | 0.592 | 0.432 | 0.595 | 0.558 | 0.556 | 0.660 | 0.600 |
| DINOv3 photo, caffe | 0.584 | 0.528 | 0.544 | 0.618 | 0.566 | 0.418 | 0.512 | 0.499 |
| DINOv3 SAT, caffe | 0.571 | 0.539 | 0.298 | 0.601 | 0.595 | 0.562 | 0.509 | 0.474 |
| SARATR-X v1, unit | 0.385 | 0.375 | 0.220 | 0.396 | 0.403 | 0.267 | 0.368 | 0.354 |
| TerraMind, caffe | 0.348 | 0.288 | 0.218 | 0.384 | 0.333 | 0.155 | 0.287 | 0.288 |

On SI, L is PALSAR at 17 m and X is TerraSAR-X at 7 m.

- C-RADIO's lead over the photo DINOv3 comes mostly from ocean IoU (0.70 vs
  0.64) and from X-band.
- The SAR-pretrained encoders collapse on L-band (0.22). So does the satellite
  DINOv3 (0.30). The photo-pretrained encoders reach 0.52–0.54.
- The photo-pretrained encoders beat the SAR-pretrained ones by 0.20 (photo
  DINOv3 vs SARATR-X) to 0.26 (C-RADIO vs TerraMind).

## Band vs resolution

**The first run** (photo DINOv3, legacy `per_tile` normalisation, SI):

| resolution | mIoU | band | sensor | patches |
|---|---|---|---|---|
| 7 m | 0.6112 | X | TerraSAR-X | 1,061,404 |
| 12 m | 0.5717 | C | RADARSAT-1 | 229,372 |
| 17 m | 0.5193 | L | PALSAR | 42,608 |
| 20 m | 0.4887 | C | ENVISAT, ERS, RADARSAT-1 | 249,131 |

C-band spans two resolutions, so a line through those two points (12 m and
20 m) gives a within-band rate: 0.0104 mIoU per metre. That line predicts
0.5198 at 17 m (observed L-band: 0.5193) and 0.6236 at 7 m (observed X-band:
0.6112). This is two points, one glacier, one encoder and one normalisation. A
16 px patch covers about 112 m of ground at 7 m and about 320 m at 20 m.

**The same check on the final SI runs.** Our arithmetic on the per-resolution
table printed by `04_compare`: a line through each model's own 12 m and 20 m
C-band scores, evaluated at 17 m (L-band) and 7 m (X-band).

| run | 12 m | 20 m | L at 17 m: predicted / observed | X at 7 m: predicted / observed |
|---|---|---|---|---|
| first run, DINOv3 photo, per_tile | 0.572 | 0.489 | 0.520 / 0.519 | 0.624 / 0.611 |
| C-RADIOv4-H, caffe | 0.570 | 0.504 | 0.529 / 0.523 | 0.611 / 0.653 |
| DINOv3 photo, caffe | 0.566 | 0.497 | 0.523 / 0.544 | 0.609 / 0.618 |
| DINOv3 SAT, caffe | 0.595 | 0.497 | 0.534 / 0.298 | 0.656 / 0.601 |
| SARATR-X v1, unit | 0.403 | 0.351 | 0.370 / 0.220 | 0.436 / 0.396 |
| TerraMind, caffe | 0.333 | 0.273 | 0.296 / 0.218 | 0.370 / 0.384 |
| U-Net 48k | 0.558 | 0.626 | 0.601 / 0.432 | 0.516 / 0.595 |

- The L-band match roughly holds for C-RADIO (0.006 off) and less well for the
  photo DINOv3 (0.021 off). It fails for the satellite DINOv3 (0.236 below), SARATR-X (0.150),
  the U-Net (0.169) and TerraMind (0.078).
- The U-Net scores higher at 20 m than at 12 m on SI, the opposite direction.
- The same-sensor contrast RADARSAT-1 at 12 m above RADARSAT-1 at 20 m holds
  for all five encoders (the 20 m group is 6 images). For the U-Net the two are
  equal (0.558 vs 0.556).
- L-band on SI is 7 PALSAR images. So "band differences are resolution
  differences" is a hypothesis these data cannot settle.

**TEST cannot give a clean band claim.** TEST is 94.8 % X-band patches
(X 1,745,537, C 89,096, L 6,693). Every TEST X image is at 7 m, every C image at
20 m, and L is 8 PALSAR images on Mapple only. Within Mapple, 3-class patch mIoU
falls from 7 m to 17 m to 20 m across three bands for the three models below.
For the two probes, ERS at 20 m (2 images) scores above ENVISAT at 20 m, but
still below PALSAR at 17 m.

| Mapple, TEST | TerraSAR-X @ 7 m, X (22 images) | PALSAR @ 17 m, L (8) | ENVISAT @ 20 m, C (10) | ERS @ 20 m, C (2) |
|---|---|---|---|---|
| C-RADIOv4-H probe | 0.757 | 0.688 | 0.456 | 0.545 |
| DINOv3 photo probe | 0.689 | 0.593 | 0.417 | 0.562 |
| U-Net, 3-class | 0.789 | 0.621 | 0.548 | 0.469 |

- On COL, Sentinel-1 at 20 m scores 0.41–0.45 and TanDEM-X at 7 m 0.58–0.61.
- L-band scores high on TEST (Mapple, 0.59–0.69) and low on SI (0.52–0.54 for
  the photo models). The band label tracks glacier and resolution.
- Band is also confounded with glacier in training: JAC (159 of 438 training
  images, 36 %) is all TerraSAR-X.

## Arm 1: fronts on SI

Last-layer 4-class linear probes (NA is the 4th class), logits upsampled
bilinearly, then the official front route. 121 SI images.

| model | MDE (m) | no front | 4-class pixel mIoU |
|---|---|---|---|
| U-Net, 4-class, 48k | **1,732** | 0 | 0.70 |
| C-RADIOv4-H probe | 1,816 | 5 | 0.70 |
| DINOv3 SAT probe | 1,852 | 6 | 0.7 (printed to one decimal only) |
| DINOv3 photo probe | 2,018 | 10 | 0.64 |
| TerraMind probe | 2,563 | 5 | 0.44 |
| SARATR-X probe (unit) | 2,900 | 0 | 0.46 |

MDE per sensor at resolution on SI (m):

| model | ENVISAT@20m | ERS@20m | PALSAR@17m | RSAT@12m | RSAT@20m | TSX@7m |
|---|---|---|---|---|---|---|
| U-Net | 1,222 | 1,110 | 2,742 | 1,741 | 1,714 | 1,864 |
| C-RADIOv4-H probe | 2,060 | 1,940 | 2,249 | 1,450 | 3,189 | 1,810 |
| DINOv3 SAT probe | 2,095 | 2,006 | 3,022 | 1,352 | 1,874 | 1,905 |
| DINOv3 photo probe | 1,812 | 1,638 | 2,126 | 1,452 | 3,345 | 2,255 |
| TerraMind probe | 2,771 | 2,415 | 5,243 | 1,851 | 4,589 | 2,657 |
| SARATR-X probe | 3,181 | 2,033 | 4,482 | 3,181 | 5,041 | 2,772 |

- The C-RADIO and photo DINOv3 probes do best on RADARSAT-1 at 12 m (about
  1,450 m) and worst on RADARSAT-1 at 20 m (3,189 and 3,345 m, 6 images).
- The satellite DINOv3 probe does not collapse on RADARSAT-1 at 20 m (1,874 m).
  Its worst group is PALSAR at 17 m (3,022 m).
- The U-Net does best on ENVISAT and ERS at 20 m.

## Arm 1: TEST

Run once on 2026-09-24 for five models. The satellite DINOv3 probe was added
and scored once on 2026-09-25, with the same last-layer probe.

3-class patch mIoU:

| model | all | COL | Mapple |
|---|---|---|---|
| DINOv3 photo probe | **0.610** | 0.596 | 0.664 |
| C-RADIOv4-H probe | 0.602 | 0.584 | 0.733 |
| U-Net, 3-class, 48k | 0.594 | 0.573 | 0.764 |
| DINOv3 SAT probe | 0.586 | 0.572 | 0.667 |
| TerraMind probe | 0.385 | 0.379 | 0.418 |
| SARATR-X probe | 0.363 | 0.352 | 0.441 |

Paper-style zone IoU (per image, 4 classes including NA, averaged over images):

| model | NA | stone | glacier | ocean | mean |
|---|---|---|---|---|---|
| U-Net, 4-class, 48k | 0.903 | 0.626 | 0.640 | 0.537 | **0.677** |
| C-RADIOv4-H probe | 0.760 | 0.591 | 0.629 | 0.566 | 0.636 |
| DINOv3 photo probe | 0.761 | 0.592 | 0.604 | 0.494 | 0.613 |
| DINOv3 SAT probe | 0.773 | 0.568 | 0.558 | 0.479 | 0.595 |
| SARATR-X probe | 0.741 | 0.464 | 0.426 | 0.122 | 0.438 |
| TerraMind probe | 0.547 | 0.480 | 0.401 | 0.127 | 0.389 |
| published CaFFe baseline | 0.93 | 0.68 | 0.74 | 0.82 | 0.79 |

The probes match the U-Net on 3-class patch mIoU but trail it on this 4-class
metric, most of all on NA (0.76 vs 0.90). The 16 px grid blurs the no-data
boundary.

MDE (m):

| model | all | COL | Mapple | no front / 122 |
|---|---|---|---|---|
| U-Net, 4-class, 48k | **979** | 1,080 | 474 | 3 |
| C-RADIOv4-H probe | 2,089 | 2,349 | 505 | 37 |
| DINOv3 photo probe | 2,199 | 2,302 | 823 | 48 |
| TerraMind probe | 2,414 | 3,095 | 908 | 55 |
| SARATR-X probe | 2,796 | 3,313 | 921 | 70 |
| DINOv3 SAT probe | 2,805 | 3,563 | 526 | 60 |
| published CaFFe baseline | 753 ± 76 | 840 ± 84 | 150 ± 24 | 1 ± 1 |

MDE per sensor at resolution on TEST (m):

| model | ENVISAT@20m | ERS@20m | PALSAR@17m | S1@20m | TDX@7m | TSX@7m |
|---|---|---|---|---|---|---|
| U-Net | 721 | 627 | 712 | 2,913 | 788 | 432 |
| C-RADIOv4-H probe | 875 | 243 | 498 | 3,586 | 2,048 | 566 |
| DINOv3 photo probe | 2,288 | 327 | 571 | 3,237 | 2,112 | 892 |
| DINOv3 SAT probe | 1,031 | 791 | 633 | 4,262 | 3,226 | 402 |
| TerraMind probe | 1,377 | 1,776 | 985 | 5,624 | 2,696 | 884 |
| SARATR-X probe | no front on any image | 785 | 1,538 | 6,165 | 2,946 | 899 |

- The probes draw no front on 30–57 % of TEST images, so their MDEs are over
  easier subsets.
- Per sensor, keep those subsets in mind. On PALSAR at 17 m (8 images) and
  ERS at 20 m (2 images) the C-RADIO and photo probes have a lower MDE than the
  U-Net, over the images where they drew a front.
- Everyone is poor on Sentinel-1 at 20 m. TanDEM-X is COL only; there the
  probes are far behind the U-Net.
- The satellite DINOv3 probe has the lowest MDE of all models on TerraSAR-X at
  7 m (402 m), over the images where it drew a front, and the highest on COL.

## Published references

As recorded from the sources, not re-run. All on the 122 TEST images.

| model | source | zone IoU, 4 classes incl. NA | MDE (m) | no front / 122 |
|---|---|---|---|---|
| CaFFe baseline (U-Net + ASPP, 256 px, 5 runs) | CaFFe paper, Table 5; all-image MDE from the AMD-HookNet paper, Table III | 0.79 ± 0.02 (about 0.75 without NA) | 753 ± 76; COL 840 ± 84, Mapple 150 ± 24 | 1 ± 1 |
| AMD-HookNet (TGRS 2023) | AMD-HookNet paper, Table III; IoU from the AMD-HookNet++ paper, Table III | 0.744 ± 0.010 | 438 ± 22 | 0 ± 1 |
| HookFormer (TGRS 2024) | AMD-HookNet++ paper, Table III | 0.755 ± 0.003 | **353 ± 16** | 0 |
| AMD-HookNet++ (TGRS 2026) | AMD-HookNet++ paper, Table III | **0.782 ± 0.004** | 367 ± 30 | 0 |

- HookFormer has the best published MDE in this comparison. AMD-HookNet++
  leads on IoU, and its paper calls its own MDE 4.0 % worse than HookFormer's.
- The AMD-HookNet++ paper prints IoU in per cent (78.2 ± 0.4); this table
  shows it as a fraction. Its per-class table (Table IV) shows that 78.2 is the
  unweighted mean of the four zone classes including NA: (97.5 + 57.6 + 72.3 +
  85.5) / 4. Without NA the same classes give 71.8 (our arithmetic).
- The no-front rule is the same as ours. The official CaFFe code leaves an
  image without a predicted front out of the MDE and counts it separately; the
  CaFFe and AMD-HookNet papers report that count as ∅. AMD-HookNet uses the
  same definition and the CaFFe evaluation tool.
- The official MDE pools all front-pixel distances over all images, each times
  its image's resolution. It is not a mean of per-image MDEs. Our code does the
  same (see [Methods](#methods)).
- Against these numbers our best MDE on all TEST images, 979 m (the C-RADIO
  decoder over three seeds, and the U-Net), is far behind.

## Arm 2: layer sweep

3-class patch mIoU on SI. Probes on a 10 % patch subsample (about 950 k
patches, about 450 steps). Only the ranking is used; these values sit 0.02–0.04
below the full runs. The taps are at 1/4, 1/2, 3/4 and 7/8 depth and the last
block, with the final norm applied.

| DINOv3 photo, layer | 6 | 12 | **18** | 21 | 24 |
|---|---|---|---|---|---|
| all | 0.251 | 0.539 | **0.585** | 0.581 | 0.548 |
| C / L / X | 0.258 / 0.086 / 0.254 | 0.481 / 0.316 / 0.579 | 0.550 / 0.488 / 0.609 | 0.535 / 0.494 / 0.608 | 0.509 / 0.510 / 0.574 |

| C-RADIOv4-H, layer | 8 | 16 | 24 | 28 | **32** |
|---|---|---|---|---|---|
| all | 0.352 | 0.496 | 0.442 | 0.426 | **0.587** |
| C / L / X | 0.325 / 0.164 / 0.374 | 0.460 / 0.357 / 0.523 | 0.406 / 0.359 / 0.468 | 0.392 / 0.356 / 0.449 | 0.519 / 0.501 / 0.628 |

| DINOv3 SAT, layer | 6 | 12 | 18 | **21** | 24 |
|---|---|---|---|---|---|
| all | 0.243 | 0.430 | 0.532 | **0.567** | 0.540 |
| C / L / X | 0.218 / 0.110 / 0.252 | 0.356 / 0.126 / 0.480 | 0.507 / 0.256 / 0.559 | 0.540 / 0.287 / 0.595 | 0.527 / 0.266 / 0.561 |

By the rule fixed in advance (best SI score) the heads use photo layer 18,
C-RADIO layer 32 and satellite layer 21. From layer 12 on, the satellite
model's L-band is far below the photo model's (0.287 vs 0.494 at layer 21).

## Arm 2: heads and parameter counts

| row | head | DINOv3 (SAT / photo), 1024-d | C-RADIOv4-H, 1280-d |
|---|---|---|---|
| 1 | linear probe on the 16 px grid, logits upsampled bilinearly | 4,100 | 5,124 |
| 2 | AnyUp → per-pixel features → linear probe (240 k pixels from 600 tiles) | 4,100 | 5,124 |
| 3 | `GridDecoder`: 1×1 reduction, four ×2 upsampling stages, image skip at 5 scales | 544,388 | 568,964 |
| 4 | `PixelHead`: 1×1 reduction → AnyUp at full resolution → image map → 2 convs | 114,452 (printed 166,628) | 130,836 (printed 183,012) |
| 5 | row 3 on 4 concatenated layers (DINOv3 6+12+18+24, C-RADIO 8+16+24+32) | 839,300 (SAT) | 937,604 |
| 6 | `AnyUpDecoder`: AnyUp to 1/4 resolution → the last 3 stages of row 3 | 239,652 (printed 281,348) | 264,228 (printed 305,924) |

- "Printed" is the count in the run logs and the summary CSVs on Drive. The
  heads of the reported runs built all five image-stem levels, but row 4 used
  only one and row 6 three: 52,176 (row 4) and 41,696 (row 6) parameters never
  got a gradient. Outputs were not affected. The current code builds only the
  levels a head uses and prints the lower count.
- AnyUp itself (0.88 M parameters) is frozen. It is used through a value swap:
  upsampling the 4 probe logits gives the same result as probing the upsampled
  features (max difference 1e-5), with about 1/256 of the memory.
- For scale: the U-Net has 7.8 M parameters.

## Arm 2: SI

MDE (m), no-front count, 4-class pixel mIoU and minutes to train and score.

| row | C-RADIOv4-H (L32) | DINOv3 SAT (L21) | DINOv3 photo (L18, replaced; SI only) |
|---|---|---|---|
| 1 | 1,791 (6), 0.681, 2.3 min | 1,853 (2), 0.694, 2.3 min | 1,776 (1), 0.693 |
| 2 | 1,534 (5), 0.612, 8.9 min | 1,849 (0), 0.678, 8.8 min | 2,103 (1), 0.632, 8.7 min |
| 3, seed 0 | 1,273 (0), 0.743, 6.8 min | 1,301 (0), 0.760, 6.2 min | 1,236 (0), 0.766, 5.6 min |
| 3, seed 1 | 1,180 (0), 0.750, 6.2 min | 1,345 (0), 0.760, 5.6 min | not run |
| 3, seed 2 | 1,280 (0), 0.747, 6.2 min | 1,343 (0), 0.769, 5.6 min | not run |
| **3, mean ± SD** | **1,244 ± 56** | **1,329 ± 25** | |
| 4 | 1,445 (0), 0.721, 33.9 min | 1,402 (0), 0.757, 33.1 min | 1,359 (0), 0.757, 33.0 min |
| 5 (2,000 tiles) | 1,345 (0), 0.738, 16.4 min | 1,465 (0), 0.751, 13.9 min | not run |
| 6 | 1,375 (0), 0.738, 16.7 min | 1,384 (0), 0.762, 15.9 min | 1,305 (0), 0.754, 15.9 min |

References on SI: U-Net 1,732 m (0 no front); last-layer probes C-RADIO
1,816 m, photo DINOv3 2,018 m, satellite DINOv3 1,852 m.

- Rows 3–6 drew a front on every SI image.
- Row 3 was the best head for every encoder on SI. No AnyUp variant and no
  4-layer input beat it.
- Row 5 reached the lowest training loss (0.080 C-RADIO, 0.099 satellite
  DINOv3) and did not do better. Row 5 also trained on 2,000 tiles instead of
  4,000, so the low loss cannot be put down to the extra layers alone.
- On SI, AnyUp helped the linear probe only for C-RADIO (1,791 → 1,534 m).
  Satellite DINOv3: 1,853 → 1,849 m. Photo DINOv3: 1,776 → 2,103 m.
- The satellite DINOv3's row-3 seeds range over 44 m (1,301–1,345), SD 25 m.
- The SI lead of row 3 over the U-Net (about 490 m for C-RADIO, about 400 m for
  the satellite DINOv3) did not carry over to TEST.

## Arm 2: TEST

Run once on 2026-09-25, loading the saved heads; nothing was trained or chosen
on TEST. 122 images. MDE (m) with no-front count, COL / Mapple MDE, and 4-class
pixel mIoU.

| row | C-RADIOv4-H | COL / Mapple | mIoU4 | DINOv3 SAT | COL / Mapple | mIoU4 |
|---|---|---|---|---|---|---|
| 1 | 2,115 (38) | 2,397 / 469 | 0.678 | 2,660 (42) | 3,268 / 535 | 0.677 |
| 2 | 1,915 (21) | 2,116 / 603 | 0.646 | 2,083 (36) | 2,390 / 703 | 0.662 |
| 3, seed 0 | 945 (7) | 1,029 / 466 | 0.686 | 1,021 (2) | 1,114 / 570 | 0.694 |
| 3, seed 1 | 1,003 (9) | 1,086 / 502 | 0.680 | 1,508 (11) | 1,729 / 545 | 0.709 |
| 3, seed 2 | 990 (9) | 1,085 / 424 | 0.683 | 985 (2) | 1,090 / 495 | 0.703 |
| **3, mean ± SD** | **979 ± 31** | 1,067 / 464 | | **1,171 ± 292** | 1,311 / 537 | |
| 4 | 1,316 (17) | 1,437 / 453 | 0.691 | 1,226 (7) | 1,361 / 578 | 0.705 |
| 5 | 1,005 (2) | 1,142 / 324 | 0.692 | 1,249 (7) | 1,442 / 423 | 0.708 |
| 6 | 1,225 (12) | 1,356 / 480 | 0.688 | 1,076 (1) | 1,217 / 457 | 0.698 |
| U-Net, one run | 979 (3) | 1,080 / 474 | | | | |

1. **Matched, not beaten.** The C-RADIO decoder (row 3) averages 979 ± 31 m,
   the same as the U-Net's 979 m, per glacier too. It misses more fronts (7–9
   vs 3).
2. **The satellite DINOv3 decoder is worse than the U-Net** on TEST:
   1,171 ± 292 m. One seed collapsed to 1,508 m with 11 missed fronts. With one
   run per head this could have been misreported by up to about 500 m.
3. **Seed spread:** it showed up on TEST for the satellite DINOv3 only (SD 25 m
   on SI, 292 m on TEST). C-RADIO's SD was larger on SI (56 m) than on TEST
   (31 m).
4. **The decoder closes most of the probe's front gap.** It about halves the
   probe's MDE (C-RADIO 2,115 → 979 m, satellite DINOv3 2,660 → 1,171 m) and cuts
   missed fronts from 38–42 to 2–11.
5. **AnyUp helps a linear probe on TEST for both encoders** (2,115 → 1,915 m;
   2,660 → 2,083 m). Under a learned head it never beats C-RADIO's decoder. The
   satellite DINOv3's row 6 (1,076 m) lies inside the range of its row-3 seeds.
6. **C-RADIO vs satellite DINOv3 under learned heads is a 2–2 split.** C-RADIO
   is better in rows 3 (979 vs 1,171) and 5 (1,005 vs 1,249). The satellite
   DINOv3 is better in rows 4 (1,226 vs 1,316) and 6 (1,076 vs 1,225). Per seed
   in row 3, satellite seed 2 (985 m) beats C-RADIO seed 2 (990 m).
7. **Row 5 is mixed.** C-RADIO: 1,005 m with 2 misses and the best Mapple
   result of any head (324 m). Satellite DINOv3: 1,249 m.
8. **Everything is far behind the published CaFFe models.** The CaFFe
   baseline reaches 753 ± 76 m on all 122 TEST images (COL 840 m, Mapple
   150 m); the best published MDE, HookFormer's, is 353 ± 16 m (see
   [Published references](#published-references)).

## Photo vs satellite DINOv3

Same ViT-L/16, same recipe, last-layer probe, TEST:

| probe | 3-class patch mIoU (COL / Mapple) | paper-style IoU | MDE (m) | no front / 122 | COL / Mapple MDE | TSX@7m MDE |
|---|---|---|---|---|---|---|
| DINOv3 photo (LVD-1689M) | **0.610** (0.596 / 0.664) | **0.613** | **2,199** | **48** | 2,302 / 823 | 892 |
| DINOv3 SAT (SAT-493M) | 0.586 (0.572 / 0.667) | 0.595 | 2,805 | 60 | 3,563 / **526** | **402** |

- Overall the photo model is ahead on zones (+0.024) and fronts (606 m lower
  MDE, 12 fewer missed fronts).
- The satellite model is ahead on Mapple (526 vs 823 m) and on TerraSAR-X at
  7 m (402 vs 892 m).
- The two MDEs are over different image subsets (60 vs 48 images without a
  front).
- On SI the satellite probe had looked better on fronts (1,852 vs 2,018 m).
  That did not carry over to TEST.
- The photo model's learned heads were scored on SI only, so there is no
  head-level TEST comparison.

## AnyUp precision check

All head training and scoring ran under bf16 autocast. In AnyUp this puts the
pixel coordinates and RoPE angles in bf16. At 512 px only 385 of 512
coordinates stay distinct (max error 0.0029). The check was decided after the
TEST run and uses SI only.

- Method: the saved row 2, 4 and 6 heads, one tile from each of 12 SI images
  spread over the 6 sensor-at-resolution groups. Three variants: fp32, bf16 as
  run, bf16 with fp32 positions.
- Rule fixed before the numbers: re-run the AnyUp rows only if more than 1 % of
  valid pixels change class, or the feature error exceeds 10 % of AnyUp's own
  effect (fp32 AnyUp vs bilinear).

| encoder | variant | feature error | AnyUp effect | pixels changing class, rows 2 / 4 / 6 |
|---|---|---|---|---|
| DINOv3 SAT L21 | bf16 as run | 0.0100 | 0.407 | 0.37 / 0.15 / 0.13 % |
| DINOv3 SAT L21 | bf16 + fp32 positions | 0.0092 | | 0.34 / 0.13 / 0.11 % |
| C-RADIOv4-H L32 | bf16 as run | 0.0077 | 0.309 | 0.20 / 0.11 / 0.13 % |
| C-RADIOv4-H L32 | bf16 + fp32 positions | 0.0068 | | 0.18 / 0.09 / 0.12 % |

As run, bf16 moves AnyUp's output by 2.4–2.5 % of AnyUp's own effect and flips
at most 0.37 % of pixels. Both are far below the limits, so the AnyUp rows were
not re-run. Keeping positions in fp32 removes only about 10 % of the error.
This covers inference only. Whether bf16 changed what rows 2, 4 and 6 learned
in training was not measured.

## Approach A: band and resolution conditioning

Exploratory, added after arm 2's TEST run; SI only (`notebooks/09_conditioning.ipynb`).
Row 3's decoder ("plain") against the same decoder with FiLM conditioning on
each image's band (X / C / L) and resolution ("film"). FiLM's last layer starts
at zero, so both start identical. Both use the same frozen features (satellite
DINOv3 layer 21, C-RADIO layer 32), the same 4,000 training tiles (a uniform
sample of 12,595) in the same order, seeds 0, 1 and 2, and row 3's recipe.

| SI, MDE in m | plain, seeds 0 / 1 / 2 | plain mean ± SD | film, seeds 0 / 1 / 2 | film mean ± SD | missed fronts, plain / film |
|---|---|---|---|---|---|
| DINOv3 SAT L21 | 1,453 / 1,390 / 1,458 | 1,434 ± 38 | 1,293 / 1,336 / 1,494 | 1,374 ± 106 | 0 / 0 |
| C-RADIOv4-H L32 | 1,627 / 1,571 / 1,545 | 1,581 ± 42 | 1,581 / 1,418 / 1,523 | 1,507 ± 83 | 1 / 0 |

Rule fixed before the run: film helps if its mean is lower than plain's by more
than the larger of the two SDs, with no more missed fronts. **Verdict: no clear
effect for either encoder** (difference 59 m within 106 m; 74 m within 83 m).
The 4-class pixel mIoU is unchanged (0.756 vs 0.751; 0.705 vs 0.706).

Per sensor at resolution, mean over seeds (MDE in m; not covered by the rule):

| SI | ENVISAT @ 20 m | ERS @ 20 m | PALSAR @ 17 m | RSAT @ 12 m | RSAT @ 20 m | TSX @ 7 m |
|---|---|---|---|---|---|---|
| C-RADIO plain | 1,567 | 1,555 | 1,780 | 1,461 | 1,069 | 1,624 |
| C-RADIO film | 1,245 | 1,368 | 986 | 1,446 | 939 | 1,644 |
| DINOv3 SAT plain | 1,622 | 1,467 | 2,828 | 1,301 | 1,553 | 1,356 |
| DINOv3 SAT film | 1,523 | 1,306 | 2,401 | 1,317 | 1,704 | 1,302 |

The largest gains are on L-band (PALSAR, 7 SI images), then on ENVISAT and ERS
at 20 m (27 and 28 images). RADARSAT-1 at 20 m (6 images) improved for C-RADIO
but got worse for the satellite DINOv3 (1,553 to 1,704 m). Not covered by the
rule; a hint, not a result.

**The tile sample moves the result more than the seed.** The plain decoder
here is row 3 retrained on a uniform tile sample with seeded initial weights.
The reported row 3 (tilted sample) scored 1,301 / 1,345 / 1,343 m (DINOv3 SAT,
mean 1,329) and 1,273 / 1,180 / 1,280 m (C-RADIO, mean 1,244) on SI. The
retrained means are about 100 m and 340 m worse, several times the seed SDs. The
tilted sample kept tiles early in file order more often; which glaciers that
favoured has not been measured. Against the U-Net's 1,732 m on SI, the
retrained decoders are still ahead (by about 300 m and 150 m). The TEST run of
arm 2 used the reported, tilted-sample heads and was not repeated.

Run time: 43 min (DINOv3 SAT) and 49 min (C-RADIO) for one extraction plus six
trainings each, on code from before the training speed-ups.

## Where the test error comes from

Added on 2026-09-27, after every TEST run (`notebooks/11_predictions.ipynb`,
part 1). It is a description of the test set after the fact, computed from the
saved per-image results. Nothing was retrained or scored again, and it changes
no choice.

By eye, in the image viewer (notebook 10), the worst TEST images of every
model, the U-Net included, were Columbia's Sentinel-1 images at 20 m. There are
18 of them, 15 % of the 122 TEST images. Over all 22 saved TEST runs (arm 1 and
arm 2) they hold 7–14 % of the pooled front distances but 15–38 % of the summed
error. The pooled MDE was recomputed from the saved per-image distances; all 22
runs matched their saved MDE within 0.5 m.

| TEST, MDE in m | all 122 images | without the 18 COL Sentinel-1 images | the 18 only | share of the summed error | share of the distances | no front: all / the 18 |
|---|---|---|---|---|---|---|
| U-Net, 4-class | 979 | 730 | 3,447 | 0.323 | 0.092 | 3 / 0 |
| C-RADIO decoder (row 3), seed 0 | 945 | 671 | 2,914 | 0.376 | 0.122 | 7 / 0 |
| C-RADIO decoder (row 3), seed 1 | 1,003 | 735 | 2,808 | 0.362 | 0.129 | 9 / 0 |
| C-RADIO decoder (row 3), seed 2 | 990 | 732 | 2,816 | 0.352 | 0.124 | 9 / 0 |
| DINOv3 SAT decoder (row 3), seed 0 | 1,021 | 809 | 2,447 | 0.310 | 0.129 | 2 / 0 |
| DINOv3 SAT decoder (row 3), seed 1 | 1,508 | 1,270 | 3,294 | 0.257 | 0.118 | 11 / 3 |
| DINOv3 SAT decoder (row 3), seed 2 | 985 | 773 | 2,339 | 0.321 | 0.135 | 2 / 0 |
| C-RADIO probe (arm 1) | 2,089 | 1,809 | 4,808 | 0.215 | 0.094 | 37 / 7 |

- **The key verdict holds with and without these images.** With them, the
  C-RADIO decoder gives 979 ± 31 m and the U-Net 979 m. Without them, the
  decoder's seeds give 671, 735 and 732 m and the U-Net 730 m. A match, not a
  win, either way.
- MDE leaves out images without a front. The C-RADIO probe drew no front on 7
  of the 18 images and the satellite DINOv3 decoder's seed 1 on 3, so their
  "the 18 only" MDE is over fewer images.
- Why these images are hard was not measured.

## Pixel size or radar band?

Added on 2026-09-27 (experiment B in `notebooks/12_resolution.ipynb`, then
`notebooks/13_bands.ipynb`). SI only; the test glaciers were not touched.
Exploratory and after every TEST run: it changes no earlier choice. Nothing was
retrained. The saved heads and the U-Net were run, unchanged, on SI images
shrunk to coarser pixels.

Why: in CaFFe each band comes with its own pixel sizes, satellites and years,
and on the test glaciers with its own glacier, so the per-band tables cannot
separate them (see [Band vs resolution](#band-vs-resolution)).

**Written down before any number.**

- Models, nine rows: the saved 4-class U-Net; C-RADIO (layer 32) and the
  satellite DINOv3 (layer 21), each with the row-1 linear probe and the row-3
  decoder, seeds 0, 1 and 2.
- Shrinking: the radar image by area averaging, the zone labels by nearest
  neighbour, the true front by area averaging with every pixel above 0 kept,
  then thinned to 1 px. The bounding box is scaled.
- Control: each head, rerun on the original images, must reproduce its saved SI
  result exactly (front and per-image zone IoU), or the run stops. Amendment,
  also before any number: the saved U-Net SI file came from the model in memory
  right after training and cannot be reproduced bit for bit by a freshly loaded
  U-Net. So every U-Net number here is rerun on one fresh-load path, and its
  comparison with the saved file is information only.
- Experiment B (notebook 12): the 32 SI TerraSAR-X images (X-band, 7 m) are
  shrunk to 20 m. Read-out: pooled MDE on X-band at 7 m (X@7), on the same
  images at 20 m (X@20), and on the SI C-band images that really are 20 m
  (C@20: ENVISAT, ERS and RADARSAT-1, 61 images). Rule:
  r = (X@20 − X@7) / (C@20 − X@7). r ≥ 0.5: pixel size explains the X-vs-C
  difference for that model; r < 0.5: it does not; |C@20 − X@7| < 100 m:
  nothing to explain.
- Notebook 13: the 21 SI RADARSAT-1 images at 12 m are shrunk to 20 m. Same
  band, same satellite, and C-band at 20 m is in training. Rule: pixel size
  hurts a model if its pooled MDE on the shrunk copies is at least 100 m worse
  than on the 12 m originals. The r rule against the real C@20 is reported too.
  L-band was not shrunk: 17 m to 20 m is too small a step.
- Also in notebook 13: pooled MDE per glacier, band, satellite and pixel size
  from the saved per-image results, SI and TEST. Descriptive only.

**Controls.** Every head reproduced its saved SI result exactly, in both
notebooks. The U-Net (information only) was identical on 24 of 32 images,
pooled MDE +0.0 m against the saved file (notebook 12), and on 8 of 21 images,
−0.2 m (notebook 13).

**Experiment B: X-band, 7 m → 20 m.** SI, pooled MDE in m.

| model | X@7 | X@20, shrunk | C@20, real | r | verdict by the rule | median change per image |
|---|---|---|---|---|---|---|
| C-RADIO probe (row 1) | 1,821 | 2,497 | 2,060 | 2.83 | pixel size explains it | +401 |
| C-RADIO decoder, seed 0 | 1,158 | 1,710 | 1,497 | 1.63 | pixel size explains it | +620 |
| C-RADIO decoder, seed 1 | 1,092 | 1,289 | 1,302 | 0.94 | pixel size explains it | +228 |
| C-RADIO decoder, seed 2 | 1,233 | 1,373 | 1,409 | 0.79 | pixel size explains it | +166 |
| DINOv3 SAT probe (row 1) | 1,901 | 5,367 (5 no front) | 1,893 | – | nothing to explain (< 100 m) | +3,469 |
| DINOv3 SAT decoder, seed 0 | 1,247 | 2,138 | 1,422 | 5.08 | pixel size explains it | +1,028 |
| DINOv3 SAT decoder, seed 1 | 1,313 | 1,875 | 1,386 | – | nothing to explain (< 100 m) | +804 |
| DINOv3 SAT decoder, seed 2 | 1,282 | 1,842 | 1,495 | 2.63 | pixel size explains it | +729 |
| U-Net, 4-class, all rerun | 1,864 | 2,450 (1 no front) | 1,223 | −0.91 | pixel size does not explain it | +310 |

r above 1 means the shrunk X-band images end up worse than the real C-band
20 m images. No row missed a front at 7 m.

**Within C-band: RADARSAT-1, 12 m → 20 m.** SI, pooled MDE in m.

| model | 12 m | 20 m, shrunk | change | verdict by the rule | r against the real C@20 | median change per image |
|---|---|---|---|---|---|---|
| C-RADIO probe (row 1) | 1,446 | 1,771 | +325 | hurts | 0.53 | +146 |
| C-RADIO decoder, seed 0 | 1,309 | 1,523 | +214 | hurts | 1.14 | +126 |
| C-RADIO decoder, seed 1 | 1,330 | 1,385 | +55 | no clear change | – (under 100 m to C@20) | −78 |
| C-RADIO decoder, seed 2 | 1,153 | 1,223 | +69 | no clear change | 0.27 | −53 |
| DINOv3 SAT probe (row 1) | 1,514 | 1,870 | +356 | hurts | 0.94 | +692 |
| DINOv3 SAT decoder, seed 0 | 1,069 | 1,377 | +307 | hurts | 0.87 | +266 |
| DINOv3 SAT decoder, seed 1 | 1,168 | 1,525 | +357 | hurts | 1.63 | +234 |
| DINOv3 SAT decoder, seed 2 | 1,125 | 1,430 | +305 | hurts | 0.82 | +296 |
| U-Net, 4-class, all rerun | 1,741 | 1,755 | +14 | no clear change | −0.03 | +25 |

No row missed a front on the RADARSAT-1 images, before or after shrinking.

**Every band, from the saved results (descriptive).** Pooled MDE in m. For the
decoders, the mean of the three seeds' pooled MDEs. The C-RADIO probe here is
arm 2's row 1, not the arm-1 probe of the previous section.

| split, glacier | band, satellite, pixel size | images | C-RADIO probe | C-RADIO decoder | DINOv3 SAT probe | DINOv3 SAT decoder | U-Net |
|---|---|---|---|---|---|---|---|
| SI | C, RADARSAT-1, 12 m | 21 | 1,446 | 1,264 | 1,514 | 1,121 | 1,741 |
| SI | C, ENVISAT, 20 m | 27 | 2,002 | 1,394 | 2,049 | 1,424 | 1,222 |
| SI | C, ERS, 20 m | 28 | 1,831 | 1,471 | 1,582 | 1,426 | 1,110 |
| SI | C, RADARSAT-1, 20 m | 6 | 3,965 | 1,140 | 3,020 | 1,562 | 1,714 |
| SI | C, all, 20 m | 61 | 2,060 | 1,403 | 1,893 | 1,434 | 1,223 |
| SI | L, PALSAR, 17 m | 7 | 1,458 | 1,544 | 3,604 | 2,530 | 2,742 |
| SI | X, TerraSAR-X, 7 m | 32 | 1,821 | 1,161 | 1,901 | 1,280 | 1,864 |
| TEST, COL | C, Sentinel-1, 20 m | 18 | 4,997 | 2,846 | 5,269 | 2,694 | 3,447 |
| TEST, COL | X, TanDEM-X, 7 m | 47 | 2,102 | 762 | 2,912 | 1,059 | 788 |
| TEST, Mapple | C, ENVISAT, 20 m | 10 | 724 | 584 | 912 | 795 | 721 |
| TEST, Mapple | C, ERS, 20 m | 2 | 245 | 371 | 712 | 512 | 627 |
| TEST, Mapple | C, Sentinel-1, 20 m | 15 | 255 | 362 | 489 | 237 | 198 |
| TEST, Mapple | C, all, 20 m | 27 | 408 | 442 | 665 | 496 | 490 |
| TEST, Mapple | L, PALSAR, 17 m | 8 | 496 | 376 | 759 | 664 | 712 |
| TEST, Mapple | X, TerraSAR-X, 7 m | 22 | 523 | 497 | 428 | 541 | 432 |

**Reading.**

1. **Coarser pixels hurt every model from 7 m to 20 m.** All nine rows got
   worse on the shrunk X-band images (median change per image +166 to
   +3,469 m).
2. **On SI, the X-band advantage over C-band disappears at equal pixel size.**
   By the rule, pixel size explains the X-vs-C difference for 6 of 9 rows (all
   four C-RADIO rows, satellite DINOv3 decoder seeds 0 and 2). 2 rows had no
   difference to explain (the satellite DINOv3 probe and decoder seed 1).
3. **The U-Net is the exception.** At 7 m it was already worse on X-band
   (1,864 m) than on the real C-band 20 m images (1,223 m). Pixel size cannot
   explain that (r = −0.91).
4. **Within C-band, 12 m → 20 m hurts 6 of 9 rows**: both probes, every
   satellite DINOv3 decoder seed and C-RADIO decoder seed 0. C-RADIO decoder
   seeds 1 and 2 (+55 and +69 m) and the U-Net (+14 m) show no clear change,
   although the U-Net did get worse from 7 m to 20 m in experiment B.
5. **No evidence that X-band itself helps.**
6. **No consistent band ranking in the saved results.** On SI, L-band is the
   worst group for the satellite DINOv3 probe (3,604 m) and decoder
   (2,530 m), the U-Net (2,742 m) and the C-RADIO decoder (1,544 m). The
   C-RADIO probe scores 1,458 m there, second only to its RADARSAT-1 12 m
   group (1,446 m). On Mapple, L-band is the best band for the C-RADIO decoder
   (376 m) and the worst for the satellite DINOv3 probe and decoder and the
   U-Net. Glacier and satellite together differ far more than bands: on
   Columbia's Sentinel-1 images the five models score 2,694–5,269 m, on
   Mapple's 198–489 m, both C-band at 20 m.
7. **L-band is only 7 SI images and 8 TEST images** (all PALSAR, the TEST ones
   all on Mapple).

**Caveats.**

- A shrunk image is not a real 20 m product: speckle and processing differ.
- X-band at 20 m is a combination no model saw in training. So "worse than
  the real C-band 20 m images" (r above 1) may be unfamiliarity, not band
  physics. C-band at 20 m is in training, which is why notebook 13 shrinks
  C-band images too.
- At equal pixel size the remaining difference still mixes band with
  satellite, years and scene.
- SI only: one glacier, 32 X-band and 21 RADARSAT-1 images.
- Exploratory and after every TEST run; it changes no earlier choice.

Tables: `MyDrive/sar-transfer/resolution_test/` (`restest_table.csv`, called
`restest_X7_table.csv` by the current code; `restest_C12_table.csv`;
`restest_bands_summary.csv`), not in git. Run time on an A100-SXM4-40GB:
9.6 min (notebook 12), 5.7 min (notebook 13, part 2).

## Frozen features without training (notebook 14)

Added on 2026-09-28 (`notebooks/14_embeddings.ipynb`), after every TEST run.
Descriptive only: nothing is trained, no result file changes, and it changes no
choice. It looks at the frozen encoder features themselves, with no probe and
no decoder, on the two TEST images of the README GIFs:

- hard: Columbia, Sentinel-1, 2020-03-08 (`COL_2020-03-08_S1_20_3_094`),
  702 × 1,492 px, a map of 44 × 94 patches, 4,136 valid, 81 of them holding a
  front pixel;
- easy: Mapple, Sentinel-1, 2017-09-08 (`Mapple_2017-09-08_S1_20_2_009`),
  405 × 382 px, 26 × 24 patches, 624 valid, 14 holding a front pixel.

Three encoders at the study's layers: C-RADIOv4-H layer 32, satellite DINOv3
layer 21, photo DINOv3 layer 18. Same preparation as the study (`caffe`
normalisation, 512 px tiles, no overlap). Each image's tiles are stitched into
one map of 16 px patches; a patch's zone is the majority of its pixels (NA,
stone, glacier, ocean).

**Method.**

- **Query patch, by a fixed rule, not by eye:** the purest (≥ 90 %) ocean patch
  closest to the true front that does not itself contain the front; ties
  toward the middle of the front. A glacier query, picked by the same rule, is
  the contrast. No fallback was needed: every query patch reached 90 %.
- **Similarity search:** the cosine similarity of the query to every patch of
  the image. Reported: the class shares of the 100 most similar patches (top
  100) and of the top k20 = max(10, round(0.2 × n)), n = the image's valid
  patches of the query's class. The top-100 shares compare the encoders on one
  image. The k20 shares take the same 20 % of the query's class on both images,
  so they compare the images.
- **Margin and AUC:** per patch, cos(patch, ocean centre) − cos(patch, glacier
  centre). The centres are the means of the image's own ocean and glacier
  patches, each vector scaled to length 1 first. The AUC is the chance that a
  random ocean patch has a larger margin than a random glacier patch (ties
  count half): 1.0 = fully apart, 0.5 = not at all. Ranks only; nothing is
  trained and no threshold is chosen.
- **AnyUp 1/4, whole image (as head 6):** per 512 px tile the features are
  upsampled to 128 × 128, one vector per 4 px block, in bf16 as in the reported
  runs. The query is the same patch: the mean of its 16 blocks, which are left
  out of the list. The top 1,600 blocks cover the area of 100 patches, and
  16 × k20 blocks the area of k20 patches.
- **Front window (AnyUp at full resolution, as heads 2 and 4):** a 256 × 256 px
  window around the ocean query. There the raw patches, AnyUp 1/4 and AnyUp at
  full resolution are compared pixel by pixel, by the same AUC over the
  window's pixels (centres from the window).

**Query patches.** Row and column in 16 px patches; all four are 1 patch from
the true front.

| image | query | patch (row, col) | its pixels | valid patches of its class (n) | k20 |
|---|---|---|---|---|---|
| COL | ocean | (31, 46) | 99 % ocean, 1 % stone | 499 | 100 |
| COL | glacier | (32, 52) | 100 % glacier | 1,828 | 366 |
| Mapple | ocean | (8, 15) | 100 % ocean | 113 | 23 |
| Mapple | glacier | (10, 15) | 100 % glacier | 272 | 54 |

Valid patches per class: COL 273 NA, 1,536 stone, 1,828 glacier, 499 ocean;
Mapple 36 NA, 203 stone, 272 glacier, 113 ocean.

**Similarity search: class shares of the matches**, as glacier / ocean / stone
(the NA share is 0.01 or less everywhere). Raw: 16 px patches. AnyUp 1/4: 4 px
blocks, the top 1,600 and the top 16 × k20.

| image, query | encoder | raw, top 100 | raw, top k20 | AnyUp 1/4, top 1,600 | AnyUp 1/4, top 16 × k20 |
|---|---|---|---|---|---|
| COL, ocean (k20 = 100) | C-RADIOv4-H L32 | 0.04 / 0.86 / 0.10 | 0.04 / 0.86 / 0.10 | 0.01 / 0.94 / 0.04 | 0.01 / 0.94 / 0.04 |
| COL, ocean | DINOv3 SAT L21 | 0.09 / 0.77 / 0.14 | 0.09 / 0.77 / 0.14 | 0.01 / 0.94 / 0.04 | 0.01 / 0.94 / 0.04 |
| COL, ocean | DINOv3 photo L18 | 0.09 / 0.66 / 0.25 | 0.09 / 0.66 / 0.25 | 0.04 / 0.78 / 0.17 | 0.04 / 0.78 / 0.17 |
| COL, glacier (k20 = 366) | C-RADIOv4-H L32 | 0.47 / 0.33 / 0.20 | 0.50 / 0.24 / 0.26 | 0.41 / 0.45 / 0.12 | 0.43 / 0.26 / 0.30 |
| COL, glacier | DINOv3 SAT L21 | 0.74 / 0.19 / 0.07 | 0.65 / 0.21 / 0.14 | 0.84 / 0.11 / 0.05 | 0.64 / 0.21 / 0.15 |
| COL, glacier | DINOv3 photo L18 | 0.74 / 0.08 / 0.17 | 0.66 / 0.09 / 0.24 | 0.63 / 0.14 / 0.23 | 0.55 / 0.15 / 0.28 |
| Mapple, ocean (k20 = 23) | C-RADIOv4-H L32 | 0.23 / 0.76 / 0.01 | 0.04 / 0.96 / 0.00 | 0.27 / 0.67 / 0.05 | 0.21 / 0.79 / 0.00 |
| Mapple, ocean | DINOv3 SAT L21 | 0.29 / 0.60 / 0.11 | 0.04 / 0.96 / 0.00 | 0.34 / 0.62 / 0.04 | 0.20 / 0.80 / 0.00 |
| Mapple, ocean | DINOv3 photo L18 | 0.08 / 0.92 / 0.00 | 0.00 / 1.00 / 0.00 | 0.14 / 0.81 / 0.04 | 0.15 / 0.85 / 0.00 |
| Mapple, glacier (k20 = 54) | C-RADIOv4-H L32 | 0.70 / 0.29 / 0.01 | 0.83 / 0.17 / 0.00 | 0.58 / 0.38 / 0.04 | 0.63 / 0.36 / 0.01 |
| Mapple, glacier | DINOv3 SAT L21 | 0.70 / 0.29 / 0.01 | 0.76 / 0.24 / 0.00 | 0.58 / 0.40 / 0.02 | 0.57 / 0.42 / 0.01 |
| Mapple, glacier | DINOv3 photo L18 | 0.56 / 0.44 / 0.00 | 0.76 / 0.24 / 0.00 | 0.54 / 0.43 / 0.03 | 0.65 / 0.34 / 0.01 |

**Ocean vs glacier AUC over the whole image.** COL: 499 ocean and 1,828 glacier
patches, 7,785 and 28,713 blocks. Mapple: 113 and 272 patches, 1,784 and 4,159
blocks.

| encoder | COL, raw | COL, AnyUp 1/4 | Mapple, raw | Mapple, AnyUp 1/4 |
|---|---|---|---|---|
| C-RADIOv4-H L32 | 0.935 | 0.947 | 0.999 | 0.999 |
| DINOv3 SAT L21 | 0.872 | 0.922 | 0.990 | 0.999 |
| DINOv3 photo L18 | 0.737 | 0.738 | 0.975 | 0.977 |

**Front window: ocean vs glacier AUC over the window's pixels.** COL: y 256–511,
x 616–871, 17,758 glacier and 33,565 ocean pixels. Mapple: y 8–263, x 120–375,
20,557 glacier and 26,305 ocean pixels. These compare the three versions with
each other, not with the whole-image AUCs (a different area).

| encoder | COL: raw / AnyUp 1/4 / AnyUp full | Mapple: raw / AnyUp 1/4 / AnyUp full |
|---|---|---|
| C-RADIOv4-H L32 | 0.987 / 0.989 / 0.986 | 0.995 / 0.995 / 0.995 |
| DINOv3 SAT L21 | 0.970 / 0.979 / 0.977 | 0.991 / 0.997 / 0.997 |
| DINOv3 photo L18 | 0.991 / 0.992 / 0.991 | 0.967 / 0.967 / 0.967 |

**Reading** (descriptive, one image per glacier).

1. **Every encoder separates ocean from glacier much worse on the hard Columbia
   Sentinel-1 image than on Mapple**: whole-image AUC 0.737–0.935 against
   0.975–0.999 (raw). C-RADIO is best on Columbia, the photo DINOv3 worst. Like
   for like (k20), the ocean query's matches are also less pure on Columbia
   (ocean 0.66–0.86) than on Mapple (0.96–1.00).
2. **The ocean patch next to the front looks like ocean to every encoder.** On
   Columbia its top 100 hold 0.04–0.09 glacier; the rest is mostly ocean
   (0.66–0.86), then stone (0.10–0.25). The idea that ice mélange in front of the glacier looks like
   glacier to the encoders came from looking at Columbia's Sentinel-1 images in
   the viewer (notebook 10). It was a hypothesis, never measured, and at this
   spot the features do not support it. C-RADIO instead sees the glacier patch
   next to the front as partly ocean (0.33 of its top 100, 0.24 of its k20).
3. **AnyUp 1/4 helps a little.** On Columbia it raises the whole-image AUC by
   0.05 (satellite DINOv3), 0.01 (C-RADIO) and about 0 (photo DINOv3), and makes
   the ocean query's matches purer (ocean 0.94 / 0.94 / 0.78 against
   0.86 / 0.77 / 0.66 raw). In the 256 px front window all versions score
   0.967–0.997, and AnyUp adds at most 0.009.
4. This does not change what the saved results show: over the 22 saved TEST
   runs, the 18 Columbia Sentinel-1 images hold 15–38 % of the summed error
   (see [Where the test error comes from](#where-the-test-error-comes-from)).
   Why they are hard stays open.

**Caveats.**

- Descriptive: two images, the README GIF images, looked at after every TEST
  run. Nothing here is a test score, and it changes no choice.
- One query patch per class and image: a picture, not a measurement. A share is
  best read against how common each class is in the image (counts above).
- The centres come from each image's own true labels, so the AUCs are
  optimistic: they say how well distance can tell the two classes apart on
  that image, not how well a model would do.
- AnyUp was trained on photos, not on SAR.

Run time on an A100-SXM4-40GB: 7.2 min in total, of it 2.7 min for the
features of all three encoders, AnyUp included. The figures,
`embeddings_numbers.csv` (30 rows) and one `embeddings_gif_<stem>.npz` per
image are in `MyDrive/sar-transfer/embeddings/`, not in git. The saved outputs
of the notebook hold all 20 figures.

## Caveats in full

**Protocol.** Each set of models was scored on TEST once and not changed
afterwards. But TEST results did shape later work. Arm 2 and the switch from the
photo to the satellite DINOv3 were decided after arm 1's TEST run. The AnyUp
precision check, the code fixes and approach A (`09_conditioning`,
exploratory, SI only) came after arm 2's TEST run. A TEST run of
approach A would be a second look at the test glaciers.

**Training tiles of rows 3–6 were not a uniform sample.** The reported runs kept
the first 4,000 candidate tiles, then drew a slot in [0, 4 × 4,000) for each
later tile. So every later tile replaced a kept one with probability 1/4, however
many tiles came before. With 12,595 candidate training tiles, early tiles in
file order survived with probability about 0.58 and the last ones about 0.25,
against 0.32 for a uniform sample (roughly 2.3×). The heads saw more of the glaciers and sensors that come
first in the scan order. The draw depends only on image order and valid-patch
counts, not on the encoder, so rows 3, 4 and 6 of every encoder trained on the
same tiles. Head-vs-head and encoder-vs-encoder comparisons stay fair. Row 5
(2,000 tiles) used a different, equally tilted set. The current code draws a
uniform sample by default; `legacy_tile_sampler=True` recovers the old set.

**The seed did not set the initial weights of the reported heads.** Each head
was built before the seed was set, so its initial weights came from the global
random state. The seed set the tile order and the flips. The three seeds did
differ, so the seed spreads are real. A reported head can only be recreated from
its saved weights. The current code sets the seed before building the head.

**Retraining will not reproduce the reported heads.** With the current code,
retraining uses the uniform sampler and seeded initial weights. Even with
`legacy_tile_sampler=True` the initial weights now differ. In the default mode
the linear-probe patch subsample can also shift, rarely: numpy's bounded integer
draw sometimes uses an extra random word. For the exact reported patch
subsample use `legacy_tile_sampler=True`. The saved heads (`heads__*.pt`) are
the only exact record. In an offline CPU test, heads trained by the old code
gave byte-identical test CSVs with the current code. The reported heads
themselves were not re-scored, and CUDA was not tested.

**Parameter counts in the Drive CSVs.** Summaries written by the reported runs
keep the printed counts for rows 4 and 6 (166,628 / 183,012 and 281,348 /
305,924). New summaries record the lower counts. A table that joins old and new
summaries can show both counts for the same row.

**Rows 2, 4, 5 and 6 are not controlled ablations of row 3.** Row 4 has about
1/5 of row 3's parameters and one image scale instead of five. Row 5 used 2,000
tiles instead of 4,000 (16 passes over them instead of 8) and 4× the input
width.

**AnyUp was trained on photos.** Its guide image expects ImageNet
normalisation. It is untested on SAR beyond this study.

**The layer choice uses a proxy.** Layers for the front heads were chosen by
3-class zone patch mIoU, not by MDE.

**SI did not predict TEST.** The decoders' SI lead over the U-Net (about 490 m
C-RADIO, about 400 m satellite DINOv3) became a tie (C-RADIO) and a deficit
(satellite DINOv3) on TEST. The satellite probe looked better than the photo
probe on SI fronts and was worse on TEST.

**Our U-Net is a weak reference.** Paper-style IoU 0.677 vs 0.79 published
(ocean 0.54 vs 0.82); MDE on all TEST images 979 vs 753 m, COL 1,080 vs 840 m,
Mapple 474 vs 150 m. It is a plain
U-Net without ASPP, one run, fixed 48k iterations, no early stopping. The
3-class and 4-class U-Nets are separate models.

**The SAR encoders got mismatched inputs.** TerraMind expects VV + VH in dB and
got one 8-bit channel copied twice. SARATR-X was pretrained on SARDet-180K:
186,600 small target chips (vehicles, ships, aircraft and other targets such as
oil tanks and bridges) from 14 public datasets at about 0.1–25 m, not on
glacier scenes.

**Small groups and confounds.** L-band is 7 SI images and 8 TEST images, all
PALSAR; the TEST ones are all on Mapple. RADARSAT-1 at 20 m is 6 SI images. ERS
on Mapple is 2 images. JAC (36 % of training images) is all TerraSAR-X; COL is
only Sentinel-1 and TanDEM-X.

**MDE depends on the no-front count.** A model that misses many fronts is scored
on an easier subset. The CaFFe paper describes MDE as averaged over images; the
official code pools distances, and we follow the code.

**The same nominal probe has slightly different values in different runs.** The
C-RADIO last-layer probe scores 1,816 m (5 no front) on SI in arm 1 and 1,791 m
(6) as row 1 in arm 2; on TEST 2,089 m (37) vs 2,115 m (38).

**Saved notebook outputs.** The reference lines printed in the saved outputs of
`08_heads` (cells `STEP 2 on SI` and `ONE-TIME test run`) say "DINOv3" for
the photo probe (SI 2,018 m, TEST 2,199 m). The satellite probe's references
are 1,852 m (SI) and 2,805 m (TEST).
The cell sources now name both.

**No confidence intervals**, except the row-3 seed SDs. One dataset.

**Supply chain.** C-RADIO runs made before the revision pin was added ran remote
code fetched at run time. A behavioural backdoor inside third-party weights
cannot be ruled out by inspection; here its worst case would be lower accuracy.

## Negative result: Cosmos-Reason2-8B

For a while the project tested whether a vision-language model could supply
labels. `nvidia/Cosmos-Reason2-8B` was asked whether a calving front advanced
or retreated between two SAR images of the same glacier.

- The hosted API returned 404, so the model ran from Hugging Face in bf16 on an
  A100-40GB, with greedy decoding.
- 56 image pairs from the training glaciers: 7 advances and 7 retreats per
  glacier, area change of at least 5 %, at least 90 days apart, 49 of 56 from
  the same sensor. Ground truth is the label-derived glacier area within a
  shared image frame.

| setup | accuracy | advance / retreat / unparsed | advance recall |
|---|---|---|---|
| direct | 0.500 | 0 / 56 / 0 | 0.00 |
| reason | 0.429 | 8 / 43 / 5 | 0.11 |
| rule-assisted (given the label-derived areas) | 0.911 | 23 / 33 / 0 | 0.82 |
| colour composite, direct | 0.500 | all 56 "retreat" | 0.00 |
| colour composite, reason | 0.232 | 34 unparsed | 0.14 |

- The colour composite puts both dates in one RGB image (red = earlier, green
  and blue = later).
- The two direct setups answered "retreat" for all 56 pairs. The two reasoning
  setups gave a few "advance" answers but were below chance or unparsable.
  Composite reasoning: 34 of 56 answers hit the 1,024-token limit; of the 22
  parsed, 13 were correct (0.59, p ≈ 0.3).
- The rule-assisted setup would score 1.000 by copying the areas it was given.
  All 5 of its errors turned a true advance into "retreat".
- Verdict: no evidence that Cosmos-Reason2-8B can judge calving-front change in
  SAR. The arm was closed at 8B. The planned 32B test was not run.

## Provenance: what the code fetches

| source | what | check |
|---|---|---|
| Hugging Face | DINOv3 photo and SAT (gated, needs `HF_TOKEN`) | safetensors; no revision pin |
| Hugging Face | C-RADIOv4-H with remote code | revision pin; SHA-256 of 26 Python files compared with a reviewed copy (the other 2 of the 28 fetched files are JSON configs) |
| Hugging Face | SARATR-X v1 checkpoint | `weights_only=True` with `argparse.Namespace` allow-listed; no revision pin |
| Hugging Face, via terratorch | TerraMind 1.0 base | terratorch's loader uses `weights_only=True` (read, the library not audited line by line); no revision pin |
| GitHub release (`wimmerth/anyup`) | AnyUp weights, if missing on Drive | size (3,541,624 bytes) and the full SHA-256 |
| PANGAEA | CaFFe `data_raw.zip` | no checksum |
| PyPI | timm, einops, open_clip_torch (C-RADIO), terratorch (TerraMind); numpy, pandas, pillow, tqdm if missing | no pinned versions |
| Google Drive | `.env`, data zip, results, model cache | your own storage |

- The notebooks upload nothing to third-party services. The code of the closed
  Cosmos arm is not part of this repository.
- C-RADIO remote-code review: all 28 files were read. The only network-related
  content is a table of checkpoint URL strings and one `torch.hub.load` that our
  configuration never reaches.
- AnyUp: inference code vendored from `wimmerth/anyup` at commit
  `351807a9c4287368732cc247f26c7c81c9139af4` (12 Python files), with its LICENSE and
  NOTICE.md. The docstring in its `__init__.py` is ours. No `torch.hub`.
- Pickle scan: the AnyUp, SARATR-X, TerraMind and C-RADIO `.pth.tar` checkpoints
  were scanned for GLOBAL opcodes without executing anything. They hold only
  tensor helpers and, for SARATR-X and C-RADIO, one `argparse.Namespace`.
- Our loaders refuse a checkpoint that carries a `__reduce__` payload (tested).
- The official CaFFe code was tested outside this repository and never copied
  in.
- The weight files and their SHA-256 manifest are kept locally, not in git.

## Faster feature copy-back

Added on 2026-09-27 and checked in `notebooks/11_predictions.ipynb` on an
A100-SXM4-80GB, before its part 2 ran. The encoder output is cast to float16 on
the GPU before it is copied to the CPU: half the bytes, and no numpy
conversion. Measured on 298 real training tiles per encoder, old and new path
in alternating order, compared bit for bit:

| encoder | forward, ms per tile | copy-back, ms per tile: old → new | encoder pass (forward + copy-back) | features, old vs new |
|---|---|---|---|---|
| C-RADIOv4-H, layer 32 | 10.4 | 9.3 → 1.2 | 41 % faster | identical on every batch |
| DINOv3 SAT, layer 21 | 7.8 | 7.8 → 1.2 | 42 % faster | identical on every batch |

The features are bit-identical, so no reported number changes. The reported
runs used the old path, and the times under [Run times](#run-times) were
measured with it. Notebooks 11–13 ran with the new path, and every head in
them reproduced its saved results exactly.

## Run times

On a Colab A100.

| step | time |
|---|---|
| feature extraction, one encoder × normalisation | 1.7–6.1 min; four-layer row-5 extraction 9–14 min |
| first encoder loop (`03_encoders`) | 43 min |
| U-Net, 48k iterations | 19.9 min (A100-40GB) |
| `06_fronts`, four encoders | 22 min |
| `07_test`: encoders / U-Nets / satellite DINOv3 probe | 25 min / about 5 min / 6.5 min |
| layer sweep, per encoder | 13–14 min |
| arm 2 heads on SI | rows 1–4: C-RADIO 57 min, photo DINOv3 51 min (rows 2–4); C-RADIO rows 5, 6 and row-3 seeds: 71 min; satellite DINOv3, all rows: 110 min |
| arm 2 heads on TEST, both encoders | no total printed; the printed step times sum to about 1.5 h |
| AnyUp precision check | 0.5 min per encoder |
| Cosmos, 134 calls | 37 min |

## Source files

Result CSVs are written to `MyDrive/sar-transfer/results/` and are not in git.

| table | CSVs |
|---|---|
| arm 1 zones on SI | `probe_zones__<run>.csv` (U-Net: also `unet_pixel__<run>.csv`) |
| arm 1 fronts on SI | `fronts_mde__<run>.csv`, `fronts_images__<run>.csv`, `zones4_pixel__<run>.csv` |
| arm 1 TEST | `TEST_probe_zones3__<run>.csv`, `TEST_fronts_mde__<run>.csv`, `TEST_zones4_paper__<run>.csv`, `TEST_zones4_pixel__<run>.csv`, `TEST_fronts_images__<run>.csv` |
| layer sweep | `layersweep__<run>.csv` |
| arm 2 SI / TEST | `heads_{val,test}_mde__<run>__row<k>[__s<seed>].csv` (plus `_images`, `_zones4`, `_paper`), `heads_{val,test}_summary__<run>.csv` |
| precision check | `check_anyup_precision__<run>.csv` |
| Cosmos | `cosmos_pairs_Cosmos-Reason2-8B_v1.csv`, `cosmos_pairs_Cosmos-Reason2-8B_v1_with_composite.csv` |

`<run>` is `<encoder>__<normalisation>__tile<size>`, with `__L<layer>` for arm 2
and `__na4` for the 4-class front runs.
