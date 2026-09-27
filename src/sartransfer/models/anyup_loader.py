"""Load the vendored AnyUp with the authors' weights from Drive, checked by hash."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch

from .anyup import AnyUp

WEIGHTS_NAME = "anyup_multi_backbone.pth"
WEIGHTS_SHA256 = "b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59"   # full hash of the release file, checked 2026-09-25
WEIGHTS_BYTES = 3_541_624
WEIGHTS_URL = ("https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/"
               "anyup_multi_backbone.pth")


def _ok(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size != WEIGHTS_BYTES:
        return False
    return hashlib.sha256(p.read_bytes()).hexdigest() == WEIGHTS_SHA256


def fetch_anyup_weights(path: str | Path) -> Path:
    """Make sure the released weights are at `path` (3.5 MB); download if not.

    Refuses to write under /content/drive unless Drive is really mounted, so a
    missing mount cannot silently turn the Drive path into a local folder.
    """
    import os
    import urllib.request

    p = Path(path)
    if _ok(p):
        return p
    if str(p).startswith("/content/drive") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Drive is not mounted: run the mount cell first, then this cell again")
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading AnyUp weights -> {p}")
    urllib.request.urlretrieve(WEIGHTS_URL, p)
    if not _ok(p):
        raise RuntimeError(f"downloaded file at {p} does not match the released weights")
    return p

_FP32_POSITIONS = False


@contextmanager
def fp32_positions(on: bool = True):
    """Inside this block `upsample_with_value` builds AnyUp's pixel coordinates and
    RoPE angles in fp32, even under bf16 autocast (code review 2026-09-25).

    Off by default: every reported run used the vendored behaviour, where the
    coordinates take the image encoder's dtype (bf16 under autocast).
    """
    global _FP32_POSITIONS
    old, _FP32_POSITIONS = _FP32_POSITIONS, on
    try:
        yield
    finally:
        _FP32_POSITIONS = old


# AnyUp's guide image must be ImageNet-normalised RGB (its README).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_anyup(weights_path: str | Path, device: str = "cuda") -> AnyUp:
    p = Path(weights_path)
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != WEIGHTS_BYTES or digest != WEIGHTS_SHA256:
        raise RuntimeError(f"{p}: size {len(raw)} / sha256 {digest} do not match the released "
                           f"weights ({WEIGHTS_BYTES} / {WEIGHTS_SHA256[:12]}...)")
    sd = torch.load(p, map_location="cpu", weights_only=True)   # never executes code
    model = AnyUp()
    model.load_state_dict(sd, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def guide_image(gray_uint8: torch.Tensor) -> torch.Tensor:
    """(N, 1 or 3, H, W) uint8/float 0-255 grey tiles -> ImageNet-normalised RGB float."""
    x = gray_uint8.float() / 255.0
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def upsample_with_value(model: AnyUp, guide: torch.Tensor, feats: torch.Tensor,
                        value: torch.Tensor, out_size: tuple[int, int] | None = None,
                        q_chunk_size: int | None = None) -> torch.Tensor:
    """AnyUp's attention computed from (guide image, feats), applied to `value`.

    AnyUp's output is attention @ V with V = the input features, unprojected
    (their CrossAttention, "weight the unprojected V"). So the attention can be
    computed once from the full-width features and applied to ANY tensor on the
    same patch grid:

      value = feats                -> identical to model(guide, feats)
      value = W feats + b          -> identical to W * model(guide, feats) + b,
                                      because each attention row sums to 1

    That makes "upsample then linear probe" exact at 4 channels instead of 1024,
    and lets a learned 1x1 reduction sit before the upsampler with gradients
    flowing only through the attention-weighted sum. This wrapper reproduces
    AnyUp.forward / AnyUp.upsample step by step with the vendored modules; it
    does not modify them.
    """
    import torch.nn.functional as F
    from .anyup.utils.img import create_coordinate

    out_size = out_size if out_size is not None else tuple(guide.shape[-2:])
    enc = model.image_encoder(guide)
    h = enc.shape[-2]
    if _FP32_POSITIONS:                       # see fp32_positions(); off in every reported run
        with torch.autocast(enc.device.type, enabled=False):
            coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=torch.float32)
            enc = enc.float().permute(0, 2, 3, 1).reshape(enc.shape[0], -1, enc.shape[1])
            enc = model.rope(enc, coords)
    else:                                     # vendored behaviour: coordinates in enc's dtype
        coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = model.rope(enc, coords)
    enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)

    b, c, hf, wf = feats.shape
    q = F.adaptive_avg_pool2d(model.query_encoder(enc), output_size=out_size)
    k = F.adaptive_avg_pool2d(model.key_encoder(enc), output_size=(hf, wf))
    k = torch.cat([k, model.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
    k = model.aggregation(k)
    return model.cross_decode(q, k, value, q_chunk_size=q_chunk_size)
