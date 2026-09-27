"""Frozen encoders behind one interface: tiles in, (N, gh, gw, dim) patch grid out.

Every backend receives the SAME input: our normalised tiles (`per_image` or
`caffe`, see data.tiling), grey replicated to 3 channels. Each model's own
preprocessing (ImageNet / CLIP / Sentinel-1 statistics) is bypassed on purpose,
so encoders are compared on identical inputs.

Backends, each checked against the model's own public code (2026-09-23):

- `dinov3` (Hugging Face transformers). ViT-L/16 at 512 px returns 1029 tokens:
  32x32 patches + 5 prefix (1 CLS + 4 registers). The prefix count is inferred
  from token count vs grid size, never hard-coded.
- `radio` (C-RADIOv4, trust_remote_code). `model(x)` returns (summary, spatial);
  spatial holds patch tokens only, NLC. It normally expects [0, 1] input and
  applies CLIP mean/std itself; `make_preprocessor_external()` switches that off
  so it takes already-standardised input. `preferred_resolution` is 512.
- `terramind` (IBM/ESA TerraMind 1.0 via terratorch, Apache-2.0). Sentinel-1 GRD
  modality expects 2 channels (VV, VH) in dB, standardised with its pretraining
  statistics; terratorch backbones take inputs already standardised. CaFFe has
  one channel of unknown polarisation, so it is copied into both VV and VH -- a
  known mismatch, documented. No CLS or register tokens; 2D sin-cos positions.
- `hivit` (SARATR-X v1): our own HiViT encoder in `models/hivit.py`, layout
  read from the checkpoint keys and loaded strictly. v1 was pretrained on [0, 1]
  input with no normalisation, so its fair setting is NORM = "unit".
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import numpy as np
import torch


@dataclass
class EncoderSpec:
    model_id: str
    name: str
    kind: str                       # dinov3 | radio | terramind | hivit
    trust_remote_code: bool = False
    batch: int = 32                 # tiles per forward pass at 512 px on an A100-40GB
    hf_file: str | None = None      # hivit: checkpoint file inside the HF repo
    revision: str | None = None     # remote-code models: pinned commit, hash-checked before loading


def verify_remote_code(spec: EncoderSpec, token: str | None = None) -> Path:
    """Refuse to run remote code that is not the reviewed copy.

    Downloads only the *.py files of the pinned revision into the HF cache (the
    same snapshot transformers loads the model code from) and compares the
    SHA-256 of every .py below the snapshot folder, subfolders included, keyed
    by relative path, with `remote_code_hashes.REMOTE_CODE_HASHES`. Any extra,
    missing or changed file raises before anything is executed.
    """
    import hashlib

    from huggingface_hub import snapshot_download

    from .remote_code_hashes import REMOTE_CODE_HASHES

    if spec.revision is None:
        raise ValueError(f"{spec.model_id}: remote code needs a pinned revision")
    expected = REMOTE_CODE_HASHES[spec.model_id]
    d = Path(snapshot_download(spec.model_id, revision=spec.revision, allow_patterns=["*.py"], token=token))
    got = {q.relative_to(d).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
           for q in d.rglob("*.py")}
    bad = sorted(k for k in set(expected) | set(got) if expected.get(k) != got.get(k))
    if bad:
        raise RuntimeError(f"{spec.model_id}@{spec.revision[:12]}: remote code differs from the reviewed copy: {bad}")
    print(f"remote code verified: {len(got)} files == reviewed copy ({spec.model_id}@{spec.revision[:12]})")
    return d


ENCODERS = {
    "dinov3-l-photo": EncoderSpec("facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov3-l-photo", "dinov3"),
    "dinov3-l-sat": EncoderSpec("facebook/dinov3-vitl16-pretrain-sat493m", "dinov3-l-sat", "dinov3"),
    "cradio-v4-h": EncoderSpec("nvidia/C-RADIOv4-H", "cradio-v4-h", "radio",
                               trust_remote_code=True, batch=16, revision="0057b339059c0b9e1b4ba996f975410ebbfdfcc8"),
    "terramind-v1-base-s1": EncoderSpec("ibm-esa-geospatial/TerraMind-1.0-base",
                                        "terramind-v1-base-s1", "terramind"),
    "saratrx-v1": EncoderSpec("waterdisappear/SARATR-X", "saratrx-v1", "hivit",
                              hf_file="weight/186K_all/checkpoint-800.pth"),
    # SARATR-X v2 deliberately not included (decision 2026-09-24): access-gated,
    # paper under review; v1 is the SAR-chip encoder in this study.
}


class FrozenEncoder:
    """A frozen backbone that maps normalised tiles to a (gh, gw, dim) patch grid."""

    def __init__(self, spec: EncoderSpec, token: str | None = None,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.patch = 16
        self.dim = 0
        self._n_prefix: int | None = None

        if spec.kind == "dinov3":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec.model_id, token=token, dtype=dtype)
            cfg = self.model.config
            self.patch = int(getattr(cfg, "patch_size", 16))
            self.dim = int(getattr(cfg, "hidden_size", 0))
        elif spec.kind == "radio":
            from transformers import AutoModel
            verify_remote_code(spec, token)          # pinned commit, files == reviewed copy
            self.model = AutoModel.from_pretrained(spec.model_id, token=token,
                                                   trust_remote_code=True, revision=spec.revision)
            self.model.make_preprocessor_external()   # we feed standardised input
            self.patch = int(self.model.patch_size)
        elif spec.kind == "terramind":
            from terratorch import BACKBONE_REGISTRY
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_base", pretrained=True, modalities=["S1GRD"])
        elif spec.kind == "hivit":
            from huggingface_hub import hf_hub_download
            from .hivit import load_hivit
            path = hf_hub_download(spec.model_id, spec.hf_file, token=token)
            self.model = load_hivit(path)
        else:
            raise ValueError(f"unknown encoder kind {spec.kind!r}")

        self.model = self.model.to(device).eval()
        print(f"{spec.name}: {sum(q.numel() for q in self.model.parameters()) / 1e6:.1f} M parameters loaded")
        # Frozen. Nothing trainable sits in front during extraction, so
        # inference_mode below is safe.
        self.model.requires_grad_(False)

    def __repr__(self) -> str:
        return (f"FrozenEncoder({self.spec.name}, kind={self.spec.kind}, patch={self.patch}, "
                f"dim={self.dim}, n_prefix={self._n_prefix})")

    def _grid(self, tokens: torch.Tensor, tile: int) -> torch.Tensor:
        """(B, T, D) -> (B, gh, gw, D), dropping any prefix tokens."""
        gh = gw = tile // self.patch
        n_patch = gh * gw
        n_prefix = tokens.shape[1] - n_patch
        if n_prefix < 0 or n_prefix > 16:
            raise RuntimeError(
                f"{tokens.shape[1]} tokens for a {tile}px tile at patch {self.patch}: "
                f"expected {n_patch} patch tokens plus a small prefix, got prefix {n_prefix}")
        if self._n_prefix is None:
            self._n_prefix = n_prefix
        elif self._n_prefix != n_prefix:
            raise RuntimeError(f"prefix count changed: {self._n_prefix} -> {n_prefix}")
        self.dim = int(tokens.shape[-1])
        return tokens[:, n_prefix:, :].reshape(tokens.shape[0], gh, gw, -1)

    def _tokens(self, b: torch.Tensor) -> torch.Tensor:
        """One batch (B, 3, H, W) -> token sequence (B, T, D)."""
        kind = self.spec.kind
        if kind == "dinov3":
            return self.model(pixel_values=b.to(self.dtype)).last_hidden_state
        with torch.autocast("cuda", dtype=self.dtype):
            if kind == "radio":
                out = self.model(b)
                spatial = out[1] if isinstance(out, (tuple, list)) else out.features
                return spatial                               # NLC, patch tokens only
            if kind == "terramind":
                layers = self.model({"S1GRD": b[:, :2]})     # grey copied into VV and VH
                return layers[-1]                            # last layer, normed
            if kind == "hivit":
                return self.model(b)                         # (B, N, 512), no prefix
        raise ValueError(kind)

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks (for the layer sweep)."""
        kind = self.spec.kind
        if kind == "dinov3":
            return int(self.model.config.num_hidden_layers)
        if kind == "radio":
            inner = getattr(self.model, "radio_model", self.model)
            return len(inner.model.blocks)
        raise NotImplementedError(f"layer sweep not supported for {kind}")

    def _layer_tokens(self, b: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        """Token sequences after block k (1-based), final norm applied, for each k.

        dinov3: HF `hidden_states[k]` is the output of block k (index 0 = the
                embeddings); the model's own backbone variant applies `norm` to
                intermediates, so the same is done here. k = n_layers equals the
                usual `last_hidden_state`.
        radio:  `forward_intermediates(indices=[k-1], norm=True)`; returns spatial
                tokens only (no prefix), NLC.
        """
        kind = self.spec.kind
        if kind == "dinov3":
            out = self.model(pixel_values=b.to(self.dtype), output_hidden_states=True)
            hs = out.hidden_states
            return {k: self.model.norm(hs[k]) for k in layers}
        if kind == "radio":
            inner = getattr(self.model, "radio_model", self.model)
            with torch.autocast("cuda", dtype=self.dtype):
                feats = inner.forward_intermediates(b, indices=[k - 1 for k in layers], norm=True,
                                                    output_fmt="NLC", intermediates_only=True)
            return {k: f for k, f in zip(layers, feats)}
        raise NotImplementedError(f"layer sweep not supported for {kind}")

    @torch.inference_mode()
    def encode_layers(self, tiles: np.ndarray, layers: list[int],
                      batch: int | None = None) -> dict[int, np.ndarray]:
        """Like `encode`, but returns {layer: (N, gh, gw, dim) fp16} for several layers."""
        if tiles.ndim != 4 or tiles.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tiles.shape}")
        batch = batch or self.spec.batch
        tile = tiles.shape[-1]
        out: dict[int, list] = {k: [] for k in layers}
        x = torch.from_numpy(tiles)
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            for k, tok in self._layer_tokens(b, layers).items():
                out[k].append(to_host_fp16(self._grid(tok, tile)))
        return {k: np.concatenate(v) for k, v in out.items()}

    @torch.inference_mode()
    def encode(self, tiles: np.ndarray, batch: int | None = None,
               layer: int | tuple[int, ...] | None = None) -> np.ndarray:
        """tiles (N, 3, tile, tile) float32 -> features (N, gh, gw, dim) float16.

        `layer` (1-based block index) taps an intermediate layer, final norm
        applied, as chosen by the layer sweep. None = the model's final output.
        A tuple of layers returns them concatenated on the channel axis
        (N, gh, gw, len(layers) * dim), in the given order.
        """
        if tiles.ndim != 4 or tiles.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tiles.shape}")
        if isinstance(layer, (tuple, list)):
            d = self.encode_layers(tiles, list(layer), batch=batch)
            return np.concatenate([d[k] for k in layer], axis=-1)
        if layer is not None:
            return self.encode_layers(tiles, [layer], batch=batch)[layer]
        batch = batch or self.spec.batch
        tile = tiles.shape[-1]
        out = []
        x = torch.from_numpy(tiles)
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            tok = self._tokens(b)
            out.append(to_host_fp16(self._grid(tok, tile)))
        return np.concatenate(out) if out else np.empty((0,), dtype=np.float16)


def to_host_fp16(t: torch.Tensor) -> np.ndarray:
    """Features -> float16 numpy, cast on the device first (speed-up 2026-09-27).

    Copies half the bytes and skips numpy's float16 conversion on the CPU. Same
    values as the old `t.float().cpu().numpy().astype(np.float16)`: bf16 -> fp32
    is exact, and both paths round fp32 -> fp16 to nearest-even.
    `checks.copy_back_check` compares both paths on real features.
    """
    return t.to(torch.float16).cpu().numpy()


def to_host_fp16_old(t: torch.Tensor) -> np.ndarray:
    """The path used for every run up to 2026-09-27, kept for the comparison."""
    return t.float().cpu().numpy().astype(np.float16)
