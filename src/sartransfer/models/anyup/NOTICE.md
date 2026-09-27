# AnyUp -- vendored copy

Source: https://github.com/wimmerth/anyup, commit `351807a9c4287368732cc247f26c7c81c9139af4`, fetched 2026-09-25.
Licence: Creative Commons Attribution 4.0 International (see LICENSE in this folder).
Paper: T. Wimmer et al., "AnyUp: Universal Feature Upsampling", arXiv:2510.12764.

Only the inference code is kept (model, layers and two utilities). Training code,
the NATTEN attention variant, backbones, dataloaders and visualisation are not
included. The upstream code files are unmodified; the one-line docstring in this
package's `__init__.py` was written for this project.

The weights are not included. `sartransfer.models.anyup_loader.fetch_anyup_weights`
downloads `anyup_multi_backbone.pth` (3,541,624 bytes, SHA-256
b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59) from the
authors' release
https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth
into Google Drive when it is missing there, and every load checks size and hash.
No torch.hub is used.
