#!/usr/bin/env python
"""Producer: frozen foundation-model embeddings of the 263 GRAPE **baseline** colour fundus photos.

    .venv-torch/bin/python scripts/run_grape_cfp_embeddings.py --backbone retfound_cfp
    .venv-torch/bin/python scripts/run_grape_cfp_embeddings.py --backbone dinov2_reg

Each baseline CFP (``{subject}_{lat}_1.jpg``) → one frozen embedding, cached to
``data/derived/grape_<backbone>_emb.npz`` = {emb (N,D), eye_id (N,), camera (N,), prog_plr2/plr3/md (N,)}.
The consumer (Phase C) PCA-compresses ``emb`` into the ``fundus_emb`` modality of the progression engine.

Frozen-feature conventions are each model's own headline linear-eval protocol (so the embedding is faithful,
not an ad-hoc read):
  * ``retfound_cfp`` — RETFound (Zhou 2023, Nature) ViT-L/16 @224, GAP head (``global_pool='avg'`` + fc_norm),
    1024-d. Weights = the non-gated ``bitfount/RETFound_MAE`` mirror (the original CFP model), pre-fetched to
    ``data/models/retfound_cfp.bin`` (or ``$RETFOUND_CFP_PTH``). ImageNet norm.
  * ``dinov2_reg`` — DINOv2-with-registers ViT-L/14 @224, feature = [CLS] ⊕ avg-pooled patch tokens (registers
    excluded), 2048-d (Oquab 2023 §7 linear-eval). Model's own norm.
  * ``convnext_*`` / ``resnet50`` — ImageNet-supervised CNN, global-avg-pooled features (the generic-supervised
    control for Phase E). Model's own norm.

Runs on CPU (263 images ≈ a few minutes for ViT-L). ``HF_HUB_OFFLINE=1`` is honoured (all timm backbones here
are pre-cached). No fine-tuning — frozen encoder only (14–40 events → linear-probe regime).
"""

from __future__ import annotations

import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# NOTE: the .venv-torch env has torch/timm + pandas but NOT the data-layer deps (openpyxl/pyyaml), so this
# producer reads a pre-generated MANIFEST CSV (absolute cfp paths + eye_id + camera + labels) instead of
# importing the openpyxl-backed loader. Generate it once with the main env:
#   .venv/bin/python -c "import sys;sys.path.insert(0,'src');from glaucoma_triage import config;\
#     from glaucoma_triage.datasets import grape_cfp;\
#     grape_cfp.load_grape_cfp(config.raw_dir('grape')).to_csv('data/derived/grape_cfp_manifest.csv',index=False)"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# timm-cached backbones (non-RETFound). retfound_cfp is handled separately (local .bin).
TIMM_BACKBONES = {
    "dinov2_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2": "vit_large_patch14_dinov2.lvd142m",
    "resnet50": "resnet50.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "convnext_base": "convnext_base.fb_in1k",
    "convnext_large": "convnext_large.fb_in1k",
}
_IS_VIT = {"dinov2_reg", "dinov2"}


def _load_retfound_cfp():
    """RETFound-CFP ViT-L/16 GAP head (mirrors run_retfound_contender._load_retfound, CFP weights)."""
    import timm

    path = os.environ.get("RETFOUND_CFP_PTH", "data/models/retfound_cfp.bin")
    if not Path(path).exists():
        raise SystemExit(f"RETFound-CFP weights not found at {path}. Fetch the non-gated bitfount/RETFound_MAE "
                         "mirror: curl -sL https://huggingface.co/bitfount/RETFound_MAE/resolve/main/"
                         "pytorch_model.bin -o data/models/retfound_cfp.bin")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder") and k not in ("mask_token",)}
    if "norm.weight" in sd:   # map MAE final-norm -> timm GAP fc_norm (applied after pooling)
        sd["fc_norm.weight"] = sd.pop("norm.weight")
        sd["fc_norm.bias"] = sd.pop("norm.bias")
    model = timm.create_model("vit_large_patch16_224", num_classes=0, global_pool="avg", img_size=224)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"RETFound-CFP loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    return model.eval(), (_IMAGENET_MEAN, _IMAGENET_STD), 224


def _load_timm(backbone: str):
    import timm

    is_vit = backbone in _IS_VIT
    res = 224
    model = timm.create_model(TIMM_BACKBONES[backbone], pretrained=True, num_classes=0,
                              **({"img_size": res} if is_vit else {})).eval()
    cfg = timm.data.resolve_model_data_config(model)
    mean, std = np.array(cfg["mean"], dtype=np.float32), np.array(cfg["std"], dtype=np.float32)
    print(f"{backbone} ({TIMM_BACKBONES[backbone]}) loaded @ {res}px; norm mean={mean.round(3).tolist()}")
    return model, (mean, std), res


def _fov_crop(img: "Image.Image", thresh: int = 12) -> "Image.Image":
    """Crop a colour fundus photo to its retinal field-of-view (the bright circle), dropping the black border.

    Essential preprocessing: without it, resizing the full frame makes the encoder key on the camera's border
    /FOV signature (GRAPE's TOPCON 1556² vs Canon 2136² frames) rather than the retina — an acquisition
    confound (the embedding otherwise predicts the camera at AUC≈0.99). Threshold the luminance, take the
    bounding box of the foreground, expand to a centred square, crop. Falls back to the full frame if the mask
    is degenerate.
    """
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    mask = g > thresh
    cols = np.where(mask.mean(axis=0) > 0.02)[0]
    rows = np.where(mask.mean(axis=1) > 0.02)[0]
    if len(cols) < 8 or len(rows) < 8:
        return img
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0
    W, H = img.size
    left, upper = int(round(cx - half)), int(round(cy - half))
    right, lower = int(round(cx + half)), int(round(cy + half))
    # clamp to image bounds (a small off-centre disc near an edge just clips slightly)
    left, upper = max(0, left), max(0, upper)
    right, lower = min(W, right), min(H, lower)
    return img.crop((left, upper, right, lower))


def _disc_crop(img: "Image.Image", cx: float, cy: float, r: float, margin: float = 3.5) -> "Image.Image":
    """Crop a square centred on the optic disc (from the contour geometry), side ≈ ``2*margin*r``.

    Glaucomatous progression is a peripapillary/disc phenomenon, so a disc-centred crop concentrates the signal
    the full-fundus embedding dilutes — and because ``r`` scales with the camera resolution (Canon 2136² vs
    TOPCON 1556²), a disc-radius-relative crop also SCALE-NORMALISES across cameras (shrinking the batch effect).
    The square is clamped to fit fully in-frame (no black padding).
    """
    W, H = img.size
    half = min(r * margin, cx, cy, W - cx, H - cy)
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def _make_embed(backbone: str, model, norm, res: int, crop: str = "fov"):
    mean, std = norm
    is_vit_dino = backbone in _IS_VIT
    npref = getattr(model, "num_prefix_tokens", 0)

    @torch.no_grad()
    def embed(path: str, disc: tuple | None = None) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if crop == "disc" and disc is not None and not any(np.isnan(v) for v in disc):
            img = _disc_crop(img, *disc)
        elif crop == "fov":
            img = _fov_crop(img)
        img = img.resize((res, res))
        x = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
        xt = torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32)
        if is_vit_dino:                                    # DINOv2: [CLS] ⊕ avg-pooled patch tokens
            tok = model.forward_features(xt)
            return torch.cat([tok[:, 0], tok[:, npref:].mean(dim=1)], dim=-1).numpy()[0]
        return model(xt).numpy()[0]                        # RETFound GAP head / CNN global-avg-pool

    return embed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="retfound_cfp",
                    choices=["retfound_cfp", *TIMM_BACKBONES])
    ap.add_argument("--manifest", default="data/derived/grape_cfp_manifest.csv")
    ap.add_argument("--crop", default="fov", choices=["fov", "disc", "none"],
                    help="fov = crop to the retinal circle (default); disc = crop a scale-normalised square "
                         "around the optic disc (uses contour geometry; concentrates glaucoma signal + shrinks "
                         "the camera batch effect); none = full frame (diagnostic).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pandas as pd

    if not Path(args.manifest).exists():
        raise SystemExit(f"manifest {args.manifest} not found — generate it with the main env (see the header "
                         "comment in this file).")
    eyes = pd.read_csv(args.manifest)
    have = eyes[eyes["exists"]].reset_index(drop=True)
    print(f"[grape-cfp] {len(have)}/{len(eyes)} baseline CFPs on disk "
          f"(cameras: {have['camera'].value_counts().to_dict()})")
    if have.empty:
        raise SystemExit("no baseline CFPs found — run scripts/download_grape.py --cfps")

    if args.backbone == "retfound_cfp":
        model, norm, res = _load_retfound_cfp()
    else:
        model, norm, res = _load_timm(args.backbone)
    embed = _make_embed(args.backbone, model, norm, res, crop=args.crop)
    print(f"[grape-cfp] crop mode: {args.crop}")
    has_disc = {"disc_cx", "disc_cy", "disc_r"}.issubset(have.columns)

    embs, t0, n = [], time.time(), len(have)
    for i, row in enumerate(have.itertuples(index=False)):
        disc = (row.disc_cx, row.disc_cy, row.disc_r) if has_disc else None
        embs.append(embed(row.cfp_path, disc))
        if (i + 1) % 25 == 0 or (i + 1) == n:
            el = time.time() - t0
            print(f"  {i + 1:>3}/{n}  {el:5.0f}s  {(i + 1) / el:4.1f}/s", flush=True)
    emb = np.asarray(embs, dtype=np.float32)

    suffix = "" if args.crop == "fov" else f"_{args.crop}"
    out = Path(args.out) if args.out else Path(f"data/derived/grape_{args.backbone}{suffix}_emb.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        emb=emb,
        eye_id=have["eye_id"].to_numpy().astype(str),
        camera=have["camera"].to_numpy().astype(str),
        prog_plr2=have["prog_plr2"].to_numpy(),
        prog_plr3=have["prog_plr3"].to_numpy(),
        prog_md=have["prog_md"].to_numpy(),
        backbone=np.array(args.backbone),
    )
    print(f"wrote {out}: {emb.shape[0]} eyes, D={emb.shape[1]}, backbone={args.backbone}")


if __name__ == "__main__":
    main()
