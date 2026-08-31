"""Full-fundus (fov) RETFound-CFP embeddings (ImageNet norm) for ALL GRAPE CFPs — baseline + follow-up — keyed by
(eye_id, visit, camera). Feeds the serial-fundus honest-vs-circular analysis (run_grape_serial_fundus.py).

Two steps (RETFound needs torch; the workbook needs openpyxl in the main env):
  1. .venv:        build the manifest (already done):
       data/derived/grape_followup_cfp_manifest.csv  columns eye_id,visit,cfp,cfp_path,camera,exists
  2. .venv-torch:  .venv-torch/bin/python scripts/run_grape_followup_cfp_embeddings.py
       -> data/derived/grape_retfound_followup_emb.npz   {emb (N,1024), eye_id, visit, cfp, camera}

RETFound-CFP = ViT-L/16 GAP head, local weights data/models/retfound_cfp.bin (bitfount/RETFound_MAE mirror),
ImageNet normalization (NOT timm's default 0.5/0.5/0.5 — that is the known norm bug). fov crop = de-border to the
retinal circle. ~631 CPU inferences, ~15-30 min.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_retfound():
    import timm
    path = os.environ.get("RETFOUND_CFP_PTH", "data/models/retfound_cfp.bin")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder") and k not in ("mask_token",)}
    if "norm.weight" in sd:                                  # MAE final-norm -> timm GAP fc_norm
        sd["fc_norm.weight"] = sd.pop("norm.weight")
        sd["fc_norm.bias"] = sd.pop("norm.bias")
    m = timm.create_model("vit_large_patch16_224", num_classes=0, global_pool="avg", img_size=224)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"RETFound-CFP loaded ({len(miss)} missing, {len(unexp)} unexpected keys)", flush=True)
    return m.eval()


def _fov_crop(img, thresh=12):
    g = np.asarray(img.convert("L"), np.uint8)
    mask = g > thresh
    cols = np.where(mask.mean(0) > 0.02)[0]
    rows = np.where(mask.mean(1) > 0.02)[0]
    if len(cols) < 8 or len(rows) < 8:
        return img
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])
    cx, cy = (x0 + x1) / 2., (y0 + y1) / 2.
    half = max(x1 - x0, y1 - y0) / 2.
    W, H = img.size
    return img.crop((max(0, int(cx - half)), max(0, int(cy - half)), min(W, int(cx + half)), min(H, int(cy + half))))


@torch.no_grad()
def _embed(model, path):
    img = _fov_crop(Image.open(path).convert("RGB")).resize((224, 224))
    x = (np.asarray(img, np.float32) / 255. - _MEAN) / _STD
    return model(torch.tensor(x.transpose(2, 0, 1)[None])).numpy()[0]


def main():
    man = pd.read_csv("data/derived/grape_followup_cfp_manifest.csv")
    man = man[man["exists"]].reset_index(drop=True)
    print(f"[followup-cfp] {len(man)} CFPs | eyes {man.eye_id.nunique()} | cameras {man.camera.value_counts().to_dict()}",
          flush=True)
    model = _load_retfound()
    embs, t0, n = [], time.time(), len(man)
    for i, row in enumerate(man.itertuples(index=False)):
        embs.append(_embed(model, row.cfp_path))
        if (i + 1) % 25 == 0 or (i + 1) == n:
            el = time.time() - t0
            print(f"  {i + 1:>3}/{n}  {el:5.0f}s  {(i + 1) / el:4.1f}/s", flush=True)
    emb = np.asarray(embs, dtype=np.float32)
    out = "data/derived/grape_retfound_followup_emb.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, eye_id=man.eye_id.to_numpy().astype(str), visit=man.visit.to_numpy().astype(int),
             cfp=man.cfp.to_numpy().astype(str), camera=man.camera.to_numpy().astype(str))
    print(f"[followup-cfp] wrote {out}  emb {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
