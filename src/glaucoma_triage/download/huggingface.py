"""Download the Harvard-GF dataset from its public Hugging Face mirror.

The HF dataset ``harvardairobotics/Harvard-GF`` is public and ungated (no token/ToU). It
ships one ~20.5 GB ``Dataset/dataset.zip`` (all 3,300 ``.npz`` files, each with the 200x200
``rnflt`` map and the 200x200x200 ``oct_bscans`` volume) plus a small
``ReadMe/data_summary.csv`` manifest (decoded demographics + label + split).

Licence: CC BY-NC-ND 4.0 (non-commercial research, no derivatives) — inherent to the
dataset, not the mirror.

The zip is large and this can be a multi-hour transfer, so the downloader is **resumable**:
it streams with an HTTP ``Range`` header from wherever the ``.part`` left off, refreshes the
(1-hour) signed CDN URL on expiry, and retries with backoff across dropped connections.
After download it extracts and (by default) *slims* every ``.npz`` by dropping the 3D
``oct_bscans`` volume, matching the project's RNFLT+VF-only footprint.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

_CHUNK = 1 << 20
_HF_RESOLVE = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def hf_resolve_url(repo_id: str, filename: str) -> str:
    return _HF_RESOLVE.format(repo_id=repo_id, filename=filename)


def _sign(repo_id: str, filename: str) -> tuple[str, int]:
    """Follow the resolve endpoint to a fresh signed CDN URL; return (url, total_size)."""
    resp = requests.get(
        hf_resolve_url(repo_id, filename), stream=True, allow_redirects=True, timeout=(30, 60)
    )
    resp.raise_for_status()
    signed, total = resp.url, int(resp.headers.get("content-length", 0))
    resp.close()
    return signed, total


def _resumable_download(
    repo_id: str, filename: str, dest: Path, retries: int = 10000, backoff: int = 5
) -> Path:
    """Resumable, self-healing download of one repo file to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    signed, total = _sign(repo_id, filename)
    pos = tmp.stat().st_size if tmp.exists() else 0
    attempt = 0
    with tqdm(total=total or None, initial=pos, unit="B", unit_scale=True, desc=dest.name) as bar:
        while not total or pos < total:
            headers = {"Range": f"bytes={pos}-"} if pos else {}
            try:
                with requests.get(
                    signed, headers=headers, stream=True, timeout=(30, 120)
                ) as resp:
                    if resp.status_code == 416:  # range beyond EOF -> already complete
                        break
                    resp.raise_for_status()
                    with open(tmp, "ab" if pos else "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=_CHUNK):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            pos += len(chunk)
                            bar.update(len(chunk))
                if not total:
                    break  # size unknown: one pass
            except (requests.RequestException, OSError) as exc:
                attempt += 1
                if attempt > retries:
                    raise
                time.sleep(min(backoff * attempt, 60))
                try:  # refresh the (possibly expired) signed URL and resume from pos
                    signed, total = _sign(repo_id, filename)
                except requests.RequestException:
                    pass
                tqdm.write(f"[hf] resume {dest.name} from {pos/1e9:.2f} GB (retry {attempt}: {exc})")
    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"size mismatch for {dest.name}: {tmp.stat().st_size} != {total}")
    tmp.replace(dest)
    return dest


def slim_npz(path: Path, drop: tuple[str, ...] = ("oct_bscans",)) -> None:
    """Re-save an ``.npz`` in place keeping every array except those in ``drop``."""
    with np.load(path, allow_pickle=True) as z:
        keep = {k: z[k] for k in z.files if k not in drop}
    tmp = path.with_suffix(".slim.npz")
    np.savez_compressed(tmp, **keep)
    tmp.replace(path)


def slim_all(dest_dir: Path, drop: tuple[str, ...] = ("oct_bscans",)) -> int:
    """Slim every ``.npz`` under ``dest_dir`` (drops the 3D b-scans by default)."""
    npz = sorted(Path(dest_dir).rglob("*.npz"))
    for path in tqdm(npz, desc="slim npz", unit="file"):
        slim_npz(path, drop=drop)
    return len(npz)


def _already_extracted(dest_dir: Path, min_npz: int = 100) -> bool:
    return sum(1 for _ in Path(dest_dir).rglob("*.npz")) >= min_npz


def download_harvard_gf_hf(
    repo_id: str,
    zip_filename: str,
    summary_csv: str,
    dest_dir: Path,
    slim: bool = True,
    keep_zip: bool = False,
    force: bool = False,
) -> Path:
    """Download + extract (+ slim) Harvard-GF from Hugging Face into ``dest_dir``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Small manifest first (cheap, and the loader prefers it).
    try:
        _resumable_download(repo_id, summary_csv, dest_dir / Path(summary_csv).name)
    except requests.RequestException as exc:
        print(f"[hf] could not fetch summary csv ({exc}); loader will fall back to npz fields.")

    if not force and _already_extracted(dest_dir):
        print(f"[hf] Harvard-GF already extracted at {dest_dir}, skipping download.")
    else:
        zip_dest = dest_dir / Path(zip_filename).name
        print(f"[hf] downloading {zip_filename} (~20.5 GB, resumable)...")
        _resumable_download(repo_id, zip_filename, zip_dest)
        print(f"[hf] extracting {zip_dest.name}...")
        with zipfile.ZipFile(zip_dest) as zf:
            zf.extractall(dest_dir)
        if not keep_zip:
            zip_dest.unlink(missing_ok=True)
            print("[hf] removed zip to reclaim disk.")

    # Log the real npz schema once (for verification) before the irreversible slim.
    sample = next(iter(sorted(dest_dir.rglob("*.npz"))), None)
    if sample is not None:
        with np.load(sample, allow_pickle=True) as z:
            dims = {k: getattr(z[k], "shape", ()) for k in z.files}
        print(f"[hf] npz keys in {sample.name}: {dims}")

    if slim:
        print("[hf] slimming npz files (dropping 3D oct_bscans)...")
        n = slim_all(dest_dir)
        print(f"[hf] slimmed {n} npz files.")

    print(f"[hf] Harvard-GF ready at {dest_dir}")
    return dest_dir
