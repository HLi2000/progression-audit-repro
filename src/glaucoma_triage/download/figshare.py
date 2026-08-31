"""Download the PAPILA dataset from figshare (anonymous, CC BY 4.0).

figshare serves PAPILA as a single ~590 MB ``PAPILA.zip``. We resolve the current
version's file(s) via the figshare REST API (robust to version bumps), stream them to
disk with an md5 check, then extract. Re-running is idempotent: an already-downloaded
(md5-verified) archive and an already-extracted tree are both skipped.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
_CHUNK = 1 << 20  # 1 MiB


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    if dest.exists() and expected_md5 and _md5(dest) == expected_md5:
        print(f"[figshare] {dest.name} already present and md5-verified, skipping download.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))
    if expected_md5:
        got = _md5(tmp)
        if got != expected_md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"md5 mismatch for {dest.name}: expected {expected_md5}, got {got}")
    tmp.replace(dest)
    return dest


def _already_extracted(dest_dir: Path, sentinel_names: tuple[str, ...]) -> bool:
    """True if the extracted tree already contains the expected data folders."""
    existing = {p.name for p in dest_dir.rglob("*") if p.is_dir()}
    return all(name in existing for name in sentinel_names)


def download_papila(
    article_id: int,
    dest_dir: Path,
    sentinel_dirs: tuple[str, ...] = ("ClinicalData", "FundusImages"),
    force: bool = False,
) -> Path:
    """Download and extract PAPILA into ``dest_dir``. Returns ``dest_dir``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force and _already_extracted(dest_dir, sentinel_dirs):
        print(f"[figshare] PAPILA already extracted at {dest_dir}, skipping.")
        return dest_dir

    meta = requests.get(FIGSHARE_API.format(article_id=article_id), timeout=60).json()
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(f"figshare article {article_id} lists no files.")

    for finfo in files:
        name = finfo["name"]
        archive = dest_dir / name
        print(f"[figshare] fetching {name} ({finfo.get('size', 0) / 1e6:.0f} MB)...")
        _stream_download(finfo["download_url"], archive, finfo.get("supplied_md5"))
        if zipfile.is_zipfile(archive):
            print(f"[figshare] extracting {name}...")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest_dir)
    print(f"[figshare] PAPILA ready at {dest_dir}")
    return dest_dir
