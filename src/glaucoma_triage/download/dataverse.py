"""Download the Harvard-GF dataset from Harvard Dataverse (gated).

Harvard-GF is access-gated: the anonymous API returns an empty file list, so files are
only visible once you have accepted the dataset's Terms of Use and hold an API token.

Two paths:
  * With ``DATAVERSE_TOKEN`` set, we list the dataset version and stream each file to
    ``dest_dir`` via the native access API.
  * Without a token, we print clear manual-download instructions and return without
    error, so the pipeline can still be developed/tested against manually-placed files.

The downstream loader only ever reads ``rnflt``/``md``/``tds``/demographics/label from
each ``.npz`` — the 200x200x200 ``oct_bscans`` are left on disk (or absent) untouched.
"""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

_CHUNK = 1 << 20


def _manual_instructions(doi: str, base: str, dest_dir: Path) -> None:
    print(
        "\n".join(
            [
                "",
                "=" * 78,
                "Harvard-GF requires manual access (gated on Harvard Dataverse).",
                "=" * 78,
                f"  1. Open: {base}/dataset.xhtml?persistentId={doi}",
                "  2. Log in / register, then accept the dataset Terms of Use.",
                "  3. Either:",
                "       a) Download the files/zip from the web UI, OR",
                "       b) Create an API token (Account -> API Token) and set",
                "          DATAVERSE_TOKEN in your .env, then re-run this script.",
                "  4. Place/extract the .npz files under:",
                f"       {dest_dir}",
                "     (any nested layout is fine; the loader globs for *.npz and infers",
                "      the split from the parent folder name).",
                "=" * 78,
                "",
            ]
        )
    )


def _list_files(base: str, doi: str, token: str) -> list[dict]:
    url = f"{base}/api/datasets/:persistentId/versions/:latest"
    resp = requests.get(
        url, params={"persistentId": doi}, headers={"X-Dataverse-key": token}, timeout=60
    )
    resp.raise_for_status()
    return resp.json().get("data", {}).get("files", [])


def _download_datafile(base: str, file_id: int, dest: Path, token: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    url = f"{base}/api/access/datafile/{file_id}"
    with requests.get(
        url, headers={"X-Dataverse-key": token}, stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)


def download_harvard_gf(
    doi: str,
    base: str,
    dest_dir: Path,
    token: str | None = None,
) -> Path:
    """Download Harvard-GF into ``dest_dir`` if a token is available.

    With no token, prints manual instructions and returns ``dest_dir`` unchanged.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not token:
        _manual_instructions(doi, base, dest_dir)
        return dest_dir

    files = _list_files(base, doi, token)
    if not files:
        print("[dataverse] token set but no files returned — check ToU acceptance / token.")
        _manual_instructions(doi, base, dest_dir)
        return dest_dir

    print(f"[dataverse] {len(files)} file(s) found; downloading to {dest_dir}...")
    for entry in files:
        data_file = entry.get("dataFile", {})
        file_id = data_file.get("id")
        name = data_file.get("filename", f"file_{file_id}")
        subdir = entry.get("directoryLabel", "")
        target = dest_dir / subdir / name if subdir else dest_dir / name
        _download_datafile(base, file_id, target, token)
    print(f"[dataverse] Harvard-GF ready at {dest_dir}")
    return dest_dir
