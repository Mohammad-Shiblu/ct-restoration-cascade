"""
download_lodopab.py
===================
Download the LoDoPaB-CT dataset from Zenodo (record 3384092).

Change DATA_PATH below to wherever you want the files saved.

Usage
-----
    conda run -n tensor python data_prep/download_lodopab.py
    conda run -n tensor python data_prep/download_lodopab.py --parts train
    conda run -n tensor python data_prep/download_lodopab.py --parts train validation test

The total dataset is ~55 GB compressed (~114 GB extracted).
Each split can be downloaded and extracted independently.

Zenodo stores each split as two zip archives:
    observation_{part}.zip     — noisy post-log sinograms
    ground_truth_{part}.zip    — clean CT reconstructions

After extraction each zip produces files named:
    observation_{part}_{idx:03d}.hdf5    shape (128, 1000, 513)
    ground_truth_{part}_{idx:03d}.hdf5   shape (128, 362, 362)

Each HDF5 file holds up to 128 samples under the key 'data'.
Zip files are deleted after successful extraction to save space.
"""

import argparse
import os
import time
import zipfile
from pathlib import Path

import requests

# ── Override with LODOPAB_DATA_PATH env var, or edit the fallback below ──────

DATA_PATH = Path(
    os.environ.get(
        "LODOPAB_DATA_PATH",
        Path(__file__).parent.parent / "medical_image_datasets" / "lodopab",
    )
)

# ─────────────────────────────────────────────────────────────────────────────

ZENODO_RECORD_ID = "3384092"
ZENODO_API_URL   = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def _zenodo_files() -> dict:
    """Fetch filename → download-URL mapping from Zenodo API."""
    print(f"Fetching Zenodo record {ZENODO_RECORD_ID} file list …")
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    files = {}
    for f in resp.json().get("files", []):
        files[f["key"]] = f["links"]["self"]
    return files


def _download(url: str, dest: Path, chunk_size: int = 1 << 20):
    """Download url → dest with a progress bar. Resumes if partial file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers  = {}
    existing = dest.stat().st_size if dest.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    resp = requests.get(url, stream=True, headers=headers, timeout=60)
    if resp.status_code == 416:          # range not satisfiable → already complete
        print(f"  [skip] {dest.name}  (already complete)")
        return
    resp.raise_for_status()

    total      = int(resp.headers.get("Content-Length", 0)) + existing
    mode       = "ab" if existing else "wb"
    downloaded = existing
    t0         = time.time()

    with open(dest, mode) as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - t0, 1e-3)
                speed   = (downloaded - existing) / elapsed / 1e6   # MB/s
                pct     = 100 * downloaded / total if total else 0
                print(
                    f"\r  {dest.name}  {downloaded/1e9:.2f}/{total/1e9:.2f} GB"
                    f"  {pct:.1f}%  {speed:.1f} MB/s",
                    end="", flush=True,
                )
    print()


def _extract(zip_path: Path, dest_dir: Path):
    """Extract a zip archive with a simple progress indicator, then delete it."""
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            print(f"\r    {i}/{len(members)} files extracted", end="", flush=True)
    print()
    zip_path.unlink()
    print(f"  Deleted {zip_path.name}")


def _hdf5_present(data_path: Path, prefix: str, part: str) -> bool:
    """Return True if at least the first expected HDF5 file already exists."""
    return (data_path / f"{prefix}_{part}_000.hdf5").exists()


# ── Main ──────────────────────────────────────────────────────────────────────

def download(parts=("train", "validation", "test")):
    data_path = DATA_PATH
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory : {data_path}\n")

    all_files = _zenodo_files()
    if not all_files:
        raise RuntimeError("No files found in Zenodo record — check your internet connection.")

    print(f"Zenodo record has {len(all_files)} files total.\n")

    for part in parts:
        if part not in ("train", "validation", "test"):
            raise ValueError(f"Unknown part '{part}'. Choose from: train, validation, test")

        print(f"=== {part} ===")

        for prefix in ("observation", "ground_truth"):
            zip_name = f"{prefix}_{part}.zip"
            zip_dest = data_path / zip_name

            # Skip entirely if HDF5 files already extracted
            if _hdf5_present(data_path, prefix, part):
                print(f"  [skip] {prefix}_{part}  (HDF5 files already present)")
                continue

            if zip_name not in all_files:
                print(f"  [warn] {zip_name} not in Zenodo record — skipping")
                continue

            # Download zip (resumes if partial)
            print(f"  Downloading {zip_name} …")
            _download(all_files[zip_name], zip_dest)

            # Extract and delete zip
            _extract(zip_dest, data_path)

        # Optional CSV
        csv_name = f"patient_ids_rand_{part}.csv"
        if csv_name in all_files:
            dest = data_path / csv_name
            if not dest.exists():
                print(f"  Downloading {csv_name} …")
                _download(all_files[csv_name], dest)

        print(f"  Done — {part}\n")

    print("All requested parts downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LoDoPaB-CT from Zenodo.")
    parser.add_argument(
        "--parts", nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="Which splits to download (default: all three)",
    )
    args = parser.parse_args()
    download(parts=args.parts)
