"""P1-T5: acquire, extract, and verify the RadioML 2018.01A dataset.

The official DeepSig download currently requires a short browser form. This
tool does not bypass that step. It prepares the local directory, optionally
downloads from a direct URL obtained by the user, safely extracts the HDF5
file, validates its structure without loading X into RAM, and writes a
reproducibility manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "RML2018.01a"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "P1-T5"
MANIFEST_PATH = ARTIFACT_DIR / "dataset_manifest.json"

OFFICIAL_PAGE = "https://www.deepsig.ai/datasets/"
LICENSE_NAME = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
DEFAULT_ARCHIVE_NAME = "2018.01.OSC.0001_1024x2M.h5.tar.gz"
ZIP_ARCHIVE_NAME = "RadioML201801A.zip"
HDF5_NAME = "GOLD_XYZ_OSC.0001_1024.hdf5"
CLASSES_NAME = "classes.txt"
FIXED_CLASSES_JSON_NAME = "classes-fixed.json"
FIXED_CLASSES_TEXT_NAME = "classes-fixed.txt"
LICENSE_FILE_NAME = "LICENSE.TXT"
EXPECTED_ARCHIVE_FILES = (
    HDF5_NAME,
    CLASSES_NAME,
    FIXED_CLASSES_JSON_NAME,
    FIXED_CLASSES_TEXT_NAME,
    LICENSE_FILE_NAME,
)

EXPECTED_LAYOUT = {
    "X": (2_555_904, 1024, 2),
    "Y": (2_555_904, 24),
    "Z": (2_555_904, 1),
}

# Keep enough room for both the compressed archive, extracted HDF5 file, and
# temporary extraction file. This is a safety threshold, not the dataset size.
MIN_FREE_GIB_FOR_EXTRACTION = 45.0
MIN_HDF5_SIZE_GIB = 19.0
HASH_CHUNK_SIZE = 8 * 1024 * 1024
COPY_CHUNK_SIZE = 8 * 1024 * 1024


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gib(size_bytes: int) -> float:
    return round(size_bytes / 1024**3, 3)


def prepare_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def redact_url(url: str) -> str:
    """Remove query parameters and fragments before saving a URL in metadata."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def ensure_free_space(required_gib: float = MIN_FREE_GIB_FOR_EXTRACTION) -> float:
    free_bytes = shutil.disk_usage(RAW_DIR).free
    free_gib = gib(free_bytes)
    if free_gib < required_gib:
        raise RuntimeError(
            f"Insufficient disk space: {free_gib:.2f} GiB free, "
            f"at least {required_gib:.2f} GiB required."
        )
    return free_gib


def download_archive(url: str, output: Path) -> Path:
    """Download with curl resume support; the URL must be obtained legitimately."""
    prepare_directories()
    ensure_free_space()

    if not shutil.which("curl"):
        raise RuntimeError("curl is not installed; download the archive in a browser.")

    output = output.resolve()
    if RAW_DIR.resolve() not in output.parents:
        raise ValueError(f"Download output must be inside {RAW_DIR}")

    partial = output.with_name(output.name + ".part")
    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    print(f"Downloading to {partial} (resume enabled)...")
    subprocess.run(command, check=True)

    with partial.open("rb") as file:
        magic = file.read(2)
    if magic not in {b"\x1f\x8b", b"PK"}:
        raise RuntimeError(
            "The downloaded content is not a gzip or ZIP archive. The URL may point "
            "to DeepSig's registration form instead of the authorized file. "
            f"The response was kept at {partial} for inspection."
        )

    os.replace(partial, output)
    print(f"Download completed: {output}")
    print(f"Recorded source URL: {redact_url(url)}")
    return output


def locate_archive(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        archive = explicit_path.expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Archive not found: {archive}")
        return archive

    preferred = RAW_DIR / DEFAULT_ARCHIVE_NAME
    if preferred.is_file():
        return preferred
    preferred_zip = RAW_DIR / ZIP_ARCHIVE_NAME
    if preferred_zip.is_file():
        return preferred_zip

    candidates = sorted(
        path
        for pattern in ("*.tar.gz", "*.tgz", "*.zip")
        for path in RAW_DIR.glob(pattern)
        if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No .tar.gz or .zip archive found in {RAW_DIR}. Download it from "
            f"{OFFICIAL_PAGE} first."
        )
    raise RuntimeError(
        "Multiple archives found; select one with --archive:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        block = source.read(COPY_CHUNK_SIZE)
        if not block:
            break
        destination.write(block)


def extract_stream(
    source: BinaryIO,
    member_name: str,
    member_size: int,
    basename: str,
    overwrite: bool,
) -> None:
    """Write one validated archive member through an atomic partial file."""
    file_destination = RAW_DIR / basename
    if file_destination.exists() and not overwrite:
        print(f"Existing file kept: {file_destination}")
        return

    partial = file_destination.with_name(file_destination.name + ".part")
    if partial.exists():
        partial.unlink()
    print(
        f"Extracting {member_name} ({gib(member_size):.3f} GiB) "
        f"to {file_destination}..."
    )
    try:
        with partial.open("wb") as output:
            copy_stream(source, output)
        if partial.stat().st_size != member_size:
            raise RuntimeError(
                "Extracted file size does not match the archive member: "
                f"{partial.stat().st_size} != {member_size}"
            )
        os.replace(partial, file_destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def validate_member_counts(members_by_name: dict[str, list[Any]]) -> None:
    hdf5_matches = members_by_name[HDF5_NAME]
    if len(hdf5_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {HDF5_NAME} in the archive; "
            f"found {len(hdf5_matches)}."
        )
    for basename, matches in members_by_name.items():
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected at most one {basename}; found {len(matches)}."
            )


def extract_dataset_files(archive: Path, overwrite: bool = False) -> Path:
    """Extract expected files from TAR.GZ or ZIP without trusting member paths."""
    prepare_directories()
    ensure_free_space()
    destination = RAW_DIR / HDF5_NAME

    print(f"Inspecting archive: {archive}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, mode="r") as zipped:
            members_by_name: dict[str, list[zipfile.ZipInfo]] = {
                name: [] for name in EXPECTED_ARCHIVE_FILES
            }
            for member in zipped.infolist():
                basename = Path(member.filename).name
                if not member.is_dir() and basename in members_by_name:
                    members_by_name[basename].append(member)
            validate_member_counts(members_by_name)

            for basename in EXPECTED_ARCHIVE_FILES:
                matches = members_by_name[basename]
                if not matches:
                    print(f"WARNING: optional archive member not found: {basename}")
                    continue
                member = matches[0]
                with zipped.open(member, mode="r") as source:
                    extract_stream(
                        source,
                        member.filename,
                        member.file_size,
                        basename,
                        overwrite,
                    )
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, mode="r:*") as tar:
            tar_members: dict[str, list[tarfile.TarInfo]] = {
                name: [] for name in EXPECTED_ARCHIVE_FILES
            }
            for member in tar.getmembers():
                basename = Path(member.name).name
                if member.isfile() and basename in tar_members:
                    tar_members[basename].append(member)
            validate_member_counts(tar_members)

            for basename in EXPECTED_ARCHIVE_FILES:
                matches = tar_members[basename]
                if not matches:
                    print(f"WARNING: optional archive member not found: {basename}")
                    continue
                member = matches[0]
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                with source:
                    extract_stream(
                        source,
                        member.name,
                        member.size,
                        basename,
                        overwrite,
                    )
    else:
        raise RuntimeError(f"Unsupported or damaged archive: {archive}")

    print(f"Dataset extraction completed: {RAW_DIR}")
    return destination


def locate_hdf5(explicit_path: Path | None = None) -> Path:
    path = (
        explicit_path.expanduser().resolve()
        if explicit_path is not None
        else RAW_DIR / HDF5_NAME
    )
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    processed = 0
    next_report = 1024**3

    print(f"Computing SHA-256 for {path.name} ({gib(total):.3f} GiB)...")
    with path.open("rb") as file:
        while block := file.read(HASH_CHUNK_SIZE):
            digest.update(block)
            processed += len(block)
            if processed >= next_report:
                print(f"  hashed {gib(processed):.1f}/{gib(total):.1f} GiB")
                next_report += 1024**3
    return digest.hexdigest()


def inspect_classes_file() -> tuple[dict[str, Any] | None, list[str]]:
    fixed_json = RAW_DIR / FIXED_CLASSES_JSON_NAME
    original_text = RAW_DIR / CLASSES_NAME
    path = fixed_json if fixed_json.is_file() else original_text
    if not path.is_file():
        return None, [
            f"Missing class-order file: {FIXED_CLASSES_JSON_NAME} or {CLASSES_NAME}"
        ]

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    classes: list[str] = []
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                classes = [str(item).strip() for item in parsed]
        except json.JSONDecodeError:
            pass
    else:
        # The original file is commonly a Python assignment such as
        # ``classes = [...]``, rather than a bare list.
        try:
            module = ast.parse(text)
            for node in module.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = ast.literal_eval(node.value)
                    if isinstance(value, (list, tuple)):
                        classes = [str(item).strip() for item in value]
                        break
        except (SyntaxError, ValueError):
            pass

    errors = []
    if len(classes) != 24:
        errors.append(f"Expected 24 class names in {path.name}; found {len(classes)}")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "source_file": path.name,
        "count": len(classes),
        "classes": classes,
    }, errors


def inspect_hdf5(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Inspect metadata only; never materialize the full X array."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "h5py and numpy are required. Run this script in qam-mamba."
        ) from exc

    datasets: dict[str, Any] = {}
    errors: list[str] = []
    with h5py.File(path, "r") as file:
        root_keys = sorted(file.keys())
        for name, expected_shape in EXPECTED_LAYOUT.items():
            if name not in file:
                errors.append(f"Missing root dataset: {name}")
                continue
            dataset = file[name]
            shape = tuple(int(value) for value in dataset.shape)
            datasets[name] = {
                "shape": list(shape),
                "dtype": str(dataset.dtype),
                "chunks": list(dataset.chunks) if dataset.chunks else None,
                "compression": dataset.compression,
            }
            if shape != expected_shape:
                errors.append(
                    f"{name} shape mismatch: expected {expected_shape}, got {shape}"
                )
            if not np.issubdtype(dataset.dtype, np.number):
                errors.append(f"{name} dtype is not numeric: {dataset.dtype}")

        datasets["root_keys"] = root_keys

        # Read tiny boundary slices to detect truncation without loading arrays.
        if not errors:
            first_x = file["X"][0:1]
            last_x = file["X"][-1:]
            first_y = file["Y"][0:1]
            last_z = file["Z"][-1:]
            if not all(
                np.isfinite(array).all()
                for array in (first_x, last_x, first_y, last_z)
            ):
                errors.append("Boundary samples contain NaN or Inf values")

    return datasets, errors


def verify_dataset(path: Path, skip_sha256: bool = False) -> dict[str, Any]:
    prepare_directories()
    datasets, errors = inspect_hdf5(path)
    file_size_gib = gib(path.stat().st_size)
    if file_size_gib < MIN_HDF5_SIZE_GIB:
        errors.append(
            f"HDF5 file is unexpectedly small: {file_size_gib:.3f} GiB; "
            f"expected at least {MIN_HDF5_SIZE_GIB:.1f} GiB"
        )
    classes_info, classes_errors = inspect_classes_file()
    errors.extend(classes_errors)
    license_path = RAW_DIR / LICENSE_FILE_NAME
    if not license_path.is_file():
        errors.append(f"Missing publisher license file: {LICENSE_FILE_NAME}")
    checksum = None if skip_sha256 else sha256_file(path)

    manifest: dict[str, Any] = {
        "dataset": "RADIOML 2018.01A",
        "verified_at": now_iso(),
        "source": {
            "publisher": "DeepSig Inc.",
            "official_page": OFFICIAL_PAGE,
            "acquisition_method": "Official browser form or user-supplied authorized direct URL",
        },
        "license": {
            "name": LICENSE_NAME,
            "url": LICENSE_URL,
            "commercial_use_allowed": False,
        },
        "file": {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": path.stat().st_size,
            "size_gib": file_size_gib,
            "sha256": checksum,
            "official_checksum_available": False,
            "checksum_note": (
                "Local reproducibility fingerprint; no official checksum was "
                "published on the source page at verification time."
            ),
        },
        "hdf5": datasets,
        "class_order": classes_info,
        "license_file": (
            str(license_path.relative_to(PROJECT_ROOT))
            if license_path.is_file()
            else None
        ),
        "expected_layout": {
            name: list(shape) for name, shape in EXPECTED_LAYOUT.items()
        },
        "validation": {
            "passed": not errors,
            "errors": errors,
            "full_arrays_loaded_into_memory": False,
        },
        "limitations": [
            "DeepSig describes this as a historical research dataset with known errata.",
            "The dataset is synthetic and does not replace validation on real over-the-air data.",
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nManifest saved to: {MANIFEST_PATH}")

    if errors:
        raise RuntimeError("Dataset verification failed:\n- " + "\n- ".join(errors))
    return manifest


def print_initialization_status() -> None:
    prepare_directories()
    free_gib = gib(shutil.disk_usage(RAW_DIR).free)
    print("P1-T5 directory initialized.")
    print(f"Official download page: {OFFICIAL_PAGE}")
    print(f"Save the archive in:   {RAW_DIR}")
    print(f"Accepted archives:     {DEFAULT_ARCHIVE_NAME} or {ZIP_ARCHIVE_NAME}")
    print(f"Expected HDF5 name:    {HDF5_NAME}")
    print(f"Current free space:    {free_gib:.2f} GiB")
    print(f"License:               {LICENSE_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and verify the RadioML 2018.01A dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create directories and show instructions.")

    download = subparsers.add_parser(
        "download", help="Download from a direct URL obtained after the official form."
    )
    download.add_argument("--url", required=True, help="Authorized direct download URL.")
    download.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / DEFAULT_ARCHIVE_NAME,
        help="Archive output path inside the raw dataset directory.",
    )

    extract = subparsers.add_parser(
        "extract", help="Safely extract only the expected HDF5 file."
    )
    extract.add_argument("--archive", type=Path, help="Path to the downloaded archive.")
    extract.add_argument(
        "--overwrite", action="store_true", help="Replace an existing HDF5 file."
    )

    verify = subparsers.add_parser(
        "verify", help="Validate HDF5 structure and write the dataset manifest."
    )
    verify.add_argument("--hdf5", type=Path, help="Path to the HDF5 file.")
    verify.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the slow hash only for a quick preliminary check.",
    )

    run = subparsers.add_parser("run", help="Extract and perform full verification.")
    run.add_argument("--archive", type=Path, help="Path to the downloaded archive.")
    run.add_argument(
        "--overwrite", action="store_true", help="Replace an existing HDF5 file."
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "init":
            print_initialization_status()
        elif args.command == "download":
            download_archive(args.url, args.output)
        elif args.command == "extract":
            archive = locate_archive(args.archive)
            extract_dataset_files(archive, overwrite=args.overwrite)
        elif args.command == "verify":
            hdf5 = locate_hdf5(args.hdf5)
            verify_dataset(hdf5, skip_sha256=args.skip_sha256)
        elif args.command == "run":
            archive = locate_archive(args.archive)
            hdf5 = extract_dataset_files(archive, overwrite=args.overwrite)
            verify_dataset(hdf5, skip_sha256=False)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
