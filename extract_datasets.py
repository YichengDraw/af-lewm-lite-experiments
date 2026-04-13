"""Extract downloaded dataset archives and place .h5 files correctly."""
import os
import sys
import tarfile
from pathlib import Path
import shutil

os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    import zstandard as zstd
except ImportError:
    print("ERROR: zstandard not installed. Run: pip install zstandard")
    sys.exit(1)

CACHE_DIR = Path(os.environ["STABLEWM_HOME"])

EXPECTED_PATHS = {
    "pusht_expert_train": CACHE_DIR / "pusht_expert_train.h5",
    "tworoom": CACHE_DIR / "tworoom.h5",
    "reacher_eval": CACHE_DIR / "dmc" / "reacher_random.h5",
    "reacher_train": CACHE_DIR / "reacher.h5",
    "cube": CACHE_DIR / "ogbench" / "cube_single_expert.h5",
}

EXPECTED_ARCHIVE_BYTES = {
    "pusht_expert_train.h5.zst": 13_136_247_974,
    "tworoom.tar.zst": 3_425_937_909,
    "reacher.tar.zst": 23_750_614_946,
    "cube_single_expert.tar.zst": 46_184_624_478,
}


def extract_tar_zst(archive_path: Path, dest_dir: Path):
    """Extract a .tar.zst archive and return list of extracted file paths."""
    print(f"  Extracting {archive_path.name} ({archive_path.stat().st_size / 1e9:.2f} GB)...")

    dctx = zstd.ZstdDecompressor()
    extracted = []

    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if member.isfile():
                        print(f"    Found: {member.name} ({member.size / 1e9:.2f} GB)")
                        # Extract to dest_dir
                        tar.extract(member, path=dest_dir)
                        extracted.append(dest_dir / member.name)

    return extracted


def extract_h5_zst(archive_path: Path, output_path: Path):
    """Extract a single compressed HDF5 file."""
    print(f"  Extracting {archive_path.name} -> {output_path.name} ({archive_path.stat().st_size / 1e9:.2f} GB compressed)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    backup_path = output_path.with_suffix(output_path.suffix + ".debug.bak")

    with open(archive_path, "rb") as src, open(tmp_path, "wb") as dst:
        dctx.copy_stream(src, dst)

    if output_path.exists() and not backup_path.exists():
        print(f"  BACKUP: {output_path.name} -> {backup_path.name}")
        output_path.replace(backup_path)
    tmp_path.replace(output_path)
    print(f"  Wrote: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path


def ensure_path(src: Path, dst: Path):
    """Move or copy an extracted file into the exact path expected by the configs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        return
    print(f"  PLACE: {src.name} -> {dst.relative_to(CACHE_DIR)}")
    shutil.copy2(src, dst)


def locate_extracted_h5_files():
    return list(CACHE_DIR.rglob("*.h5"))


def main():
    print("=" * 60)
    print("Dataset Extraction & Placement")
    print(f"STABLEWM_HOME: {CACHE_DIR}")
    print("=" * 60)

    archives = list(CACHE_DIR.glob("*.tar.zst")) + list(CACHE_DIR.glob("*.h5.zst"))
    if not archives:
        print("\nNo dataset archives found in STABLEWM_HOME.")
        print("Datasets may still be downloading or may need manual download.")
        print("Check: https://huggingface.co/collections/quentinll/lewm")
        return

    extracted_paths = []
    for archive in archives:
        print(f"\n--- {archive.name} ---")
        expected_bytes = EXPECTED_ARCHIVE_BYTES.get(archive.name)
        if expected_bytes is not None and archive.stat().st_size < expected_bytes:
            progress = archive.stat().st_size / expected_bytes * 100
            print(
                f"  SKIP: download still incomplete "
                f"({progress:.1f}% of expected archive size)"
            )
            continue
        try:
            if archive.name.endswith(".tar.zst"):
                new_paths = extract_tar_zst(archive, CACHE_DIR)
            elif archive.name == "pusht_expert_train.h5.zst":
                new_paths = [extract_h5_zst(archive, EXPECTED_PATHS["pusht_expert_train"])]
            else:
                print(f"  SKIP: unsupported archive format {archive.name}")
                continue
        except Exception as exc:
            print(f"  ERROR: extraction failed for {archive.name}: {exc}")
            print("  This usually means the download is still incomplete.")
            continue

        extracted_paths.extend(new_paths)
        for path in new_paths:
            print(f"  Extracted to: {path}")

    discovered = locate_extracted_h5_files()

    for path in discovered:
        name = path.name
        if name == "tworoom.h5":
            ensure_path(path, EXPECTED_PATHS["tworoom"])
        elif name in {"reacher_random.h5", "reacher.h5"}:
            ensure_path(path, EXPECTED_PATHS["reacher_eval"])
            ensure_path(path, EXPECTED_PATHS["reacher_train"])
        elif name == "cube_single_expert.h5":
            ensure_path(path, EXPECTED_PATHS["cube"])
        elif name == "pusht_expert_train.h5":
            ensure_path(path, EXPECTED_PATHS["pusht_expert_train"])

    print("\n--- Verifying expected dataset paths ---")

    checks = [
        EXPECTED_PATHS["pusht_expert_train"],
        EXPECTED_PATHS["tworoom"],
        EXPECTED_PATHS["reacher_eval"],
        EXPECTED_PATHS["reacher_train"],
        EXPECTED_PATHS["cube"],
    ]

    for expected_path in checks:
        expected_rel = expected_path.relative_to(CACHE_DIR)
        if expected_path.exists():
            size_gb = expected_path.stat().st_size / 1e9
            print(f"  OK: {expected_rel} ({size_gb:.2f} GB)")
        else:
            print(f"  MISSING: {expected_rel}")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)

    # Final status
    print("\nDataset Status:")
    for path in checks:
        name = path.relative_to(CACHE_DIR)
        status = "OK" if path.exists() else "MISSING"
        size = f"({path.stat().st_size / 1e9:.2f} GB)" if path.exists() else ""
        print(f"  [{status}] {name} {size}")


if __name__ == "__main__":
    main()
