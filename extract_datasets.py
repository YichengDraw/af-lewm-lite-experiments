"""Extract the official PushT dataset archive into STABLEWM_HOME."""
import os
import sys
from pathlib import Path


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
ARCHIVE_PATH = CACHE_DIR / "pusht_expert_train.h5.zst"
OUTPUT_PATH = CACHE_DIR / "pusht_expert_train.h5"
EXPECTED_ARCHIVE_BYTES = 13_136_247_974


def extract_h5_zst(archive_path: Path, output_path: Path) -> None:
    print(f"Extracting {archive_path.name} -> {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    backup_path = output_path.with_suffix(output_path.suffix + ".debug.bak")

    dctx = zstd.ZstdDecompressor()
    with archive_path.open("rb") as src, tmp_path.open("wb") as dst:
        dctx.copy_stream(src, dst)

    if output_path.exists() and not backup_path.exists():
        print(f"Backing up existing dataset: {output_path.name} -> {backup_path.name}")
        output_path.replace(backup_path)
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    print("=" * 60)
    print("Official PushT Dataset Extraction")
    print(f"STABLEWM_HOME: {CACHE_DIR}")
    print("=" * 60)

    if OUTPUT_PATH.exists():
        print(f"Dataset already present: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e9:.2f} GB)")
        return

    if not ARCHIVE_PATH.exists():
        print(f"Missing archive: {ARCHIVE_PATH}")
        print("Run tools/download_official_datasets.ps1 first.")
        sys.exit(1)

    if ARCHIVE_PATH.stat().st_size < EXPECTED_ARCHIVE_BYTES:
        progress = ARCHIVE_PATH.stat().st_size / EXPECTED_ARCHIVE_BYTES * 100
        print(f"Archive is incomplete: {progress:.1f}% of expected size")
        sys.exit(1)

    extract_h5_zst(ARCHIVE_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
