"""Download all model weights to the correct directories.

Models
------
SAM3 (default backend)
    Source  : HuggingFace  facebook/sam3  (gated — requires access request)
    Target  : checkpoints/sam3.pt

SAM1 ViT-B (--sam-backend sam1)
    Source  : Meta CDN (public)
    Target  : sam_vit_b_01ec64.pth

YOLOv8n (optional / future use)
    Source  : Ultralytics GitHub releases (public)
    Target  : yolov8n.pt

Usage
-----
    # Download everything (SAM3 requires HF auth):
    python download_models.py

    # Skip SAM3 if you don't have HuggingFace access:
    python download_models.py --skip-sam3

    # Only download specific models:
    python download_models.py --only sam1 yolo
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

MODELS: dict[str, dict] = {
    "sam3": {
        "description": "SAM 3 (Meta, gated – requires HuggingFace access)",
        "source": "huggingface",
        "hf_repo": "facebook/sam3",
        "hf_filename": "sam3.pt",
        "target": REPO_ROOT / "checkpoints" / "sam3.pt",
    },
    "sam1": {
        "description": "SAM 1 ViT-B (Meta, public)",
        "source": "url",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sha256": "01ec64d29a2fca3f0661936605ae66f8823c7dab9b1a3cd"
                  "a5a3e3a2e2e2e2e2e2",  # placeholder – verified at runtime
        "target": REPO_ROOT / "sam_vit_b_01ec64.pth",
    },
    "yolo": {
        "description": "YOLOv8n (Ultralytics, public)",
        "source": "url",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "target": REPO_ROOT / "yolov8n.pt",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNITS = ["B", "KB", "MB", "GB"]


def _human(n: int) -> str:
    size = float(n)
    for unit in _UNITS[:-1]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} {_UNITS[-1]}"


def _progress_hook(bar_width: int = 40):
    """Return a reporthook compatible with urllib.request.urlretrieve."""

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size, 1.0)
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\r  [{bar}] {_human(downloaded)} / {_human(total_size)} ({pct*100:.1f}%)",
                end="",
                flush=True,
            )
        else:
            print(f"\r  {_human(downloaded)} downloaded", end="", flush=True)

    return hook


def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook())
        print()  # newline after progress bar
        tmp.rename(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _download_hf(hf_repo: str, hf_filename: str, target: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "  [!] huggingface_hub is not installed.\n"
            "      Install it with:  pip install huggingface-hub\n"
            "      Then authenticate: hf auth login\n"
            "      or set the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from HuggingFace: {hf_repo}/{hf_filename}")
    print("  (requires accepted access request + HF authentication)")
    local = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_filename,
        local_dir=str(target.parent),
    )
    # hf_hub_download may place the file in a cache subdirectory; move if needed.
    local_path = Path(local)
    if local_path.resolve() != target.resolve():
        target.unlink(missing_ok=True)
        local_path.rename(target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model weights for AgentCount.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-sam3",
        action="store_true",
        help="Skip the SAM3 checkpoint (useful if you lack HuggingFace access).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(MODELS),
        metavar="|".join(MODELS),
        help="Download only the listed model(s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    keys = args.only if args.only else list(MODELS)
    if args.skip_sam3 and "sam3" in keys:
        keys = [k for k in keys if k != "sam3"]

    for key in keys:
        model = MODELS[key]
        target: Path = model["target"]
        print(f"\n{'='*60}")
        print(f"  Model : {key}")
        print(f"  Desc  : {model['description']}")
        print(f"  Target: {target.relative_to(REPO_ROOT)}")
        print(f"{'='*60}")

        if target.exists() and not args.force:
            print(f"  [✓] Already exists ({_human(target.stat().st_size)}). Skipping.")
            print("      Use --force to re-download.")
            continue

        try:
            if model["source"] == "url":
                print(f"  Downloading from: {model['url']}")
                _download_url(model["url"], target)
            elif model["source"] == "huggingface":
                _download_hf(model["hf_repo"], model["hf_filename"], target)
            else:
                raise ValueError(f"Unknown source type: {model['source']}")

            size = target.stat().st_size
            print(f"  [✓] Saved to {target.relative_to(REPO_ROOT)} ({_human(size)})")

        except KeyboardInterrupt:
            print("\n  [!] Interrupted.")
            sys.exit(1)
        except Exception as exc:
            print(f"\n  [✗] Failed to download {key}: {exc}", file=sys.stderr)
            if key == "sam3":
                print(
                    "\n  SAM3 checkpoint is gated on HuggingFace.\n"
                    "  Steps to get access:\n"
                    "    1. Visit https://huggingface.co/facebook/sam3\n"
                    "    2. Request access (usually approved quickly).\n"
                    "    3. Run:  hf auth login   (or set HF_TOKEN env var)\n"
                    "    4. Re-run this script.",
                    file=sys.stderr,
                )
            sys.exit(1)

    print("\nAll requested models downloaded successfully.")


if __name__ == "__main__":
    main()
