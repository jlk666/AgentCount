"""Example CLI entrypoint for single and batch plate analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from plate_count_ai.config.settings import settings
from plate_count_ai.utils.batch_report import generate_batch_report
from plate_count_ai.workflows.plate_workflow import PlateWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("plate_count_ai.main")


def _default_metadata() -> dict[str, Any]:
    return {
        "sample_id": "sample_001",
        "dilution": 0.01,
        "volume": 0.1,
        "replicate_id": "r1",
    }


def _collect_batch(batch_dir: Path, metadata_template: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in sorted(batch_dir.iterdir()):
        if path.suffix.lower() in image_exts:
            md = dict(metadata_template)
            md["sample_id"] = path.stem
            items.append({"image_path": str(path), "metadata": md})
    return items


def _load_batch_from_csv(csv_path: Path, metadata_template: dict[str, Any]) -> list[dict[str, Any]]:
    """Load batch items from CSV with columns: image_path + metadata fields."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Batch CSV not found: {csv_path}")

    required_cols = {"image_path"}
    items: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Batch CSV has no header: {csv_path}")
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Batch CSV missing required columns {sorted(missing)}: {csv_path}")

        for idx, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            if not raw_path:
                logger.warning("Skipping row %d with empty image_path", idx)
                continue

            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = (Path.cwd() / image_path).resolve()
            if not image_path.exists():
                logger.warning("Skipping row %d; image does not exist: %s", idx, image_path)
                continue

            md = dict(metadata_template)
            for key, value in row.items():
                if key == "image_path" or value is None:
                    continue
                text = value.strip()
                if text == "":
                    continue
                if key in {"dilution", "volume"}:
                    try:
                        md[key] = float(text)
                    except ValueError:
                        logger.warning("Row %d has non-numeric %s=%r; keeping default %r", idx, key, text, md.get(key))
                else:
                    md[key] = text

            if not md.get("sample_id"):
                md["sample_id"] = image_path.stem
            items.append({"image_path": str(image_path), "metadata": md})
    return items


def _write_batch_summary(outputs: list[dict[str, Any]], output_dir: Path, label: str = "batch") -> Path:
    """Write a clean summary CSV for a completed batch run."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"{label}_summary_{timestamp}.csv"
    fieldnames = ["sample", "replicate", "dilution", "volume_ml", "colony_count", "cfu_per_ml"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for state in outputs:
            md = state.get("metadata") or {}
            writer.writerow({
                "sample": md.get("sample_id", ""),
                "replicate": md.get("replicate_id", ""),
                "dilution": md.get("dilution", ""),
                "volume_ml": md.get("volume", ""),
                "colony_count": state.get("colony_count", ""),
                "cfu_per_ml": state.get("cfu_per_ml", ""),
            })
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate Count AI workflow runner")
    parser.add_argument("--image", type=str, help="Path to one plate image")
    parser.add_argument("--metadata", type=str, help="Metadata JSON string")
    parser.add_argument("--batch-dir", type=str, help="Directory with plate images")
    parser.add_argument("--batch-csv", type=str, help="CSV with image_path and metadata columns")
    parser.add_argument(
        "--sam-backend",
        type=str,
        choices=["sam1", "sam3"],
        help="Choose segmentation backend (sam1 or sam3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for selected backend.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM1 model type when --sam-backend sam1.",
    )
    parser.add_argument(
        "--sam-text-prompt",
        type=str,
        help="Text prompt used by SAM3 backend.",
    )
    args = parser.parse_args()

    config_overrides: dict[str, Any] = {}
    if args.sam_backend:
        config_overrides["sam_backbone"] = args.sam_backend
    if args.checkpoint:
        config_overrides["detector_model_path"] = args.checkpoint
    if args.sam_model_type:
        config_overrides["sam_model_type"] = args.sam_model_type
    if args.sam_text_prompt:
        config_overrides["sam_text_prompt"] = args.sam_text_prompt

    if config_overrides.get("sam_backbone") == "sam1" and "detector_model_path" not in config_overrides:
        config_overrides["detector_model_path"] = "sam_vit_b_01ec64.pth"

    cfg = replace(settings, **config_overrides) if config_overrides else settings
    logger.info("Using backend=%s checkpoint=%s", cfg.sam_backbone, cfg.detector_model_path)
    workflow = PlateWorkflow(cfg)
    metadata = _default_metadata()
    if args.metadata:
        metadata.update(json.loads(args.metadata))

    logo_path = Path(__file__).resolve().parent.parent / "AgentCountLogo.png"

    if args.batch_dir:
        batch_items = _collect_batch(Path(args.batch_dir), metadata)
        if not batch_items:
            logger.warning("No image files found in batch directory: %s", args.batch_dir)
            return
        label = Path(args.batch_dir).name
        logger.info("Running batch analysis for %d images", len(batch_items))
        outputs = workflow.run_batch(batch_items)
        run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        summary_path = _write_batch_summary(outputs, cfg.output_dir, label=label)
        report_path = cfg.output_dir / f"{label}_report_{run_ts}.html"
        generate_batch_report(outputs, report_path, logo_path=logo_path,
                              run_label=label, timestamp=run_ts)
        logger.info("Batch analysis complete. Results produced: %d", len(outputs))
        logger.info("Batch summary CSV: %s", summary_path)
        logger.info("Batch HTML report: %s", report_path)
        return

    if args.batch_csv:
        batch_items = _load_batch_from_csv(Path(args.batch_csv), metadata)
        if not batch_items:
            logger.warning("No valid rows found in batch CSV: %s", args.batch_csv)
            return
        label = Path(args.batch_csv).stem
        logger.info("Running CSV batch analysis for %d images", len(batch_items))
        outputs = workflow.run_batch(batch_items)
        run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        summary_path = _write_batch_summary(outputs, cfg.output_dir, label=label)
        report_path = cfg.output_dir / f"{label}_report_{run_ts}.html"
        generate_batch_report(outputs, report_path, logo_path=logo_path,
                              run_label=label, timestamp=run_ts)
        logger.info("CSV batch analysis complete. Results produced: %d", len(outputs))
        logger.info("Batch summary CSV: %s", summary_path)
        logger.info("Batch HTML report: %s", report_path)
        return

    image_path = args.image
    if image_path is None:
        candidates = sorted(settings.sample_image_dir.glob("*"))
        image_candidates = [p for p in candidates if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        if not image_candidates:
            logger.warning("No sample images found in %s", settings.sample_image_dir)
            logger.info("Provide --image to run a single analysis.")
            return
        image_path = str(image_candidates[0])
        logger.info("Using default sample image: %s", image_path)

    logger.info("Starting plate analysis for image: %s", image_path)
    state = workflow.run(image_path=image_path, metadata=metadata)
    logger.info("Analysis complete. Colony count: %s", state.get("colony_count"))
    logger.info("CFU/mL: %s", state.get("cfu_per_ml"))
    logger.info("Annotated image: %s", state.get("annotated_image_path"))
    logger.info("Result JSON: %s", state.get("result_json_path"))


if __name__ == "__main__":
    main()
