"""Example CLI entrypoint for single and batch plate analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Any

from plate_count_ai.config.settings import settings
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate Count AI workflow runner")
    parser.add_argument("--image", type=str, help="Path to one plate image")
    parser.add_argument("--metadata", type=str, help="Metadata JSON string")
    parser.add_argument("--batch-dir", type=str, help="Directory with plate images")
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

    if args.batch_dir:
        batch_items = _collect_batch(Path(args.batch_dir), metadata)
        if not batch_items:
            logger.warning("No image files found in batch directory: %s", args.batch_dir)
            return
        logger.info("Running batch analysis for %d images", len(batch_items))
        outputs = workflow.run_batch(batch_items)
        logger.info("Batch analysis complete. Results produced: %d", len(outputs))
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
