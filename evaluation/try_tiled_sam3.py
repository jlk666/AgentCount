"""Development-only overlapping-tile SAM3 experiment.

This is a diagnostic run, not a replacement for the frozen holdout baseline.
It uses the baseline model, prompt, enhancement, and mask filters on each tile.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plate_count_ai.config.settings import settings
from plate_count_ai.models.colony_detector import ColonyDetector
from plate_count_ai.utils.image_utils import crop_to_circle, detect_plate_circle, load_image
from plate_count_ai.utils.visualization_utils import draw_annotations, save_image


HERE = Path(__file__).resolve().parent
def starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, tile_size - overlap))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def ownership_bounds(positions: list[int], index: int, tile_length: int, full_length: int) -> tuple[float, float]:
    left = 0.0 if index == 0 else (positions[index - 1] + tile_length + positions[index]) / 2
    right = float(full_length) if index == len(positions) - 1 else (positions[index] + tile_length + positions[index + 1]) / 2
    return left, right


def run_one(detector: ColonyDetector, row: dict[str, str], tile_size: int, overlap: int, out: Path, verbose: bool) -> dict:
    image = load_image(str(ROOT / row["image_path"]))
    circle = detect_plate_circle(image)
    if circle["found"] and circle["radius"] is not None:
        plate = crop_to_circle(image, circle["center_x"], circle["center_y"], circle["radius"])
    else:
        plate = image
    height, width = plate.shape[:2]
    xs, ys = starts(width, tile_size, overlap), starts(height, tile_size, overlap)
    all_detections = []
    diagnostics = []

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tile = plate[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            detections, raw_masks = detector.detect_with_intermediate(tile)
            xlo, xhi = ownership_bounds(xs, xi, tile.shape[1], width)
            ylo, yhi = ownership_bounds(ys, yi, tile.shape[0], height)
            owned = []
            for det in detections:
                box = det["bbox"]
                center_x = x + (box[0] + box[2]) / 2
                center_y = y + (box[1] + box[3]) / 2
                if xlo <= center_x < xhi and ylo <= center_y < yhi:
                    shifted = dict(det)
                    shifted["bbox"] = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
                    owned.append(shifted)
            all_detections.extend(owned)
            diagnostics.append({
                "x": x, "y": y, "width": tile.shape[1], "height": tile.shape[0],
                "raw_masks": len(raw_masks), "accepted_local": len(detections),
                "accepted_owned": len(owned),
            })
            if verbose:
                print(f"  tile ({xi + 1}/{len(xs)}, {yi + 1}/{len(ys)}): raw={len(raw_masks)} accepted={len(detections)} owned={len(owned)}", flush=True)
            del raw_masks

    final = detector._nms(all_detections)
    annotated = draw_annotations(plate, final, len(final), None, None, None, [])
    overlay_path = save_image(annotated, out / f"{row['plate_id']}_tiled.png")
    return {
        "plate_id": row["plate_id"], "image_path": row["image_path"],
        "baseline_count": row["colony_count"], "tiled_count": len(final),
        "raw_masks": sum(t["raw_masks"] for t in diagnostics),
        "accepted_before_ownership": sum(t["accepted_local"] for t in diagnostics),
        "tiles": diagnostics, "plate_shape": [height, width],
        "tile_size": tile_size, "overlap": overlap,
        "circle": circle, "overlay_path": overlay_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plate_ids", nargs="*", help="Development plate IDs only")
    parser.add_argument("--all-development", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=252)
    args = parser.parse_args()
    if not 0 <= args.overlap < args.tile_size:
        raise ValueError("Overlap must be nonnegative and smaller than tile size")
    with (HERE / "development_predictions.csv").open(newline="", encoding="utf-8") as handle:
        dev = {row["plate_id"]: row for row in csv.DictReader(handle)}
    plate_ids = sorted(dev) if args.all_development else args.plate_ids
    if not plate_ids:
        parser.error("Supply plate IDs or --all-development")
    missing = set(plate_ids) - set(dev)
    if missing:
        raise ValueError(f"Not in development split: {', '.join(sorted(missing))}")

    out = HERE / ("tiled_dev" if (args.tile_size, args.overlap) == (1008, 252)
                  else f"tiled_dev_{args.tile_size}_{args.overlap}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = replace(settings, sam_backbone="sam3", detector_model_path="checkpoints/sam3.pt")
    detector = ColonyDetector(
        model_path=cfg.detector_model_path, device=cfg.device,
        backbone=cfg.sam_backbone, text_prompt=cfg.sam_text_prompt,
        text_prompts_raw=cfg.sam_text_prompts,
        enable_image_enhancement=cfg.sam_enable_image_enhancement,
        min_colony_area_ratio=cfg.min_colony_area_ratio,
        max_colony_area_ratio=cfg.max_colony_area_ratio,
        min_colony_circularity=cfg.min_colony_circularity,
        nms_iou_threshold=cfg.nms_iou_threshold,
    )
    for index, plate_id in enumerate(plate_ids, 1):
        if args.skip_existing and (out / f"{plate_id}.json").exists():
            print(f"[{index}/{len(plate_ids)}] {plate_id}: already done", flush=True)
            continue
        print(f"[{index}/{len(plate_ids)}] {plate_id}", flush=True)
        result = run_one(detector, dev[plate_id], args.tile_size, args.overlap, out, verbose=not args.all_development)
        (out / f"{plate_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  baseline={result['baseline_count']} tiled={result['tiled_count']} raw={result['raw_masks']}", flush=True)


if __name__ == "__main__":
    main()
