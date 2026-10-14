"""FastAPI server for plate counting inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from plate_count_ai import __version__
from plate_count_ai.config.settings import settings
from plate_count_ai.workflows.plate_workflow import PlateWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Plate Count AI", version=__version__)
workflow = PlateWorkflow(settings)


def _to_serializable_result(state: dict[str, Any]) -> dict[str, Any]:
    result = dict(state.get("result", {}))
    if not result:
        result = {
            "image_path": state.get("image_path"),
            "metadata": state.get("metadata"),
            "qc_status": state.get("qc_status"),
            "colony_count": state.get("colony_count"),
            "cfu_per_ml": state.get("cfu_per_ml"),
            "validation_status": state.get("validation_status"),
            "warnings": state.get("warnings", []),
            "annotated_image_path": state.get("annotated_image_path"),
        }
    return result


@app.post("/analyze_plate")
async def analyze_plate(
    image: UploadFile = File(...),
    metadata: str = Form(...),
) -> JSONResponse:
    """
    Analyze one agar plate image.

    metadata must be a JSON string containing:
    {
      "sample_id": "...",
      "dilution": 0.01,
      "volume": 0.1,
      "replicate_id": "r1"
    }
    """
    try:
        metadata_obj = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc
    if not isinstance(metadata_obj, dict):
        raise HTTPException(status_code=400, detail="Metadata must be a JSON object")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image file extension")
    upload_path = settings.upload_dir / f"{uuid4().hex}{suffix}"
    data = await image.read()
    upload_path.write_bytes(data)
    logger.info("Saved upload %s", upload_path.name)

    try:
        result_state = workflow.run(str(upload_path), metadata_obj)
    except Exception as exc:
        logger.exception("Plate analysis failed")
        raise HTTPException(status_code=500, detail="Plate analysis failed") from exc

    result = _to_serializable_result(result_state)
    for key in ("image_path", "annotated_image_path", "sam_masks_image_path"):
        if result.get(key):
            result[key] = Path(result[key]).name
    return JSONResponse(
        content={
            "status": "ok",
            "result": result,
            "annotated_image_path": result.get("annotated_image_path"),
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
