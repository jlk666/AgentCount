"""FastAPI server for plate counting inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from plate_count_ai.config.settings import settings
from plate_count_ai.workflows.plate_workflow import PlateWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Plate Count AI", version="0.1.0")
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

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = settings.upload_dir / image.filename
    data = await image.read()
    upload_path.write_bytes(data)
    logger.info("Saved upload to %s", upload_path)

    try:
        result_state = workflow.run(str(upload_path), metadata_obj)
    except Exception as exc:
        logger.exception("Plate analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = _to_serializable_result(result_state)
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
