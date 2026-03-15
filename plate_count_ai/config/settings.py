"""Central runtime settings for the plate counting system."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch


@dataclass(slots=True)
class Settings:
    """Application configuration with sensible research defaults."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    upload_dir: Path = data_dir / "uploads"
    sample_image_dir: Path = data_dir / "sample_images"

    # Model configuration (Meta Segment Anything)
    sam_backbone: str = os.getenv("SAM_BACKBONE", "sam3")
    detector_model_path: str = os.getenv("SAM_CHECKPOINT", "checkpoints/sam3.pt")
    sam_model_type: str = "vit_b"
    sam_model_config: str = os.getenv("SAM_MODEL_CONFIG", "configs/sam3/sam3_hiera_b+.yaml")
    sam_text_prompt: str = os.getenv("SAM_TEXT_PROMPT", "bacterial colonies")
    sam_text_prompts: str = os.getenv("SAM_TEXT_PROMPTS", "")
    sam_enable_image_enhancement: bool = os.getenv("SAM_ENABLE_IMAGE_ENHANCEMENT", "1") == "1"
    sam_checkpoint_url: str = os.getenv(
        "SAM_CHECKPOINT_URL",
        "",
    )
    auto_download_sam_checkpoint: bool = os.getenv("SAM_AUTO_DOWNLOAD", "0") == "1"
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.92
    sam_min_mask_region_area: int = 20
    min_colony_area_ratio: float = 0.00002
    max_colony_area_ratio: float = 0.015
    min_colony_circularity: float = 0.2
    nms_iou_threshold: float = 0.3

    # QC thresholds
    blur_threshold: float = 100.0
    underexposed_mean_threshold: float = 45.0
    overexposed_mean_threshold: float = 210.0
    plate_presence_score_threshold: float = 0.5

    # Counting quality thresholds (microbiology guidance range)
    min_recommended_colonies: int = 30
    max_recommended_colonies: int = 300

    # Validation thresholds
    edge_margin_ratio: float = 0.03
    merge_iou_threshold: float = 0.15
    overgrown_count_threshold: int = 500

    # Optional persistence schema flags
    postgres_enabled: bool = False
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/plate_count_ai"

    @property
    def device(self) -> str:
        """Select CUDA if available, otherwise CPU."""
        return "cuda:0" if torch.cuda.is_available() else "cpu"


settings = Settings()
