"""LangGraph workflow orchestration for plate counting agents."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from plate_count_ai.agents.cfu_calculation_agent import CFUCalculationAgent
from plate_count_ai.agents.colony_counting_agent import ColonyCountingAgent
from plate_count_ai.agents.colony_detection_agent import ColonyDetectionAgent
from plate_count_ai.agents.image_qc_agent import ImageQCAgent
from plate_count_ai.agents.plate_segmentation_agent import PlateSegmentationAgent
from plate_count_ai.agents.reporting_agent import ReportingAgent
from plate_count_ai.agents.validation_agent import ValidationAgent
from plate_count_ai.config.settings import Settings, settings
from plate_count_ai.models.colony_detector import ColonyDetector


class PlateWorkflowState(TypedDict, total=False):
    image_path: str
    metadata: dict[str, Any]
    processed_image: Any
    plate_image: Any
    raw_sam_masks: list[dict[str, Any]]
    raw_sam_overlay: Any
    raw_sam_mask_count: int
    tile_diagnostics: list[dict[str, Any]]
    detections: list[dict[str, Any]]
    colony_count: int
    count_status: str
    tntc_texture_fraction: float
    cfu_per_ml: float | None
    qc_status: dict[str, Any]
    validation_status: dict[str, Any]
    warnings: list[str]
    result: dict[str, Any]
    annotated_image_path: str
    sam_masks_image_path: str
    result_json_path: str
    summary_csv_path: str


class PlateWorkflow:
    """Build and execute the multi-agent LangGraph pipeline."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.settings = cfg or settings
        self.detector = ColonyDetector(
            model_path=self.settings.detector_model_path,
            device=self.settings.device,
            backbone=self.settings.sam_backbone,
            model_type=self.settings.sam_model_type,
            model_config=self.settings.sam_model_config,
            text_prompt=self.settings.sam_text_prompt,
            text_prompts_raw=self.settings.sam_text_prompts,
            enable_image_enhancement=self.settings.sam_enable_image_enhancement,
            checkpoint_url=self.settings.sam_checkpoint_url,
            auto_download_checkpoint=self.settings.auto_download_sam_checkpoint,
            points_per_side=self.settings.sam_points_per_side,
            pred_iou_thresh=self.settings.sam_pred_iou_thresh,
            stability_score_thresh=self.settings.sam_stability_score_thresh,
            min_mask_region_area=self.settings.sam_min_mask_region_area,
            min_colony_area_ratio=self.settings.min_colony_area_ratio,
            max_colony_area_ratio=self.settings.max_colony_area_ratio,
            min_colony_circularity=self.settings.min_colony_circularity,
            nms_iou_threshold=self.settings.nms_iou_threshold,
            enable_tiling=self.settings.sam_use_tiling,
            tile_size=self.settings.sam_tile_size,
            tile_overlap=self.settings.sam_tile_overlap,
        )

        self.qc_agent = ImageQCAgent(self.settings)
        self.segmentation_agent = PlateSegmentationAgent()
        self.detection_agent = ColonyDetectionAgent(self.detector)
        self.counting_agent = ColonyCountingAgent(self.settings)
        self.cfu_agent = CFUCalculationAgent()
        self.validation_agent = ValidationAgent(self.settings)
        self.reporting_agent = ReportingAgent(self.settings)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PlateWorkflowState)

        workflow.add_node("qc", self.qc_agent.run)
        workflow.add_node("segmentation", self.segmentation_agent.run)
        workflow.add_node("detection", self.detection_agent.run)
        workflow.add_node("counting", self.counting_agent.run)
        workflow.add_node("cfu", self.cfu_agent.run)
        workflow.add_node("validation", self.validation_agent.run)
        workflow.add_node("report", self.reporting_agent.run)

        workflow.set_entry_point("qc")
        workflow.add_edge("qc", "segmentation")
        workflow.add_edge("segmentation", "detection")
        workflow.add_edge("detection", "counting")
        workflow.add_edge("counting", "cfu")
        workflow.add_edge("cfu", "validation")
        workflow.add_edge("validation", "report")
        workflow.add_edge("report", END)

        return workflow.compile()

    def run(self, image_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
        initial_state: PlateWorkflowState = {
            "image_path": image_path,
            "metadata": metadata,
            "warnings": [],
        }
        return self.graph.invoke(initial_state)

    def run_batch(self, batch_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Batch processing helper.

        batch_items should contain dictionaries with:
        - image_path: str
        - metadata: dict
        """
        outputs: list[dict[str, Any]] = []
        for item in batch_items:
            outputs.append(self.run(item["image_path"], item["metadata"]))
        return outputs
