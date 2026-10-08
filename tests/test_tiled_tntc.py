"""Behavior checks for tiled detection and TNTC reporting."""

import unittest

import numpy as np

from plate_count_ai.agents.cfu_calculation_agent import CFUCalculationAgent
from plate_count_ai.agents.colony_counting_agent import ColonyCountingAgent
from plate_count_ai.config.settings import Settings
from plate_count_ai.models.colony_detector import ColonyDetector


class TiledDetectionTests(unittest.TestCase):
    def test_overlap_ownership_counts_one_object_once(self):
        detector = object.__new__(ColonyDetector)
        detector.tile_size = 1008
        detector.tile_overlap = 252
        detector.nms_iou_threshold = 0.3
        absolute_box = [700, 700, 730, 730]

        def fake_detect(tile):
            # All four tiles cover the same object. Each reports local coords.
            fake_detect.calls += 1
            x = 0 if fake_detect.calls in (1, 3) else 492
            y = 0 if fake_detect.calls in (1, 2) else 492
            local = [absolute_box[0] - x, absolute_box[1] - y,
                     absolute_box[2] - x, absolute_box[3] - y]
            return ([{"bbox": local, "confidence": 0.9}], [])

        fake_detect.calls = 0
        detector.detect_with_intermediate = fake_detect
        detections, raw_count, diagnostics, _ = detector.detect_tiled_with_intermediate(
            np.zeros((1500, 1500, 3), dtype=np.uint8)
        )
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(raw_count, 0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], absolute_box)


class CountStatusTests(unittest.TestCase):
    def test_dense_zero_count_is_tntc_and_has_no_cfu(self):
        rng = np.random.default_rng(5)
        noisy = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        counted = ColonyCountingAgent(Settings()).run({
            "detections": [], "plate_image": noisy,
            "metadata": {"dilution": 0.1, "volume": 0.1}, "warnings": [],
        })
        self.assertEqual(counted["colony_count"], 0)
        self.assertEqual(counted["count_status"], "TNTC")
        self.assertIsNone(CFUCalculationAgent().run(counted)["cfu_per_ml"])

    def test_negative_control_zero_is_not_tntc(self):
        blank = np.full((512, 512, 3), 128, dtype=np.uint8)
        counted = ColonyCountingAgent(Settings()).run({
            "detections": [], "plate_image": blank,
            "metadata": {"is_negative_control": True, "dilution": 1, "volume": 0.1},
            "warnings": [],
        })
        self.assertEqual(counted["count_status"], "negative_control")
        self.assertIsNone(CFUCalculationAgent().run(counted)["cfu_per_ml"])

    def test_dense_negative_control_is_flagged_for_review(self):
        noisy = np.random.default_rng(8).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        counted = ColonyCountingAgent(Settings()).run({
            "detections": [], "plate_image": noisy,
            "metadata": {"is_negative_control": True, "dilution": 1, "volume": 0.1},
            "warnings": [],
        })
        self.assertEqual(counted["count_status"], "negative_control")
        self.assertTrue(any("review for contamination" in w for w in counted["warnings"]))


if __name__ == "__main__":
    unittest.main()
