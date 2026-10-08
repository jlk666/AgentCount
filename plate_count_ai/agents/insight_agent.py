"""LLM-powered insight extraction for batch colony counting results."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from plate_count_ai.config.settings import Settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a scientific data analyst specialising in quantitative microbiology.
Your role is to interpret automated colony counting results from agar plate assays
and produce clear, concise, publication-quality scientific insights.
Be precise with numbers. Use scientific terminology appropriate for
Applied and Environmental Microbiology. Do not speculate beyond the data provided.
"""

_USER_TEMPLATE = """\
Below are colony counting results from an automated image analysis pipeline (AgentCount).
The data includes colony counts per replicate, CFU/mL estimates, and pairwise statistical tests.

--- SUMMARY RESULTS ---
{summary_json}

--- PAIRWISE t-TESTS (Colony Count, Welch's, α = 0.05) ---
{count_ttest_json}

--- PAIRWISE t-TESTS (CFU/mL, Welch's, α = 0.05) ---
{cfu_ttest_json}

--- TASK ---
Write a Results-section narrative (3–4 concise paragraphs) that:
1. Describes CFU/mL values and trends across sample groups (growth over time if applicable).
2. Highlights which pairwise comparisons are statistically significant (p < 0.05) and which are not, quoting the p-values.
3. Comments on within-group reproducibility (CV%, replicate consistency).
4. Concludes with a brief biological interpretation of the findings.

Rows marked TNTC or negative_control are quality/control observations, not numeric
CFU/mL results. Do not infer a time trend unless sampling times are provided.

Use past tense and passive voice where appropriate. Quote exact CFU/mL and p-values.
"""


class InsightAgent:
    """Call a locally-deployed Ollama LLM to generate scientific insights from batch results."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._generate_url = f"{settings.llm_base_url.rstrip('/')}/api/generate"

    def _is_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self.settings.llm_base_url.rstrip('/')}/api/tags",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _model_is_pulled(self) -> bool:
        try:
            resp = requests.get(
                f"{self.settings.llm_base_url.rstrip('/')}/api/tags",
                timeout=5,
            )
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            target = self.settings.llm_model
            return any(target == m or target.split(":")[0] == m.split(":")[0] for m in models)
        except Exception:
            return False

    def generate(
        self,
        summary_rows: list[dict[str, Any]],
        count_ttests: list[dict[str, Any]],
        cfu_ttests: list[dict[str, Any]],
    ) -> str:
        """
        Generate a scientific narrative from batch results.

        Returns the insight text, or an empty string with a logged warning if
        the Ollama service or model is unavailable.
        """
        if not any(row.get("count_status") == "countable_candidate" for row in summary_rows):
            return "No plates qualified as countable candidates; numeric CFU/mL comparisons are not supported."
        if not self.settings.llm_enabled:
            logger.info("LLM insight generation disabled (LLM_ENABLED=0).")
            return ""

        if not self._is_available():
            logger.warning(
                "Ollama service not reachable at %s — skipping LLM insights.",
                self.settings.llm_base_url,
            )
            return ""

        if not self._model_is_pulled():
            logger.warning(
                "Model '%s' is not available in Ollama — skipping LLM insights. "
                "Run: ollama pull %s",
                self.settings.llm_model,
                self.settings.llm_model,
            )
            return ""

        # Format numbers for the prompt — keep it compact but readable
        def _clean(rows: list[dict]) -> list[dict]:
            cleaned = []
            for r in rows:
                row = {}
                for k, v in r.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                    else:
                        row[k] = v
                cleaned.append(row)
            return cleaned

        prompt = _USER_TEMPLATE.format(
            summary_json=json.dumps(_clean(summary_rows), indent=2),
            count_ttest_json=json.dumps(_clean(count_ttests), indent=2),
            cfu_ttest_json=json.dumps(_clean(cfu_ttests), indent=2),
        )

        payload = {
            "model": self.settings.llm_model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        logger.info(
            "Requesting LLM insights from %s using model '%s' …",
            self.settings.llm_base_url,
            self.settings.llm_model,
        )
        try:
            resp = requests.post(
                self._generate_url,
                json=payload,
                timeout=self.settings.llm_timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            logger.info("LLM insight generation complete (%d chars).", len(text))
            return text
        except requests.Timeout:
            logger.warning(
                "LLM request timed out after %ds — skipping insights.",
                self.settings.llm_timeout,
            )
        except Exception as exc:
            logger.warning("LLM request failed: %s — skipping insights.", exc)
        return ""
