"""Targeted Stage Retry Manager for Universal Loop Engineering.

Implements targeted stage re-execution:
- Reruns ONLY the failed stage and minimum required dependent stages.
- Never restarts the entire meeting pipeline for downstream failures.
- Idempotent: safe upsert & record deletion prior to re-writing.
- Structured retry tracking with explicit correction strategy parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.loop_engine.quality_engine import QualityEngine, QualityGateResult
from backend.loop_engine.stage_manager import StageState, StageTracker

logger = logging.getLogger(__name__)


class TargetedRetryManager:
    """Manages stage-scoped targeted retries and dependency cascade execution."""

    def __init__(self, tracker: StageTracker, quality_engine: QualityEngine):
        self.tracker = tracker
        self.quality = quality_engine

    def execute_stage_with_retry(
        self,
        stage: str,
        execution_func: Callable[[], Any],
        validation_func: Callable[[Any], QualityGateResult],
        correction_strategy_func: Optional[Callable[[QualityGateResult, int], None]] = None,
    ) -> Tuple[Any, QualityGateResult]:
        """Executes a pipeline stage under quality control with targeted retries."""
        self.tracker.transition(stage, StageState.PROCESSING, "Stage execution started")

        result_data = None
        gate_result = QualityGateResult(
            passed=False, quality_score=0.0, status="FAILED", failure_reason="Not executed"
        )

        while True:
            try:
                result_data = execution_func()
                gate_result = validation_func(result_data)

                if gate_result.passed:
                    status = StageState.SUCCESS if gate_result.status == "SUCCESS" else StageState.DEGRADED
                    self.tracker.transition(
                        stage, status, f"Quality gate passed (score={gate_result.quality_score})"
                    )
                    return result_data, gate_result

                # Stage failed quality gate check
                failure_reason = gate_result.failure_reason or "Quality validation failed"
                failed_val = gate_result.failed_validation or "quality_gate"
                strategy_desc = f"Targeted retry for {stage} ({failed_val})"

                if not self.tracker.can_retry(stage):
                    logger.warning(f"[TargetedRetryManager] Stage '{stage}' exhausted max retries.")
                    self.tracker.transition(
                        stage,
                        StageState.FAILED,
                        f"Failed after max retries. Reason: {failure_reason}",
                    )
                    return result_data, gate_result

                # Apply correction strategy callback if available
                attempt = self.tracker.record_retry_attempt(
                    stage=stage,
                    failure_reason=failure_reason,
                    failed_validation=failed_val,
                    correction_strategy=strategy_desc,
                    result="RETRYING",
                )

                if correction_strategy_func:
                    try:
                        correction_strategy_func(gate_result, attempt)
                    except Exception as corr_err:
                        logger.warning(f"[TargetedRetryManager] Correction callback error on {stage}: {corr_err}")

            except Exception as exc:
                logger.error(f"[TargetedRetryManager] Exception during stage '{stage}': {exc}", exc_info=True)
                gate_result = QualityGateResult(
                    passed=False,
                    quality_score=0.0,
                    status="FAILED",
                    failure_reason=f"Runtime exception: {type(exc).__name__}: {str(exc)}",
                    failed_validation="runtime_exception",
                )
                if not self.tracker.can_retry(stage):
                    self.tracker.transition(stage, StageState.FAILED, str(exc))
                    return None, gate_result

                self.tracker.record_retry_attempt(
                    stage=stage,
                    failure_reason=str(exc),
                    failed_validation="exception",
                    correction_strategy="exception_retry",
                    result="RETRYING",
                )
