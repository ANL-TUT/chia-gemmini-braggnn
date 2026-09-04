"""Result mapper: BraggnnRunResult list -> EvaluationResult.

combined_score = -cycles (AlphaEvolve maximizes score; latency should be
minimized), gated by sub-pixel error: a fast-but-wrong candidate scores 0
rather than winning on latency alone.
"""

from typing import Any, Dict, List

from skydiscover.evaluation.evaluation_result import EvaluationResult

# Reject candidates whose predictions are off from ground truth by more
# than this many sub-pixels.
MAX_ACCEPTABLE_SUBPIXEL_ERROR = 0.5


def map_braggnn_results(run_results: List[Any]) -> EvaluationResult:
    run = run_results[0]

    if not getattr(run, "success", False) or run.cycles is None:
        return EvaluationResult(
            metrics={"error": 0.0, "combined_score": 0.0},
            artifacts={"failure_stage": "run", "stderr": run.stderr},
        )

    subpixel_error = run.subpixel_error if run.subpixel_error is not None else float("inf")
    if subpixel_error > MAX_ACCEPTABLE_SUBPIXEL_ERROR:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "cycles": float(run.cycles),
                "subpixel_error": subpixel_error,
            },
            artifacts={"failure_stage": "accuracy_gate"},
        )

    metrics: Dict[str, float] = {
        "combined_score": -float(run.cycles),
        "cycles": float(run.cycles),
        "subpixel_error": subpixel_error,
    }
    return EvaluationResult(metrics=metrics, artifacts={})
