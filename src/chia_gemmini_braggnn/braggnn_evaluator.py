"""BraggnnEvaluator -- Exo -> wrapper LLM -> Gemmini build -> Verilator run.

See ~/repo/gemmini-braggnn/README.md for the underlying build/run commands
this shells out to.
"""

import contextvars
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from chia.base.ChiaFunction import ChiaFunction, get
from chia.models.opencode import OpenCodeLLM
from llm import gemini_vertex_provider
from result_mapper import map_braggnn_results
from skydiscover.evaluation.chia_evaluator import ChiaEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult

from constants import (
    BRAGGNN_BINARY,
    BRAGGNN_RUN_TIMEOUT_SECONDS,
    BUILD_CONFIG,
    EXO_TIMEOUT_SECONDS,
    EXO_WORK_DIR,
    GEMMINI_BAREMETAL_DIR,
    GEMMINI_BUILD_DIR,
    GEMMINI_BUILD_TIMEOUT_SECONDS,
    GEMMINI_ROCC_TESTS_DIR,
    VERILATOR_SIM_DIR,
    WRAPPER_LLM_MODEL,
    WRAPPER_LLM_TIMEOUT_SECONDS,
)
from prompts import _BRAGGNN_C_TEMPLATE, _BRAGGNN_H, _WRAPPER_SYSTEM_MESSAGE

# Per-task artifact so concurrent evaluate_program calls don't race.
_eval_binary: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_eval_binary", default=None,
)


@dataclass
class BraggnnBuildResult:
    success: bool
    binary_path: str
    exo_stdout: str
    wrapper_success: bool
    build_stdout: str
    build_stderr: str


@dataclass
class BraggnnRunResult:
    success: bool
    cycles: Optional[int]
    subpixel_error: Optional[float]  # gap between predicted and ground-truth
    stdout: str
    stderr: str


# TODO: exocc's output file convention (output dir / filename) is assumed
# here (-o work_dir, producing <stem>.c next to the source) -- verify
# against actual exocc behavior and adjust if it differs.
@ChiaFunction(resources={"chipyard": 0.2})
def _exo_compile(exo_program: str, work_dir: str) -> tuple:
    os.makedirs(work_dir, exist_ok=True)
    src_path = os.path.join(work_dir, "candidate.py")
    with open(src_path, "w") as f:
        f.write(exo_program)
    r = subprocess.run(
        ["exocc", "-o", work_dir, src_path],
        capture_output=True, text=True, cwd=work_dir,
        timeout=EXO_TIMEOUT_SECONDS,
    )
    c_path = os.path.join(work_dir, "candidate.c")
    generated_c = ""
    if os.path.exists(c_path):
        with open(c_path) as f:
            generated_c = f.read()
    return (r.returncode == 0 and bool(generated_c), generated_c, r.stdout + r.stderr)


@ChiaFunction(resources={"chipyard": 0.3})
def _gemmini_build(braggnn_c: str) -> BraggnnBuildResult:
    os.makedirs(GEMMINI_BAREMETAL_DIR, exist_ok=True)
    with open(os.path.join(GEMMINI_BAREMETAL_DIR, "braggnn.c"), "w") as f:
        f.write(braggnn_c)
    with open(os.path.join(GEMMINI_BAREMETAL_DIR, "braggnn.h"), "w") as f:
        f.write(_BRAGGNN_H)
    r = subprocess.run(
        [
            "make", "-C", "bareMetalC",
            "-f", f"{GEMMINI_BAREMETAL_DIR}/Makefile",
            f"abs_top_srcdir={GEMMINI_ROCC_TESTS_DIR}",
            f"src_dir={GEMMINI_BAREMETAL_DIR}",
            "XLEN=64",
            "PREFIX=examples-bareMetalC",
            "braggnn-baremetal",
        ],
        capture_output=True, text=True,
        cwd=GEMMINI_BUILD_DIR, timeout=GEMMINI_BUILD_TIMEOUT_SECONDS,
    )
    return BraggnnBuildResult(
        success=(r.returncode == 0 and os.path.exists(BRAGGNN_BINARY)),
        binary_path=BRAGGNN_BINARY,
        exo_stdout="",
        wrapper_success=True,
        build_stdout=r.stdout,
        build_stderr=r.stderr,
    )


# Matches braggnn.c's printed summary lines verbatim (braggnn.c:768,773-775):
#   "Avg cycles over %d runs: %llu"
#   "Avg error over %d runs: (X.XXXX, Y.YYYY)"
_CYCLES_RE = re.compile(r"Avg cycles over \d+ runs:\s*(\d+)")
_SUBPIXEL_ERROR_RE = re.compile(
    r"Avg error over \d+ runs:\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
)


@ChiaFunction(resources={"chipyard": 1})
def _verilator_run(binary_path: str) -> BraggnnRunResult:
    r = subprocess.run(
        ["make", f"CONFIG={BUILD_CONFIG}", "run-binary", f"BINARY={binary_path}"],
        capture_output=True, text=True, cwd=VERILATOR_SIM_DIR,
        timeout=BRAGGNN_RUN_TIMEOUT_SECONDS,
    )
    out = r.stdout + r.stderr
    cycles_m = _CYCLES_RE.search(out)
    err_m = _SUBPIXEL_ERROR_RE.search(out)
    subpixel_error = None
    if err_m:
        x_mae, y_mae = float(err_m.group(1)), float(err_m.group(2))
        # braggnn.c reports mean-|error| per axis, not a per-patch Euclidean
        # distance -- this combines the two axis MAEs into one scalar as a
        # proxy, not a mathematically identical mean distance.
        subpixel_error = (x_mae ** 2 + y_mae ** 2) ** 0.5
    return BraggnnRunResult(
        success=(r.returncode == 0 and cycles_m is not None),
        cycles=int(cycles_m.group(1)) if cycles_m else None,
        subpixel_error=subpixel_error,
        stdout=r.stdout,
        stderr=r.stderr,
    )


def _wrap_kernel(exo_c: str) -> str:
    """Single-shot, non-agentic LLM call: text in, text out -- no file-system
    tool access needed since the caller writes the result to disk itself."""
    llm = OpenCodeLLM(
        model=WRAPPER_LLM_MODEL,
        system_message=_WRAPPER_SYSTEM_MESSAGE,
        timeout_seconds=WRAPPER_LLM_TIMEOUT_SECONDS,
        logging_name="braggnn_wrapper",
        additional_providers=[gemini_vertex_provider(WRAPPER_LLM_MODEL)],
    )
    user_message = (
        f"Exo-compiled kernel code:\n```c\n{exo_c}\n```\n\n"
        f"Current braggnn.c:\n```c\n{_BRAGGNN_C_TEMPLATE}\n```"
    )
    result = get(
        llm.prompt.options(resources={"opencode_creds": 1}).chia_remote(
            llm, user_message, []
        )
    )
    return result.result if result.success else ""


class BraggnnEvaluator(ChiaEvaluator):
    """ChiaEvaluator subclass for the Exo -> Gemmini-C-API -> Verilator loop."""

    def __init__(self, artifact: Any, output_dir: str, **kwargs: Any) -> None:
        self._artifact = artifact
        super().__init__(
            build_fn=self._build,
            run_fn=self._run,
            result_mapper_fn=map_braggnn_results,
            workloads=["braggnn"],
            output_dir=output_dir,
            **kwargs,
        )

    def _build(self, program_solution: str) -> Any:
        ok, exo_c, exo_stdout = get(
            _exo_compile.options(resources={"chipyard": 0.2}).chia_remote(
                program_solution, EXO_WORK_DIR
            )
        )
        if not ok:
            return BraggnnBuildResult(
                success=False, binary_path="", exo_stdout=exo_stdout,
                wrapper_success=False, build_stdout="", build_stderr="",
            )

        braggnn_c = _wrap_kernel(exo_c)
        if not braggnn_c:
            return BraggnnBuildResult(
                success=False, binary_path="", exo_stdout=exo_stdout,
                wrapper_success=False, build_stdout="", build_stderr="",
            )

        return _gemmini_build.options(resources={"chipyard": 0.3}).chia_remote(braggnn_c)

    def _run(self, *, workload: str) -> Any:
        binary = _eval_binary.get()
        if binary is None:
            raise RuntimeError("No binary available -- build must succeed first")
        return _verilator_run.options(resources={"chipyard": 1}).chia_remote(binary)

    async def _dispatch_build(
        self,
        program_solution: str,
        label: str,
    ) -> Union[Any, EvaluationResult]:
        _eval_binary.set(None)
        result = await super()._dispatch_build(program_solution, label)

        if isinstance(result, EvaluationResult):
            return result
        if result is None:
            return None

        if not getattr(result, "success", False):
            return EvaluationResult(
                metrics={"error": 0.0, "combined_score": 0.0},
                artifacts={
                    "failure_stage": "build",
                    "error_type": "BuildFailure",
                    "stderr": result.build_stderr or result.exo_stdout,
                },
            )

        _eval_binary.set(result.binary_path)
        return result
