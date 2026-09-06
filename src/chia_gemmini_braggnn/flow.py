import logging
import os
from datetime import datetime
from typing import Optional

import ray

from chia.base.ChiaFunction import get
from chia.base.tools.BashTool import BashTool

from alphaevolve_flow import SEED_PROGRAM, run_alphaevolve_search
from chipyard_ops import capture_chisel_baseline, chisel_build, collect_chisel_diff, reset_chipyard
from constants import CHIPYARD_DIFF_SUBMODULES, CHIPYARD_PATH, DEFAULT_OUTPUT_BASE, MAX_ATTEMPTS, RUNTIME_ENV
from dumper import Dumper, dump_llm
from llm import debug, implement, make_llm

logger = logging.getLogger(__name__)


# ── Flow entry point ─────────────────────────────────────────────────
def run_flow(
    output_dir: Optional[str] = None,
) -> None:
    ray.init(address="auto", runtime_env=RUNTIME_ENV)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if output_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(DEFAULT_OUTPUT_BASE, f"run_{run_tag}")

    dump = Dumper(output_dir)
    logger.info("Output directory: %s", output_dir)

    # ----- reset chipyard to baseline (before chipyard_bash claims the
    # node's full "chipyard" resource) ------------------------------------
    logger.info("Resetting chipyard checkout to baseline")
    reset_out = get(
        reset_chipyard.options(resources={"chipyard": 0.01}).chia_remote(CHIPYARD_PATH)
    )
    logger.info("Chipyard reset: %s", reset_out or "clean")

    baseline = get(
        capture_chisel_baseline.options(resources={"chipyard": 0.01}).chia_remote(
            CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES
        )
    )

    chipyard_bash = BashTool(
        name="chipyard_bash",
        work_dir=CHIPYARD_PATH,
        timeout_seconds=300,
        task_options={"resources": {"chipyard": 1}},
    )

    logger.info("Creating LLM Node")
    llm = make_llm(chipyard_bash)

    feedback = None
    run = None
    exo_seed = SEED_PROGRAM
    for attempt in range(MAX_ATTEMPTS):
        if feedback is None:
            impl = implement(llm, chipyard_bash)
            dump_llm(dump, "implement", impl)
            logger.info("Implement finished (success=%s)", impl.success)
        else:
            impl = debug(llm, chipyard_bash, feedback)
            dump_llm(dump, f"debug_attempt{attempt}", impl)
            logger.info("Debug finished (attempt %d, success=%s)", attempt + 1, impl.success)

        chipyard_bash.stop()

        collect_chisel_diff(dump, attempt, baseline)
        artifact = chisel_build(dump, attempt)

        if not artifact.success:
            logger.error("Build failure (attempt %d)", attempt + 1)
            feedback = (
                f"chisel_build failed (attempt {attempt + 1}):\n\n"
                f"STDOUT:\n{artifact.stdout}\n\nSTDERR:\n{artifact.stderr}"
            )
            if attempt + 1 < MAX_ATTEMPTS:
                chipyard_bash = BashTool(
                    name="chipyard_bash",
                    work_dir=CHIPYARD_PATH,
                    timeout_seconds=300,
                    task_options={"resources": {"chipyard": 1}},
                )
            continue

        run = run_alphaevolve_search(dump, attempt, artifact, exo_seed, output_dir=output_dir)

        if run.terminal_status != "error" and run.best_program:
            exo_seed = run.best_program
            logger.info("Loop finished (success, attempt %d)", attempt + 1)
            return run

        logger.error("AlphaEvolve search failure (attempt %d)", attempt + 1)
        feedback = (
            f"alphaevolve search failed (attempt {attempt + 1}):\n\n"
            f"{run.error_message}"
        )
        if attempt + 1 < MAX_ATTEMPTS:
            chipyard_bash = BashTool(
                name="chipyard_bash",
                work_dir=CHIPYARD_PATH,
                timeout_seconds=300,
                task_options={"resources": {"chipyard": 1}},
            )

    logger.info("Loop finished (exhausted %d attempts)", MAX_ATTEMPTS)
    return run
