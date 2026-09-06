import logging
import os
from typing import Optional

import ray

from braggnn_evaluator import BraggnnEvaluator
from evolve_flows.evolver.node import EvolverNode
from evolve_flows.evolver.types import EvolverInput

from constants import DEFAULT_OUTPUT_BASE, PACKAGE_DIR, RUNTIME_ENV
from prompts import _EXO_SEED

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = str(PACKAGE_DIR / "config_alphaevolve.yaml")
EVOLVER_ACTOR_NAME = "braggnn-evolver"
EVOLVER_NAMESPACE = "chia-gemmini-braggnn"
SEED_PROGRAM = _EXO_SEED


# ── Callable entry point (invoked once per outer HW attempt) ─────────

def run_alphaevolve_search(
    dump,
    attempt: int,
    artifact,
    initial_program: str,
    config_path: str = DEFAULT_CONFIG,
    output_dir: Optional[str] = None,
):
    """Run the inner SW (Exo) search against a fixed HW artifact.

    Called from flow.py's outer loop after a successful chisel_build.
    No own ray.init (caller already did it via RUNTIME_ENV), no CLI/signal
    handling (the outer loop owns the process), and a per-attempt actor
    name so repeated outer attempts don't collide with stale detached
    actors.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_BASE
    os.makedirs(output_dir, exist_ok=True)

    actor_name = f"{EVOLVER_ACTOR_NAME}-attempt{attempt}"

    try:
        stale = ray.get_actor(actor_name, namespace=EVOLVER_NAMESPACE)
        logger.warning(
            "Found stale evolver actor '%s' from a previous run -- killing it",
            actor_name,
        )
        ray.kill(stale)
    except ValueError:
        pass

    evaluator = BraggnnEvaluator(
        artifact=artifact,
        output_dir=output_dir,
        timeout=3600.0,
        max_retries=1,
    )

    logger.info("Creating EvolverNode actor '%s'", actor_name)
    evolver = EvolverNode.options(
        name=actor_name,
        namespace=EVOLVER_NAMESPACE,
        lifetime="detached",
        resources={"evolver": 1.0},
        runtime_env=RUNTIME_ENV,
    ).remote()

    with open(config_path) as f:
        config_content = f.read()
    config_content = config_content.replace(
        "GCP_PROJECT", os.environ.get("GCP_PROJECT", "")
    )
    config_content = config_content.replace(
        "GE_APP_ID", os.environ.get("GE_APP_ID", "")
    )

    evolver_input = EvolverInput(
        config_path=os.path.basename(config_path),
        initial_program=initial_program,
        config_content=config_content,
    )

    logger.info("Launching AlphaEvolve search for attempt %d", attempt)
    result_ref = evolver.run_search.remote(
        evolver_input,
        build_fn=evaluator._build,
        run_fn=evaluator._run,
        result_mapper_fn=evaluator.result_mapper_fn,
        evaluator=evaluator,
    )
    result = ray.get(result_ref)

    logger.info(
        "AlphaEvolve search finished (attempt %d): terminal_status=%s, iterations=%d",
        attempt, result.terminal_status, result.iteration_count,
    )
    if result.best_program:
        dump.text(f"best_exo_attempt{attempt}.py", result.best_program)

    evaluator.close()
    ray.kill(evolver)
    return result
