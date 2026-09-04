import logging

from chia.base.ChiaFunction import get
from chia.chipyard.verilator_run_node import VerilatorRunNode

from constants import VERILATOR_TIMEOUT_CYCLES, VERILATOR_TIMEOUT_SECONDS, VERILATOR_WORK_DIR
from dumper import Dumper

logger = logging.getLogger(__name__)


def verilator_run(dump: Dumper, attempt: int, artifact):
    node = VerilatorRunNode()
    logger.info("Running verilator (attempt %d)", attempt)
    run = get(
        node.run.options(resources={"chipyard": 1}).chia_remote(
            node,
            artifact,
            VERILATOR_WORK_DIR,
            timeout_cycles=VERILATOR_TIMEOUT_CYCLES,
            timeout_seconds=VERILATOR_TIMEOUT_SECONDS,
        )
    )
    dump.text(f"verilator_run_attempt{attempt}.log", run.log)
    dump.text(f"verilator_run_attempt{attempt}.out", run.out)
    return run
