import logging
import os
import subprocess
import sys

from chia.base.ChiaFunction import ChiaFunction, get
from chia.chipyard.chisel_build_node import ChiselBuildNode
from chia.chipyard.state_def import BuildTarget

from constants import (
    BUILD_CONFIG,
    BUILD_CONFIG_PACKAGE,
    CHIPYARD_DIFF_SUBMODULES,
    CHIPYARD_PATH,
    CHISEL_BUILD_MAKE_JOBS,
    CHISEL_BUILD_TIMEOUT_SECONDS,
)
from dumper import Dumper

logger = logging.getLogger(__name__)


@ChiaFunction(resources={"chipyard": 0.05})
def reset_chipyard(chipyard_path: str) -> str:
    """Reset the chipyard checkout to its committed baseline so each run's
    implement node starts from a clean tree (the container is reused across
    runs, so a previous run's accelerator + config edits would otherwise
    persist). Reverts tracked modifications (`git checkout -- .`) and removes
    new untracked, non-ignored files/dirs (`git clean -fd`); gitignored build
    artifacts are left in place. Operates on the root chipyard repo, where the
    accelerator, config wiring, and test live. Returns a short status string.
    """
    co = subprocess.run(
        ["git", "-C", chipyard_path, "checkout", "--", "."],
        capture_output=True, text=True,
    )
    cl = subprocess.run(
        ["git", "-C", chipyard_path, "clean", "-fd"],
        capture_output=True, text=True,
    )
    return (co.stdout + co.stderr + cl.stdout + cl.stderr).strip() or "clean"


def _untracked_diff(repo_path: str) -> str:
    """New-file diffs for untracked, non-ignored files in *repo_path*, read-only.

    `git diff` ignores untracked files, so brand-new sources (the freshly
    written MemCopyRoCC.scala / tests/memcpy.c) would be omitted. We surface
    them without touching the index: list them with `git ls-files --others
    --exclude-standard`, then diff each against /dev/null with
    `git diff --no-index` (which never reads or writes the index). ls-files does
    not descend into submodules, so a repo's scan won't pick up its submodules.
    """
    listed = subprocess.run(
        ["git", "-C", repo_path, "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    ).stdout.split()
    parts = []
    for f in listed:
        # --no-index exits 1 when files differ; we only consume stdout.
        r = subprocess.run(
            ["git", "-C", repo_path, "diff", "--no-index", "--", "/dev/null", f],
            capture_output=True, text=True,
        )
        if r.stdout:
            parts.append(r.stdout)
    return "".join(parts)


@ChiaFunction(resources={"chipyard": 0.05})
def collect_diff(
    chipyard_path: str,
    submodules: list[str] | None = None,
) -> tuple[int, dict[str, str]]:
    """Collect the chipyard git diff: tracked modifications AND new untracked
    files, for the root repo and each listed submodule.

    Read-only — nothing is staged, added, or committed (untracked files are
    surfaced via ``git diff --no-index``; see :func:`_untracked_diff`). Returns
    ``(error, diffs)`` where ``diffs`` maps repo path to diff text (key ``""``
    = root chipyard). ``error=1`` if a listed submodule is missing.
    """
    submodules = submodules or []
    for sm in submodules:
        if not os.path.exists(os.path.join(chipyard_path, sm, ".git")):
            return (1, {})

    diffs: dict[str, str] = {}
    root = subprocess.run(
        ["git", "-C", chipyard_path, "diff", "--ignore-submodules=all"],
        capture_output=True, text=True,
    ).stdout
    diffs[""] = root + _untracked_diff(chipyard_path)
    for sm in submodules:
        sm_path = os.path.join(chipyard_path, sm)
        tracked = subprocess.run(
            ["git", "-C", sm_path, "diff"], capture_output=True, text=True,
        ).stdout
        diffs[sm] = tracked + _untracked_diff(sm_path)
    return (0, diffs)


def collect_chisel_diff(dump: Dumper, attempt: int) -> None:
    """Capture the chipyard git diff for this iteration and dump it.

    Records the cumulative Chisel state that produced this attempt's build —
    after the implement node (attempt 0) and after each debug edit (later
    attempts). Writes the full per-repo diff dict as JSON plus the root
    chipyard diff as a readable .diff. Never raises: a diff-capture hiccup must
    not fail the build/run loop.

    collect_diff captures both tracked modifications and new untracked files
    (read-only) for the root + submodules, so the freshly written accelerator
    (MemCopyRoCC.scala, tests/memcpy.c) shows up without any staging.
    """
    try:
        err, diffs = get(
            collect_diff.options(resources={"chipyard": 0.01}).chia_remote(
                CHIPYARD_PATH, CHIPYARD_DIFF_SUBMODULES
            )
        )
    except Exception as e:  # noqa: BLE001 - diagnostic only
        logger.warning("collect_diff failed (attempt %d): %s", attempt, e)
        return
    if err:
        logger.warning("collect_diff returned error=%d (attempt %d)", err, attempt)
        return
    dump.json(f"chisel_diff_attempt{attempt}.json", diffs)
    # The accelerator + config + test edits live in the root chipyard repo
    # (key ""); surface it as a directly-readable .diff. Append any non-empty
    # submodule diffs below it.
    parts = []
    for repo, text in diffs.items():
        if not text:
            continue
        label = repo or "(chipyard root)"
        parts.append(f"# ===== diff: {label} =====\n{text}")
    dump.text(f"chisel_diff_attempt{attempt}.diff", "\n\n".join(parts))
    logger.info("Collected chisel diff (attempt %d): %d repo(s) changed",
                attempt, sum(1 for t in diffs.values() if t))


def chisel_build(dump: Dumper, attempt: int):
    """Build the target config; dump stdout/stderr; return the BuildArtifact."""
    node = ChiselBuildNode(
        chipyard_path=CHIPYARD_PATH,
        config=BUILD_CONFIG,
        config_package=BUILD_CONFIG_PACKAGE,
        target=BuildTarget.VERILATOR,
        make_jobs=CHISEL_BUILD_MAKE_JOBS,
        timeout_seconds=CHISEL_BUILD_TIMEOUT_SECONDS,
    )
    logger.info("Building %s (attempt %d)", BUILD_CONFIG, attempt)
    artifact = get(
        node.build.options(resources={"chipyard": 1}).chia_remote(node)
    )
    dump.text(f"chisel_build_attempt{attempt}.stdout.txt", artifact.stdout)
    dump.text(f"chisel_build_attempt{attempt}.stderr.txt", artifact.stderr)
    logger.info("Build %s (rc=%s)", "OK" if artifact.success else "FAILED", artifact.returncode)
    return artifact
