from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Repo-relative paths (this package's own dir + the chia framework checkout)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_ENV = {
    "working_dir": str(PACKAGE_DIR),
    "py_modules": [
        str(_REPO_ROOT / "chia" / "chia"),
        str(_REPO_ROOT / "evolve-flows" / "skydiscover" / "skydiscover"),
        str(_REPO_ROOT / "alphaevolve-on-googlecloud" / "src" / "alpha_evolve"),
    ],
    "excludes": ["out/", "results/", "__pycache__", ".mypy_cache"],
}

# ---------------------------------------------------------------------------
# Prompts (loaded from prompts/, with ${VAR} placeholders substituted)
# ---------------------------------------------------------------------------
PROMPTS_DIR = PACKAGE_DIR / "prompts"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
OPENCODE_MODEL = "google-vertex/gemini-3.1-pro-preview"
LLM_SYSTEM_MESSAGE = (
    "You are an expert Chisel / RISC-V engineer specializing in RoCC "
    "accelerators for the Chipyard / Rocket / Gemmini ecosystem."
)
LLM_TIMEOUT_SECONDS = 1800
LLM_EXTRA_CLI_ARGS = ["--effort", "max"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_BASE = "./results"
CHIPYARD_PATH = "/home/ray/chipyard"
# Submodules collect_diff inspects in addition to the root chipyard repo. The
# accelerator + config edits land in generators/chipyard (the root repo, always
# captured under the "" key); these are included so a debugger edit that strays
# into a submodule (e.g. BOOM) is still recorded.
CHIPYARD_DIFF_SUBMODULES = [
    "generators/gemmini",
    "generators/rocket-chip",
    "generators/rocket-chip-inclusive-cache",
    "generators/rocket-chip-blocks",
]

VERILATOR_WORK_DIR = "/home/ray"
VERILATOR_TIMEOUT_CYCLES = 2000000
VERILATOR_TIMEOUT_SECONDS = 1800

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BUILD_CONFIG = "GemminiRocketConfig"
BUILD_CONFIG_PACKAGE = "chipyard"
CHISEL_BUILD_TIMEOUT_SECONDS = 60000
CHISEL_BUILD_MAKE_JOBS = 4

# ---------------------------------------------------------------------------
# Exo / BraggNN SW loop (see ~/repo/gemmini-braggnn/README.md)
# ---------------------------------------------------------------------------
GEMMINI_ROCC_TESTS_DIR = f"{CHIPYARD_PATH}/generators/gemmini/software/gemmini-rocc-tests"
GEMMINI_BAREMETAL_DIR = f"{GEMMINI_ROCC_TESTS_DIR}/bareMetalC"
GEMMINI_BUILD_DIR = f"{GEMMINI_ROCC_TESTS_DIR}/build"
BRAGGNN_BINARY = f"{GEMMINI_ROCC_TESTS_DIR}/build/bareMetalC/braggnn-baremetal"
VERILATOR_SIM_DIR = f"{CHIPYARD_PATH}/sims/verilator"
EXO_WORK_DIR = "/home/ray/exo_work"
EXO_TIMEOUT_SECONDS = 300
GEMMINI_BUILD_TIMEOUT_SECONDS = 1800
BRAGGNN_RUN_TIMEOUT_SECONDS = 1800

WRAPPER_LLM_MODEL = OPENCODE_MODEL
WRAPPER_LLM_TIMEOUT_SECONDS = 1800

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 3
