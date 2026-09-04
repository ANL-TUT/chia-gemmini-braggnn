from constants import PROMPTS_DIR


def _load_prompt(name: str, **subs: str) -> str:
    """Read prompts/<name> and replace ${KEY} placeholders with subs values.

    ``${KEY}`` (rather than str.format's ``{KEY}``) so prompt text can contain
    literal braces (Chisel/Scala snippets) without escaping.
    """
    text = (PROMPTS_DIR / name).read_text()
    for key, val in subs.items():
        text = text.replace("${" + key + "}", val)
    return text


_GEMMINI_TUNE = _load_prompt(
        "gemmini.md",
)

_DEBUGGER_PREAMBLE = _load_prompt(
        "debugger.md",
)

_EXO_SEED = _load_prompt(
        "exo.py",
)

_BRAGGNN_C_TEMPLATE = _load_prompt(
        "braggnn.c",
)

_BRAGGNN_H = _load_prompt(
        "braggnn.h",
)

_WRAPPER_SYSTEM_MESSAGE = (
    "You wrap Exo-compiled kernel code into the BraggNN Gemmini C API test "
    "harness. You are given the current braggnn.c (the harness: main(), the "
    "multi-pass MEASURE_KERNEL counter instrumentation, quantization "
    "pre/post-processing) and freshly Exo-compiled kernel C code. Replace "
    "gemmini_inference()'s body with calls into the new kernel code, adapting "
    "the MEASURE_KERNEL / kernel-name-table plumbing (K_CONV1..K_OUTPUT, "
    "NUM_KERNELS) to match whatever kernel boundaries the new schedule "
    "actually has (fused kernels mean fewer MEASURE_KERNEL calls). Leave "
    "everything else in the file unchanged. Respond with the complete, "
    "updated braggnn.c and nothing else."
)
