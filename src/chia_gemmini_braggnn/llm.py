import os

from chia.base.ChiaFunction import get
from chia.base.llm_call import QueryResult
from chia.base.tools.BashTool import BashTool
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM

from constants import LLM_SYSTEM_MESSAGE, LLM_TIMEOUT_SECONDS, OPENCODE_MODEL
from prompts import _DEBUGGER_PREAMBLE, _GEMMINI_TUNE


def gemini_vertex_provider(model: str = OPENCODE_MODEL) -> AdditionalModelProvider:
    provider_id, _, model_id = model.partition("/")
    return AdditionalModelProvider(
        id=provider_id or "google-vertex", npm="@ai-sdk/google-vertex",
        name="Google Vertex AI", models=[model_id],
        options={
            "project": os.environ.get("GCP_PROJECT"),
            "location": "global",
        },
    )


def make_llm(chipyard_bash: BashTool):
    """Build the implement/debug LLM ONCE, to be reused across the whole loop.

    Reusing a single instance is what lets ``ClaudeCodeLLM``'s
    ``@_session_tracked`` wrapper thread the session automatically: each
    ``get()`` syncs the transcript *and* advances the call counter onto this
    instance, so every ``debug`` call ``--resume``s the ``implement``
    conversation with no manual session bookkeeping. ``AntigravityLLM`` shares
    the same wrapper and result fields, so ``--llm antigravity`` threads one agy
    conversation the same way. (Session persistence for OpenCode is in
    development — each OpenCode call is independent — so reuse is just a
    convenience there; the failure context is re-supplied inline regardless.)
    """
    return OpenCodeLLM(
        model=OPENCODE_MODEL,
        system_message=LLM_SYSTEM_MESSAGE,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        logging_name="gemmini_tuner",
        additional_providers=[gemini_vertex_provider()],
        # Restrict opencode to ONLY the chipyard_bash MCP tool. Its built-in
        # write/edit/bash tools act on the opencode container's own FS, not
        # the chipyard container — deny all, allow just this MCP server.
        config={"*": "deny", f"{chipyard_bash.name}_*": "allow"},
    )


def _run_llm(llm, prompt: str, chipyard_bash: BashTool) -> QueryResult:
    """Dispatch *prompt* to *llm*'s worker (its backend's creds resource)."""
    resources = {"opencode_creds": 1}
    return get(
        llm.prompt.options(resources=resources).chia_remote(llm, prompt, [chipyard_bash])
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def implement(llm, chipyard_bash: BashTool) -> QueryResult:
    return _run_llm(llm, _GEMMINI_TUNE, chipyard_bash)


def debug(llm, chipyard_bash: BashTool, feedback: str) -> QueryResult:
    """Feedback call: diagnose and fix *feedback*. For claude this ``--resume``s
    the shared implement session (the reused instance carries the transcript +
    counter); for opencode the full failure context is inline in *feedback*."""
    return _run_llm(llm, f"{_DEBUGGER_PREAMBLE}\n\n{feedback}", chipyard_bash)
