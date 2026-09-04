import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dumping
# ---------------------------------------------------------------------------

class Dumper:
    """Writes intermediate results into out_dir, prefixing each filename with
    the timestamp at write time (so files sort by when they were produced)."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dumping run artifacts to %s", self.out_dir)

    def _path(self, name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.out_dir / f"{ts}_{name}"

    def text(self, name: str, content: str) -> None:
        with open(self._path(name), "w", errors="replace") as f:
            f.write(content or "")

    def bytes(self, name: str, content: bytes) -> None:
        with open(self._path(name), "wb") as f:
            f.write(content or b"")

    def json(self, name: str, obj: dict) -> None:
        with open(self._path(name), "w") as f:
            json.dump(obj, f, indent=2)


def dump_llm(dump: Dumper, name: str, cli) -> None:
    """Persist an LLM call's final text + full stream transcript."""
    body = f"# {name}\n\nsuccess={getattr(cli, 'success', None)}\n\n"
    usage = getattr(cli, "usage", None)   # token/cost totals (opencode)
    if usage:
        body += f"usage={json.dumps(usage)}\n\n"
    body += (getattr(cli, "result", "") or "")
    stream = getattr(cli, "stream_result", "") or ""
    if stream:
        body += "\n\n## Stream transcript\n\n" + stream
    dump.text(f"{name}.md", body)
