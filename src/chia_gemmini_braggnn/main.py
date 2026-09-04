import argparse

from constants import DEFAULT_OUTPUT_BASE
from flow import run_flow


# ── CLI ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_BASE,
        help="Results output directory (default: timestamped subdir under "
             f"{DEFAULT_OUTPUT_BASE})",
    )
    run_flow()


if __name__ == "__main__":
    main()
