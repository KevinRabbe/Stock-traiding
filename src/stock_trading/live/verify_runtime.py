from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime_state import runtime_verification_payload, verify_paper_shadow_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify persisted PAPER champion and SHADOW artifacts. Frozen SHADOW "
            "plugins are fully deserialized; the legacy V5 champion is manifest-"
            "verified and still requires injected calibration state at execution."
        )
    )
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = verify_paper_shadow_runtime(args.runtime_dir)
    print(json.dumps(runtime_verification_payload(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
